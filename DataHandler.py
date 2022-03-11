"""Defines the generic Metadata class for market fundamentals and parameters, as well as the data import and
pre-analysis.

For a quickstart guide, call DataHandler.help()

Metadata class:
	This is a generic container that bundles actual data and its metadata (i.e. information necessary to understand
	the actual data). The actual data can be read from the disk or passed directly upon instantiation,
	and the metadata is passed.

	It is highly recommended to automate data import and have metadata stored on the disk, to be passed upon
	instatiation (e.g. via a config file).

	Apart from bundling data and metadata, this class also implements methods for time series data, which extend the
	time series functionality of Pandas.


INTRODUCTION TO UNITS IN METADATA:
																										11.19.2019
	Metadata supports unit declarations (optional). This is done via a UnitHandler, which can be any package/module
	for handling explicit units in Python. This was designed with flexibility in mind, so that you could use your
	preferred package*. In the second implementation of unit handling (Nov 19, 2019), only Pint 0.9 is supported.

	*NOTE: Make sure that this is supported in the following places:
		- DataHandler.set_UnitHandler(), which performs the initialization of the system of units (incl. unit
		definitions)
		- Metadata.__parse_units(), a subroutine of __init__() which parses the units parameter and binds the
		processed units object to self.units.

	Nearly every units-related feature is specific to the UnitHandler used. However, Metadata units methods are
	defined such that every UnitHandler is given control to how certain tasks are done (e.g. how to form a quantity,
	i.e. a number * units). Therefore, Metadata-level units methods are used in the same way, but inner workings are
	UnitHandler-specifc.

	Pls. note that unit declarations is NOT the same as having explicit units -- the underlying data structures need
	not necessarily have explicit units. The units are explicitly defined in self.units instead. Thus, it is up to
	the application to check self.units and self.val.

	If you don't use the unit declaration feature, arguments to units while instantiating Metadata would simply be
	bound to self.units (without any checks). The first time you do this will issue a warning as a reminder,
	which can be suppressed by setting Metadata.opt['warn_no_units'] = False.


HANDLING UNITS WITH Pint 0.9:
																											11.19.2019
	To enable units in Metadata, you set the UnitHandler via set_UnitHandler:

	>> dh.set_UnitHandler('Pint 0.9')
	Successfuly set Pint 0.9 as unit handler.

	set_UnitHandler() will perform the necessary initialization procedure, which you can influence by passing
	UnitHandler-specific keyword arguments (pls. see help(set_UnitHandler) for more info). After calling
	set_UnitHandler(), Metadata can now parse the units parameters.

	If you need to perform additional actions on the UnitHandler, Metadata comes with the ff. auxiliary functions:

	>> dh.Metadata.UnitHandler()   # Check the set UnitHandler
	'Pint 0.9'

	>> pint = dh.Metadata.UnitHandler(get_namespace=True)   # Retreive the UnitHandler namespace

	>> dh.get_UHaux('ureg') # Get the resources of UnitHandler
	<pint.registry.UnitRegistry at 0x249a874df88>

	>> dh.new_UHaux('Q_', ureg.Quantity)


	You could only set the UnitHandler once per session. Attempting to call set_UnitHandler() again would raise the
	RunTimeError, preventing mixed UnitHandlers per use of DataHandler.



On time index:
																											11.11.2019
	The original implementation requires that time-oriented data structures have the Pandas DatetimeIndex,
	partially because this was the first understood time-index I understood. Upon learning about the other time
	indices (specifically, the PeriodIndex), I firstly considered whether other time indices should be supported. My
	current answer to this is NO, because:
		a) the Timestamp is the most general time-index -- It represents one moment in time and overlaps with the
		Period except in the duration interpretation of the Period (think about the explicit record of time:value).

		b) The very reason we implemented the preliminary time analysis is under the assumption that the time series
		data can have a messy sampling of time -- which is exactly what the DatetimeIndex is for. The PeriodIndex can
		be seen as a subset of this, wherein the sampling of time is ideal.

		c) .tsloc(), with 'ts' inspired by Timestamp, is appropriately named, that it assumes that the underlying
		index is a DatetimeIndex.

	I'm not fully closing the door on supporting the PeriodIndex -- prelim time analysis has to be upgraded because
	this can be cut short. Furthermore, .tsloc() would be incompattible, so perhaps a .perloc() can be implemented (
	not sure if this makes sense).


Author:         Pang Teng Seng, D. Kayanan
Create date:    Sep. 18, 2019
Last update:    Mar. 10, 2022
Version:        2.0
Release date:   TBA
"""
# Python
from os import path
import warnings
import logging

# Python 3rd parth
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt




def __defexceptions():
	"""Contains custom exceptions, just to clean the top-level module."""
	global DataHandlerError, MetadataOptionsError, NaNError, SignificantSubResError, TimegapError, MaxTimeLagError, \
		TimeIndexError, UnitHandlerError, UndefinedUnitHandler, Pint09Error

	class DataHandlerError(Exception):
		"""Base class wherein all defined exceptions in DataHandler are derived. Not to be raised directly."""
		pass

	class MetadataOptionsError(DataHandlerError):
		"""Raised when the options in Metadata.opt are incompatible, as detected by the constructor. A description of the
		problem is issued."""
		pass

	class NaNError(DataHandlerError):
		"""Raised when data being initialized illegally contains NaN values, as restricted by Metadata.opt['allow_nan']."""
		pass

	class SignificantSubResError(DataHandlerError):
		"""Raised when time-oriented data being initialized illegally have a significant sub-resolution time step,
		which is indicative of a change in time resolution at some point. This is restricted by
		Metadata.opt['allow__subres']."""
		pass

	class TimegapError(DataHandlerError):
		"""Raised when time-oriented data being initialized illegally have time gaps. This is restricted by
		Metadata.opt['allow__timegaps']."""
		pass

	class MaxTimeLagError(DataHandlerError):
		"""Raised when the detected resolution in time-oriented data is higher than the argument passed to max_time_lag.
		This is an ambiguous case and must be diagnosed. If max_time_lag is not used, this implies that the data is
		strictly discrete and this issue is not applicable."""
		pass

	class TimeIndexError(DataHandlerError):
		"""Raised when there are issues with the index of time-oriented data. The specific issue must be described."""
		pass

	class UnitHandlerError(DataHandlerError):
		"""Base class of exceptions triggered by unit handling"""
		pass

	class UndefinedUnitHandler(UnitHandlerError):
		"""Raised when the passed unit handler to set_UnitHandler() is not define."""
		pass

	class Pint09Error(UnitHandlerError):
		"""Raised when Pint 0.9-specific errors are encountered."""
		pass
	return


def help(version='short'):
	"Prints a quick start guide."
	if version=='short':
		print("""Quickstart:
	1) Check and adjust the read options of Metadata as dh.Metadata.opt**
	2) Declare the unit handler via set_UnitHandler() (Optional but highly recommended)***
	3) Set the read directory via set_readpath()
	4) Pick up the data as Metadata! dh.Metadata(...)*
	5) Perform auxiliary functions (e.g. time analysis, basic plotting, etc.)

For more info, use DataHandler.help('more')
For info on auxiliary functions, use DataHandler.help('aux')
	""")
	elif version=='more':
		print("""Typical and most convenient usage of DataHandler (imported as dh) is:
	
	1) Check and adjust the read options of Metadata as dh.Metadata.opt**
	2) Declare the unit handler via set_UnitHandler() (Optional but highly recommended)***
	3) Set the read directory via set_readpath()
	4) Pick up the data as Metadata! dh.Metadata(...)*
	5) Perform auxiliary functions (e.g. time analysis, basic plotting, etc.)
	
	*The most convenient way to store the data are via the Pandas-readable forms (csv, pickled Series/DataFrame, 
	Excel). All Metadata data structures can be stored in one of these file formats. 
	
	**Metadata does some preliminary checks on the read data, in the hopes of identify if there is something 
	fundamentally wrong with the data. Currently, the checks are for the time-aspects of time-oriented data. NaN 
	check is also implemented.
	
	***As of 12.2019, the only implemented Unit Handler is Pint 0.9. In this Unit Handler, the declared units are 
	checked but the underlying data (attr .val) are not Pint quantities. However, __getitem__ of the parameters 
	dstruct converts the read parameter into a Pint quantity right before returning it.
	
Metadata was designed to:

	1) Read (clean) data from the disk, and attach metadata to it (i.e. additional information to help anyone 
	use the data). Its main philosophy is that there should always be Metadata in any data file stored.
	
	2) Perform simple checks at read and additional reading features. Currently, these checks are mostly for 
	time-oriented data structures, which are necessary to allow certain features (e.g. time must be strictly 
	ascending for .tsloc() to work). These are for convenience and are incomplete; users are expected to know how to 
	use the data properly. 

Last update: 06.12.19
	""")
	elif version == 'aux':
		print("""Metadata time-oriented methods:
			calc_timestep_dist()
			get_common_period()
		
		Auxiliary functions:
			Sers_toDF()
			askwargs()
		""")

	return


class Metadata():
	"""
	This is a generic data class that bundles the actual data structure with metadata. It also supports additional
	methods for time-oriented data structures (as extensions to Time Series functionality in Pandas). It defines the
	ff. data structures:

		STRUCTURE                   ACCEPTS
		'parameters'                dictionary or Series of {name: numeric/string/bool}
		'series'                    Any of the sequence types. Numpy arrays and Pandas Series are recommended.
		'table'                     Pandas DataFrame (axis cannot be a time series-oriented index)

		TIME-ORIENTED STRUCTURES
		'time series'               Pandas Series with a DatetimeIndex*
		'time table'                Pandas DataFrame with a DatetimeIndex*

		*Required to be in ascending order. Restricted to this index type for now, but might be easily extensible to
		the other time series-oriented indices of Pandas.


	ATTRIBUTES:
			defn_short  (str) Brief description of the data.
			defn_full   (str) Complete description of the data.

			dstruct     (str) Data structure. One of 'parameters', 'series', 'time series', 'table' and 'time table'.

			units       (units object/dict {keys (str): units object}) Units of the quantity(ies), where units object
						is the units type of the UnitHandler. If the entire dataset has a common unit, only a single
						units argument is needed. Otherwise, provide a dictionary of the data identifier : unit (for
						heterogeneous parameters and tables).

			units_checked   (Bool). Set to True if UnitHandler verified the passed units.

			shape       Similar to panda's shape attr for Series and DataFrame, this is a tuple representing the
						number of data points. For the table structures, this will be a duple of (rows, columns). For
						everything else, it will be a single tuple of the number of items.

			val         The actual data structure.

			time_params     A dictionary containing parameters relevant to the time-oriented data structures. For other
						types, this is None (doubles as a check if time-oriented due to this dichotomy). Pls. see the
						next section for details.

			nans        Percentage of the data in Pandas-based data structures ('series', 'time series', 'table' and
						'time table'). For the 'parameters' structure, the current implementation avoids this case
						and sets it to None (with a warning if Metadata.opt['allow_nan'] is False).

			more_md     Stands for 'more metadata,' it is a dictionary containing any and all additional information
						users would like to store. If not used, would be left as an empty dict.


	TIME-ORIENTED ATTRIBUTES:
		Time-oriented data structures would have a dictionary in self.time_params with the ff. key, values:

		'max_time_lag'      For time-oriented data structures, this is a pandas Timedelta that allows a missing value
							for a given time to be filled in by the closest previous value, if it is within
							max_time_lag. This is useful for handling concurrent time series  but possibly with
							different resolutions (e.g. daily and weekly data). This behavior can be disabled by
							simply passing '0s', (the default behavior of the constructor) making the data strictly
							discrete.

		't_start', 't_end'  End times (i.e. first and last indices)

		'resolution'        The resolution of a time-oriented data set is defined here as the mode of the time steps
							(type pd.Timedelta). Note that there might be time steps smaller (sub-resolution time steps)
							and larger (i.e. data gaps) than this. Restrictions on the presence of these anomalies are
							controlled by the class options in Metadata.opt.

		'sub res %'         Percentage occurrence of time steps smaller than the resolution (for all sub-res time
							steps, regardless of significance).

		'time gaps %'       Percentage occurence of time gaps in the data. For discrete data, time gaps are defined
							as gaps larger than the detected resolution. For continuous data, time gaps are gaps larger
							than max_time_lag.

		'discrete'          Bool, equivalent to max_time_lag == pd.Timedelta('0s'). A discrete data set strictly doesn't
							have data in between data points.

	PARAMETERS dtype:
		The parameters dtype supports indexing as:

		myprms = Metadata(...)
		myprms[key]

		This would yield a quantity (i.e. value*units). They way that the quantity is produced depends on
		UnitHandler. If you want the underlying data (which could just be a pure numeric), you can always access it
		via the 'val' attribute (e.g. myprms.val).


	CLASS OPTIONS:
		All options are collected in the public class attribute Metadata.opt dictionary.

		Metadata.opt = {
		'warn_no_units'     : (Bool; defaults to True) Used to suppress the no units defined warning.

		'allow_timegaps'    : (Bool; defaults to False) If False, time-oriented data with gaps (i.e. gaps larger than
							the computed resolution* cannot be instantiated (raises TimeGapError)**. A time gap is
							defined as a time step > max(resolution, max_time_lag).

		'allow_subres'      : (Bool; defaults to False) If False, time-oriented data with significant*** time steps
							smaller than the computed resolution* cannot be instantiated (raises MultiResError)**.

		'warn_subres'       : (Bool; defaults to True) Relevant only when allow_subres=True. Attempts to instantiate
							time-oriented data with significant time steps smaller than the resolution* would proceed
							but a warning would be issued. Suppress this at your own risk.

		'subres_th'         : (percentage; defaults to 0.1) A time step whose % of occurrences larger than this value
							is considered significant.


		'allow_nan'         : (Bool; defaults to False) If False, data cannot have NaNs. If True, this makes the
							definition of data gaps ambiguous, so use at your own risk.
		}

		*Pls. see self.time_params['resolution'].
		**Metadata currently supports NO data cleaning. By raising exceptions, users are encouraged to do the cleaning
		themselves.
		***As defined by Metadata.opt['subres_th']


	NOTE on sub-resolution, time gaps and NaNs:

		Sub-resolution time steps and time gaps are data imperfections that can be forbidden by 'allow_subres' and
		'allow_timegaps', respectively. The only difference is that sub-res steps that occur less frequent than the
		threshhold 'subres_th' are always tolerated, even if allow_subres=False. Furthermore, if any is forbidden,
		then NaN's in the data must also be forbidden (otherwise, this is ambiguous, and it's simpler to forbid NaNs).


	CLASS PRIVATE ATTRIBUTES:
		__readpath          Points to the directory containing your data file for whenever you need to read from the
							disk. Not to be accessed directly. Pls. use DataHandler.set_readpath() instead.

		__DefinedUnitHandlers       Tuple of implemented UnitHandler (string keys)
		__UnitHandler               UnitHandler namespace bound here
		__UnitHandler_aux           UnitHandler resources bound here. Can be appended, but existing keys cannot be
									remapped.
		__UnitHandler_key           UnitHandler string name

		*_structures                Are tuples of data structure string keys (grouped in different categories,
									and should be self-explanatory). __data_structures contains all defined data
									structures.


	More on TABLES:

		Tables and time tables can be homogeneous or heterogeneous. A homogeneous table is a collection of parallel
		Series, and thus have a single unit. Conversely, a heterogeneous table is composed of different quantities,
		and each must have a declared unit (as a dict).


	CLASS METHODS:
		UnitHandler()           Prints UnitHandler's name or returns the UnitHandler namespace.
		get_UHaux()             Querries the auxiliary items (packaged as a dict) of the UnitHandler.
		new_UHaux()             Adds a new object to the UnitHandler auxiliary items. Overwriting existing keys is
								not permitted.
		toqty()                 Returns the passed numeric data as a quantity (as defined by UnitHandler).

	TIME-ORIENTED METHODS:
		get_period()

		tsloc()

		calc_res()              A method used to determine if
	"""
	# Metadata options (public attribute)
	opt = {
		'warn_no_units':    True,
		'allow_timegaps':   False,
		'allow_subres':     False,
		'warn_subres':      True,
		'subres_th':        0.1,
		'allow_nan':        False,
	}
	# Todo: add option to mix in units for methods that querry values (e.g. tsloc). Need to implement a hidden
	# instance method that returns a quantity (i.e. numeric w/ units; UnitHandler-specific)

	# private attributes
	__readpath = ''
	__DefinedUnitHandlers = ('Pint 0.9', )
	__UnitHandler = None
	__UnitHandler_aux = {}
	__UnitHandler_key = None


	# Metadata data structure definitions
	__time_data_structures = ('time series', 'time table')
	__data_structures = ('parameters', 'series', 'table') + __time_data_structures
	__dictlike_structures = ('parameters', 'table', 'time table')
	__pandas_structures = ('series', 'table') + __time_data_structures

	assert set(__time_data_structures).issubset(__data_structures)
	assert set(__dictlike_structures).issubset(__data_structures)
	assert set(__pandas_structures).issubset(__data_structures)


	def __init__(self, defn_short, dstruct, units, values=None, max_time_lag='0s', filename=None, subdir=None,
	             reader='infer', defn_full="", more_md=None, report='full', **reader_kwgs):
		"""
		Initialize the metadata container.

		PARAMETERS
		a) Basic parameters:
			defn_short      (str) defn_short attribute
			defn_full       (str; optional) defn_full attribute
			dstruct         (str) dstruct attribute
			units           Units object coercible or dict {key: Units object coercible} (heterogeneous dict-like
							data structure, where the units object can be accepted by UnitHandler, and the keys match
							the keys of the data.


		b) Conditionally required parameters:

			b1) Must use one of values or filename:

			values          If the data is directly available (i.e. does not need to be read from the disk), then pass
							via this parameter.

			filename        (str) If the data is to be read from the disk, pass the filename here (including the
							extension).

			b2) If filename is used, then:

			Metadata.__readpath     *NOT A PARAMETER* Pls. set the class attribute Metadata.__readpath to the directory
									containing your data.

			subdir          (str path; optional) For added flexibility, you can specify a relative location from
							Metadata.__readpath via this parameter.

			reader          (callable; optional) The function used to read the file from disk. The default behavior
							is to choose one of Pandas' .read_csv(), .read_excel() and .read_pickle() based on the
							file extension.

			reader_kwgs     Arguments for the reader function, to be passed as a dictionary of the arguments.



			b3) If dstruct is 'time series' or 'time table':

			max_time_lag    (optional; defaults to '0s') This parameter is used only for the time series and time
							table data structures; pls. pass a valid argument to pandas.Timedelta() > 0s.

							When accessing data at a specific time, and no entry exists for that time, then the
							closest previous value would be returned if and only if it is within the specified period of
							max_time_lag.

							By setting a max_time_lag, you make a discrete time series/table into a continuous one,
							by holding the previous value up to max_time_lag (quasi zero-order hold). This
							feature is used when dealing with multiple time series of possibly different sampling
							rates (e.g. days vs weeks) and/or you are working on a finer time step than the sampling
							of your data.

							The default behavior is to allow no lag, as in the default behavior when indexing time
							series in Pandas. This means that the data is strictly discrete.

							If max_time_lag is used, then the data has gaps iff:
								pd.Series(data.index).diff().max() > pd.Timedelta(max_time_lag)

		c) Other parameters:

			more_md          (optional; defaults to empty dict) Set the instance.more_md dictionary to this.

			report           (Str; defaults to 'full') Controls the way __init__() reports upon a successful execution:

							report='full'       Opening statements and __repr__() are printed to the console.

							report='brief'      Just the read file or that values were read is reported (single-line).

							report='log'        Writes the report='brief' message to the specified log file. If this
												is used, it is recommended that the logger must have been initialized
												by other Python modules.

		USAGE:
			1) Reading the file from disk or via argument
				- via disk: Pass the name of the file to param filename, and make sure param path is set properly.
				Set the reader function. The current implementation works if the first argument of the reader is the
				path to the file. Pass additional reader parameters via reader_kwgs.

				- via direct argument: Pass the data to param values.


		"""
		# todo Note: As Md usu. reads from disk, there is no type checking of the underlying val (types are expected
		#  based on the reader used)
		_opt = Metadata.opt
		# Safety check during development
		assert dstruct in ('parameters', 'series', 'time series', 'table', 'time table')

		# --------------------------------------------------------------------------- 0a) Parameter checks
		if dstruct not in Metadata.__data_structures:
			raise ValueError("Invalid argument passed to 'dstruct'. Pls. pass one of: {}".format(", ".join(
				Metadata.__data_structures)))

		if (filename is None) == (values is None):
			raise ValueError("Must use only one of parameters filename (to read a file) and values (to pass the data "
			                 "directly).")

		# Check string params
		if not isinstance(defn_short, str): raise TypeError("Pls. pass a string for parameter 'defn_short'.")
		if not isinstance(defn_full, str):  raise TypeError("Pls. pass a string for parameter 'defn_full'.")

		if more_md is not None and not isinstance(more_md, dict): raise TypeError("Pls. pass dictionary for parameter "
		                                                                        "'more_md'.")


		# --------------------------------------------------------------------------- 0b) Check Metadata settings
		# Check time controls (if consistent restrictions on subres + time gaps w/ NaNs)
		if _opt['allow_nan'] and not(_opt['allow_timegaps'] and _opt['allow_subres']):
			raise MetadataOptionsError("If any of time gaps or significant sub resolution is not allowed, "
			                           "then NaN values should also be forbidden. Pls. adjust Metadata.opt "
			                           "accordingly.")

		# Check the subres threshhold value
		if not 0 <= _opt['subres_th'] < 100:
			raise MetadataOptionsError("Option 'subres_th' must be a percentage within [0, 100)")
		if 5 < _opt['subres_th']:
			warnings.warn("Recommended subres_th < 5; otherwise, this loses meaning.")

		# Check if Metadata.__readpath has not been set (and Metadata is instructed to read from disk)
		if Metadata.__readpath == '' and values is None:
			raise RuntimeError("Pls. first set the read path via DataHandler.set_readpath()")

		# ----------------------------------------------------------------------------- 0c) Parse reader
		if reader == 'infer' and filename is not None:
			rstripped = filename.rsplit(sep='.', maxsplit=1)
			if len(rstripped) == 1:
				raise ValueError("Invalid filename given. There is no extension.")

			if rstripped[1] == 'pkl':
				reader = pd.read_pickle
			elif rstripped[1] == 'csv':
				reader = pd.read_csv
			elif rstripped[1] in ('xlsx', 'xls', 'xlsm'):
				reader = pd.read_excel
			else:
				raise NotImplementedError("File type: {} is not supported.".format(rstripped[1]))

		if filename is not None and not callable(reader):
			raise ValueError("The reader parameter is not callable.")

		# ----------------------------------------------------------------------------- 1) Basic attributes
		self.defn_short = defn_short
		self.defn_full = defn_full
		self.dstruct = dstruct
		if more_md is None:
			self.more_md = {}
		else:
			self.more_md = more_md

		# --------------------------------------------------------------------- 2) Bind values + simple check on values
		#                                                                        - check NaNs, measure shape
		# ---------------------------------------- 2.1a) Read values from disk
		if filename:
			# Fix target path
			if subdir:
				targetpath = path.join(Metadata.__readpath, subdir, filename)
			else:
				targetpath = path.join(Metadata.__readpath, filename)

			# Read
			if any(reader is pdfunc for pdfunc in (pd.read_csv, pd.read_excel, pd.read_pickle)):
				self.val = reader(targetpath, **reader_kwgs)
				# Todo - for 'parameters' data struct, read as csv via pd.read_csv() but convert to dict
			else:
				raise NotImplementedError

		# ---------------------------------------- 2.21b) Values passed (assumes only one of two modes)
		else:
			self.val = values

		# ---------------------------------------- 2.2) Measure shape
		try:
			self.shape = self.val.shape
		except AttributeError:
			self.shape = (len(self.val),)

		# ---------------------------------------- 2.3) Check NaNs
		if isinstance(self.val, dict):
			self.nans = pd.Series(self.val).sum() / np.prod(self.shape) * 100
		elif isinstance(self.val, (pd.Series, pd.DataFrame)):
			self.nans = self.val.isna().sum().sum() / np.prod(self.shape) * 100
		elif isinstance(self.val, np.ndarray):
			self.nans = np.isnan(self.val).sum() / np.prod(self.shape) * 100
		else:
			raise NotImplementedError('NaN check for values type is not yet implemented.')


		if not _opt['allow_nan'] and self.nans > 0:
			raise NaNError("Detected illigel NaNs ({:0.2f} %) in '{}' .".format(self.nans, self.defn_short))


		# ----------------------------------------------------------------------------- 3) Time params
		if self.dstruct in Metadata.__time_data_structures:
			self.__prelim_time_analysis(max_time_lag)
		else:
			self.time_params = None

		# ----------------------------------------------------------------------------- 4) Check units
		self.__parse_units(units, report)

		# ----------------------------------------------------------------------------- 5) Done -- REPORT
		# .......................................................... Prep msg string
		if filename is not None:
			msg_start = "Read file: {}".format(filename)
		else:
			msg_start = "Read values"

		if self.units_checked:
			units_check_str = " (units checked)"
		else:
			units_check_str = ""

		brief_msg = "{} successfully{}.".format(msg_start, units_check_str)

		# .......................................................... a) report='full'
		if report == 'full':
			print("{}{}, \n\n\t{}\n".format(msg_start, units_check_str, "\n\t".join(self.__repr__().split("\n"))))

		# .......................................................... b) report='log'
		elif report == 'log':
			logging.info(brief_msg)
		# .......................................................... c) report='brief' (fallback)
		else:
			print(brief_msg)

		return


	def __prelim_time_analysis(self, max_time_lag):
		"""Subroutine of __init__. This performs the preliminary analysis of time-oriented data structures:
			- analyze index (correct type + inc time)
			- measure resolution (verifies max_time_lag as well)
			- check sub res time steps
			- check time gaps

		Sets self.time_params
		"""
		_opt = Metadata.opt
		# ----------------------------------------------------------------------- 0) Cast max_time_lag to pd.Timedelta
		max_time_lag = pd.Timedelta(max_time_lag)
		zero_s = pd.Timedelta('0s')

		if max_time_lag < zero_s:
			raise ValueError("max_time_lag should be a non-negative Timedelta")

		# ----------------------------------------------------------------------- 1) Check index type
		if not isinstance(self.val.index, pd.DatetimeIndex):
			raise TypeError("The time-oriented data structures currently only support the pandas.DatetimeIndex.")

		# ----------------------------------------------------------------------- 2) Measure the resolution
		# a) Calculate the distribution of time steps (also verifies that time is inc)
		#    Note that time must be inc for Metadata.tsloc() to work.
		#    Time is never automatically sorted, to make this explicit to user.

		# Series of time step : % occurrence
		#resdist = self.calc_timestep_dist()
		resdist = calc_timestep_dist(self)

		# b) Resolution = mode of time steps
		res = resdist.loc[resdist == resdist.max()].index[0]

		# c) Check that max_time_lag > res
		# if max_time_lag == zero_s (i.e. discrete), then this issue is not applicable
		if zero_s < max_time_lag < res:
			raise MaxTimeLagError("The detected time resolution in '{}' is higher than the passed "
			                      "max_time_lag. This makes the use of max_time_lag vague.".format(self.defn_short))


		# ----------------------------------------------------------------------- 3) Check sub-resolution time steps
		# 3.1 detect
		subres_all = resdist.loc[resdist.index < res]
		subres_sig = subres_all.loc[resdist >= _opt['subres_th']]

		# 3.2 react
		if not _opt['allow_subres'] and len(subres_sig) > 0:
			raise SignificantSubResError("Detected significant sub-resolution time steps in '{}'. This might "
			                             "indicate a change in resolution.".format(self.defn_short))
		elif _opt['warn_subres'] and len(subres_sig) > 0:
			warnings.warn("Detected significant sub-resolution time steps in '{}'. Attempt to continue "
			              "instantiation.".format(self.defn_short))


		# ----------------------------------------------------------------------- 4) Check time gaps
		# 4.1 detect
		# Two possible case:
		#   a) max_time_lag = 0s    Discrete data, and time gaps are steps > res
		#   b) max_time_lag >= res  Time gaps are steps > max_time_lag
		timegaps = resdist.loc[resdist.index > max(res, max_time_lag)]

		# 4.2 react
		if not _opt['allow_timegaps'] and len(timegaps) > 0:
			raise TimegapError("Detected time gaps in '{}'.".format(self.defn_short))


		# ----------------------------------------------------------------------- 5) Set time parameters
		self.time_params = {
			'max_time_lag'  : max_time_lag,
			't_start'       : self.val.index[0],
			't_end'         : self.val.index[-1],
			'resolution'    : res,
			'sub res %'     : subres_all.sum(),
			'time gaps %'   : timegaps.sum(),
			'discrete'      : max_time_lag == zero_s,
			# todo
		}
		return


	def __parse_units(self, units, report):
		"""Subroutine of __init__. Parses the units argument according to the set UnitHandler.

		Note: units is a dict {key : units} for dict-like structures. Otherwise, just a single argument. The unit
		arguments are no longer required to be a string, because this is up to UnitHandler to handle.

		PSEUDOCODE:
			1) Checks + set dictlike and units_is_dict (Bool)
				- dict-like datastructures must have {key: units str}, and that keys match the underlying data (can be
				parameters (dict), table or time table (Pandas DataFrame).
				- non-dict-like simple type check

			2) See if units are defined and set self.units
				- UnitHandler-specific
				- Exceptions encountered would be re-raised after issuing more info messages.
				- Impt: this uses dictlike bool from earlier, in order to properly parse parameter units
					e.g. if you want to collect the unit strings as an iterable, you could do:
						if dictlike:
							passedunits = set(units.values())
						else:
							passedunits = set((units,))
		"""
		# ----------------------------------------------------------------------------- 1) set bools & checks
		dictlike = self.dstruct in Metadata.__dictlike_structures
		units_is_dict = isinstance(units, dict)
		# ................................................................... For dict-like structure
		if dictlike:
			# If dict-like structure, we have two cases:
			#   a) heterogeneous: a dictionary is needed, and each key in the data has a corresponding key: unit
			#   b) homogeneous: a single unit can be passed

			if units_is_dict and set(units.keys()) != set(self.val.keys()):
				raise ValueError("For the heterogeneous dict-like structures, must pass a dictionary {key : units} to "
				                 "the 'units' parameter, where the keys match those in the data.")


		# ----------------------------------------------------------- 2) Attempt to define units and bind to self.units
		#                                                                                        (UnitHandler-specific)
		# ................................................................... a) PINT 0.9
		if Metadata.__UnitHandler_key == 'Pint 0.9':
			# iterate over passedunits and define units objects via ureg(*).units
			ureg = Metadata.get_UHaux('ureg')
			try:
				# todo - if there are more UnitHandlers, maybe this if-else can be abstracted
				if units_is_dict:
					self.units = {key: ureg(val).units for key, val in units.items()}
				else:
					self.units = ureg(units).units

			except Exception as err:
				msg = "Failed to parse units ({}): {}\n".format(units, err)
				print('ERROR: {}'.format(msg))

				if report == 'log':
					logging.error(msg)
				raise

		# ................................................................... z) NO UNIT HANDLER
		else:
			if Metadata.opt['warn_no_units']:
				warnings.warn("The unit handler has not been set. Unit arguments would simply be bound to self.units")
				Metadata.opt['warn_no_units'] = False

			self.units = units

		self.units_checked = Metadata.__UnitHandler is not None
		return


	def __getitem__(self, item):
		if self.dstruct == 'parameters':
			if isinstance(self.units, dict):
				units = self.units[item]
			else:
				units = self.units
			return Metadata.toqty(self.val[item], units)
		else:
			raise TypeError('Metadata {} type is not subscriptable.'.format(self.dstruct))


	def __repr__(self):
		# Prepare definition representation
		defn_width = 80
		defn_str = ""

		for idx in range(0, len(self.defn_short), defn_width):
			substr = self.defn_short[idx:idx + defn_width]

			# todo dash line break (a bit tricky)
			if idx == 0:
				defn_str += "Def'n:\t\t{}\n".format(substr)
			else:
				defn_str += "\t\t{}\n".format(substr)

		# Prepare the units representation. If a single unit, direct print. If dict, then fancy table.
		if isinstance(self.units, dict):
			#units_print = "\nunits:\n" + "\n".join("\t{:<30}[{}]".format(key, {'': '-'}.get(val, val))
			#                                       for key, val in self.units.items())
			maxl = min(30, max(len(key) for key in self.units.keys())+7)
			joinstr = "\t{{:<{maxl}}}[{{}}]".format(maxl=maxl)
			units_print = "\nunits:\n" + "\n".join(joinstr.format(key, {'': '-'}.get(val, val))
			                                       for key, val in self.units.items())

		else:
			units_print = "units:\t\t[{}]".format({'': '-'}.get(self.units, self.units))

		# Get period and res for time-oriented
		if self.dstruct in Metadata.__time_data_structures:
			#time_info = "\n" + self.get_period(get_str=True)
			time_info = "\n{period_info} \n\nRes:\t{res_info}\n".format(
				period_info=self.get_period(get_str=True),
				res_info={None: 'n/a'}.get(self.time_params['resolution'], self.time_params['resolution'])
			)
		else:
			time_info = ''

		return "{}\ndstruct:\t{} \nshape:\t\t{}\n{}\n{}\n".format(defn_str, self.dstruct, self.shape, units_print,
		                                                          time_info)


	def defn(self):
		"""Prints the brief and full definition."""
		print(self.defn_short)

		# defn_width = 100
		# defn_str = ""
		# for idx in range(0, len(self.defn_full), defn_width):
		# 	defn_str += "\t{}\n".format(self.defn_full[idx:idx + defn_width])
		# print("\nFull definition:\n{}".format(defn_str))
		print("\nFull definition:\n\n{}".format(self.defn_full))
		return

	# Priority
	# TODO function to get longest common time for time-oriented structures

	# todo methods for series to table and vice-versa
	# todo method to get min common time for time series table (ignoring data gaps in between for now)


	# ------------------------------------------ Time-Oriented Methods ----------------------------------------------- #
	def get_period(self, get_str=False):
		"""Convenience function to give basic information on the time index for time-oriented data structures. The
		default behavior is to print the info, but the same string can be returned by setting get_str=True."""
		if self.dstruct not in Metadata.__time_data_structures:
			raise TypeError("This method can only be called for the time series and time table data structures.")

		msg = "From: \t{}\nTo: \t{}\n({})\n".format(self.time_params['t_start'], self.time_params['t_end'],
		                                    self.time_params['t_end']-self.time_params['t_start'])
		if get_str:
			return msg
		else:
			print(msg)


	def tsloc(self, ts, col=None):
		"""This method is a modified single-timestamp indexing method in Pandas, which allows a previous data point
		to be returned if the specified timestamp does not exist, if and only if it is within self.time_params[
		'max_time_lag']. It assumes that the time index is increasing (as is normal for time series data),
		which is not checked here but in the initialization of the Metadata instance instead.

		PARAMETERS:
			ts      Pandas Timestamp coercible or Pandas Period
			col     (Optional; needed if underlying data is DataFrame) Column name

		By having a time lag:
			1) You allow your time-oriented data to return values in between samples (within max_time_lag),
			converting a discrete time series into a continuous one.
			2) You extend the reach of your data to your last entry + max_time_lag

		Returns:
			Data that matches ts (as possibly extended by self.time_params['max_time_lag']. If there is no match,
			then a KeyError is raised.

		"""
		# ................................................................... a) Type checks
		if self.dstruct not in Metadata.__time_data_structures:
			raise TypeError("This method can only be called for the time series and time table data structures.")

		if not isinstance(self.val.index, pd.DatetimeIndex):
			raise TypeError("This method can only operate on the Pandas DatetimeIndex.")

		# ................................................................... b) Pre-process ts
		# Serves to validate ts, AND to ensure that a dimension is eliminated in the querry (i.e. Series --> 1 value;
		# DataFrame --> 1 row)
		try:
			if isinstance(ts, pd.Period): # Test first if Period!
				ts = ts.to_timestamp()
			elif not isinstance(ts, pd.Timestamp):
				ts = pd.Timestamp(ts)
		except Exception as err:
			err.args = ("Could not convert arg '{}' into a pandas Timestamp", "Original message: {}".format(err.args))
			raise

		# ................................................................... c) Get Series if DataFrame
		if isinstance(self.val, pd.DataFrame):
			if col is None:
				raise TypeError("The underlying data is a DataFrame; thus, pls. specify the column via parameter "
				                "'col'.")
			targetSer = self.val[col]

		elif isinstance(self.val, pd.Series):
			targetSer = self.val
		else:
			raise NotImplementedError("Only supports pandas Series and DataFrame.")


		# --------------------------------------------------------------------------- CASE 1: Sample available
		try:
			return targetSer[ts]

		# --------------------------------------------------------------------------- CASE 2: Try to fetch previous val
		except KeyError:                                                   # (has to be within time param max_time_lag)
			if self.time_params['max_time_lag'] == pd.Timedelta('0s'):
				raise KeyError("Data is strictly discrete, and querried time {} is not available.".format(ts))

			try:
				# TODO - DEC TIME - it will be index 0 if it were dec
				prev_ts = targetSer.index[targetSer.index < ts][-1]
				# idx -1 fetches the latest (cause time index is ascending)
			except IndexError:
				raise KeyError("Querried time: {} is in the past of the data.".format(ts))

			if ts - prev_ts <= self.time_params['max_time_lag']:
				return targetSer[prev_ts]
			else:
				if prev_ts == targetSer.index[-1]:
					raise KeyError("Querried time: {} is too far into the future.".format(ts))
				else:
					raise KeyError("Querried time: {} has no available data.".format(ts))
		# End tsloc

	# ------------------------------------------ Other Methods ----------------------------------------------- #
	def plot(self, plotter='plot', **kwargs):
		"""Basic plotting feature.

		KWARGS for axes methods:
			figsize
			grid
			xlabel
			ylabel
			title
			legend
			xlim
			ylim

		KWARGS for plotters:
			kwarg       plotter
			where       step


		"""
		if plotter not in ('plot', 'step'):
			raise ValueError
		plt.figure(figsize=kwargs.get('figsize', (8, 6)))

		# ............................................................................ Plot per type
		if isinstance(self.val, (pd.Series, np.ndarray)):
			if plotter == 'plot':
				plt.plot(self.val, label=self.val.name)
			elif plotter == 'step':
				plt.step(self.val.index, self.val, where=kwargs.get('where', 'pre'))

		elif isinstance(self.val, pd.DataFrame):
			for colname, Ser in self.val.iteritems():
				if plotter == 'plot':
					plt.plot(Ser, label=colname)
				else:
					raise NotImplementedError

			kwargs['legend'] = True
		else:
			raise NotImplementedError('Basic plotting is not supported for the underlying data type.')

		# ............................................................................ polishing
		ax = plt.gca()

		if kwargs.get('grid', True):
			ax.grid()

		ax.set_ylabel(kwargs.get('ylabel', str(self.units)))
		ax.set_xlabel(kwargs.get('xlabel'))
		ax.set_title(kwargs.get('title'))


		ax.set_xlim(kwargs.get('xlim'))
		ax.set_ylim(kwargs.get('ylim'))

		if kwargs.get('legend'):
			ax.legend()
		plt.show()
		return

	# ------------------------------------------ UnitHandler Methods ----------------------------------------------- #
	@classmethod
	def get_UHaux(cls, *keys: str):
		"""Reads the private class attribute __UnitHandler_aux dictionary and returns the bound values. These are
		resources to control the set UnitHandler. Can pass multiple keys (positionally). The key:val pairs are
		UnitHandler-defined."""
		if len(keys) == 1:
			return cls.__UnitHandler_aux[keys[0]]
		else:
			return tuple(cls.__UnitHandler_aux[key] for key in keys)


	@classmethod
	def new_UHaux(cls, key, value):
		"""Adds a new key to the private class attribute __UnitHandler_aux dictionary. This allows new objects to be
		bound, as needed. Re-writing existing keys is not allowed."""
		if key in cls.__UnitHandler_aux:
			raise UnitHandlerError("Cannot overwrite existing keys.")
		cls.__UnitHandler_aux[key] = value
		return


	@classmethod
	def UnitHandler(cls, get_namespace=False):
		"""Checks the declared UnitHandler. Pass get_namespace=True if you want to get the namespace of the
		UnitHandler"""
		if get_namespace:
			return cls.__UnitHandler
		else:
			print(cls.__UnitHandler_key)

	@classmethod
	def toqty(cls, numeric, unit):
		"""Forms a quantity object (quantity = numeric x unit)."""
		if cls.__UnitHandler_key == 'Pint 0.9':
			# Proxy for checking if Pint quantity
			if hasattr(numeric, 'dimensionality') and hasattr(numeric, 'to_reduced_units'):
				raise TypeError('Must pass a numeric object, and not a Pint quantity.')
			return numeric*unit
		else:
			raise NotImplementedError


# ------------------------------------------------ END Class Metadata  ----------------------------------------------- #

# --------------------------------------------- Metadata external methods -------------------------------------------- #
def set_readpath(readpath):
	"""Convenience function to set the read path of class Metadata."""
	if not path.isdir(readpath): raise NotADirectoryError("Pls. pass an existing directory.")
	Metadata._Metadata__readpath = readpath
	return

# .............................................. UnitHandler ......................................................... #
def set_UnitHandler(handler: str, get_namespace=False, silent=False, **kwargs):
	"""Sets the unit handler of Metadata. This can only be done once per use of DataHandler.py at run time.

	Supported Unit Handlers:
		1) Pint 0.9
		2) No unit handler (binds passed arguments in units to self.units without checks)

	The UnitHandler namespace would be bound to Metadata. There are two ways to get the namespace:
		1) Apply get_namespace=True,
			pint =  set_UnitHandler('Pint 0.9', get_namespace=True)

		2) Retrieve it sometime afterwards,
			pint = Metadata.UnitHandler(get_namespace=True)

	....................................................................................................................
														Pint 0.9
	....................................................................................................................
		Additional parameters pass via keyword:
			own_defnfile        Use another definitions file (in lieu of the default). Argument to UnitRegistry() at
								ureg initialization.

			extend_defnfile     After ureg initialization (whether by loading the default or using another),
								this definition file is read to extend the initialized definitions. Argument to
								ureg.load_definitions()

			direct_defn         Pass definitions expressed as strings here. Argument to ureg.define()

			get_ureg            (Bool). If True, returns the initialized UnitRegistry. Furthermore, get_namespace is
								ignored if ureg is returned.



		Bindings to Metadata
		{
			'ureg': UnitRegistry(),
		}

		Notes:
			Programmatically extending the definitions (via ureg.define()) can be done by accessing ureg as
				ureg = get_UHaux('ureg')
				ureg.define(...)

	....................................................................................................................
	Set silent=True to omit the printed message.

	Return:
		The namespace of the unit handler, which is the same bound to Metadata._Metadata__UnitHandler
	"""
	if Metadata._Metadata__UnitHandler is not None:
		raise RuntimeError("Metadata unit handler has already been set.")
	if handler not in Metadata._Metadata__DefinedUnitHandlers:
		raise UndefinedUnitHandler
	auxmsg = ''

	# ------------------------------------------------------------------------------------ 1) Pint 0.9
	if handler == 'Pint 0.9':
		# ............................................................... a) Import and check version
		try:
			from pint import __version__

			if __version__ != '0.9':
				raise NotImplementedError("Imported a different version of Pint.")

		except ModuleNotFoundError:
			print("Pls. install Pint 0.9. More info can be found here: https://pint.readthedocs.io/en/0.9/index.html")
			raise

		import pint
		# ............................................................... b) Init Pint
		try:
			# b1) Init ureg
			if 'own_defnfile' in kwargs:
				ureg = pint.UnitRegistry(kwargs['own_defnfile'])
				auxmsg += '\n\t+non-default definition file'
			else:
				ureg = pint.UnitRegistry()

			# b2) Extend definitions
			if 'extend_defnfile' in kwargs:
				ureg.load_definitions(kwargs['extend_defnfile'])
				auxmsg += '\n\t+additional definitions from file'

			if 'direct_defn' in kwargs:
				ureg.define(kwargs['direct_defn'])
				auxmsg += '\n\t+additional definitions from str'

		except Exception as e:
			print("Pint initialization failed. Caught exception can be accessed in the first argument of Pint09Error.")
			raise Pint09Error(e)

		# ............................................................... c) Bindings to Metadata
		Metadata._Metadata__UnitHandler = pint
		Metadata._Metadata__UnitHandler_aux = {
			'ureg':     ureg,
			'Q_':       ureg.Quantity,
			'ureg.Quantity':        ureg.Quantity,
			'DimensionalityError':  pint.DimensionalityError,
		}

	# ----------------------------------------------------------------------------------------------------------- #
	if not silent:
		print('Successfuly set {} as unit handler.{}'.format(handler, auxmsg))
	Metadata._Metadata__UnitHandler_key = handler


	if kwargs.get('get_ureg', False):
		return ureg # get_namespace is ignored!
	if get_namespace:
		return Metadata._Metadata__UnitHandler


# .................................... Methods for Time-Oriented Data Structures .................................... #
def get_common_period(*timedata, report='print'):
	"""This function calculates the common time period of the Metadata time-oriented arguments (ignores data gaps,
	if any). Pass the multiple data as variable-positional parameters. If there is a common time, return (t_start,
	t_end). Otherwise, returns (None, None).

	Parameter report can be set to 'print' (prints to stdout) or 'log' (uses the logging module).
	"""
	if any(inst.dstruct not in Metadata._Metadata__time_data_structures for inst in timedata):
		raise TypeError("This method can only be called for the time series and time table data structures.")

	if len(timedata) < 2:
		raise RuntimeError("Pls. pass at least two time-oriented Metadata instances.")

	# --------------------------------------------------------------- 1) Get time ends for each argument.
	# Indexed positionally as in the arguments.
	# ts data = t_start;    te data = t_end + max_time_lag
	ts = pd.Series(inst.time_params['t_start'] for inst in timedata)
	te = pd.Series(inst.time_params['t_end']+inst.time_params['max_time_lag'] for inst in timedata)

	max_duration = (te-ts).max()
	# todo if you want a table of the time end points, make it from ts, te and te-ts

	# --------------------------------------------------------------- 2) Get the latest start + earliest end
	_ts = ts.sort_values(ascending=False)[0:1]
	ts_idx, ts_latest = _ts.index[0], _ts.iloc[0]

	_te = te.sort_values(ascending=True)[0:1]
	te_idx, te_earliest = _te.index[0], _te.iloc[0]

	# --------------------------------------------------------------- 3) Compare
	if ts_latest <= te_earliest:
		duration = te_earliest-ts_latest

		if duration/max_duration < 0.5:
			msg = "Detected a small overlap compared to available data. At least one argument will have " \
			      "{:0.2f}% of its data unused.".format((1-duration/max_duration)*100)

			warnings.warn(msg)
			if report == 'log': logging.warning(msg)

		msg = "Common period found\nFrom: \t{}\nTo: \t{}\n({})".format(ts_latest, te_earliest, duration)
		t_common =  (ts_latest, te_earliest)
	else:
		msg = "No common period found. Argument {} finishes before {} starts.".format(te_idx, ts_idx)
		t_common = (None, None)

	# --------------------------------------------------------------- 4) Return
	if report == 'print':
		print(msg)
	elif report == 'log':
		logging.info(msg)

	return t_common


def calc_timestep_dist(Md_inst=None, val=None):
	"""
	Calculates the distribution of time steps in time-oriented data. This has two usages:
		1) Internally in Metadata.__init__()
			- pass the instance (partially instantiated) via Md_inst=self
			- Used internally to determine the resolution, sub-resolution gaps and time gaps during
			Metadata.__prelim_time_analysis().

		2) Externally
			- pass the DataFrame / Series to val
			- Can be used externally, because it doesn't alter any attributes.

	Returns a Series of:

		dt_count = pd.Series({
			time step : % occurrence
		})

	Note: This also verifies that the time index is increasing.
	"""
	zero_s = pd.Timedelta('0s')

	# Check if usage is internal or external
	if val is None:
		if not isinstance(Md_inst, Metadata):
			raise TypeError("Pls. pass a Metadata instance to Md_inst.")
		if Md_inst.dstruct not in Metadata._Metadata__time_data_structures:
			raise TypeError("This method can only be called for the time series and time table data structures.")

		date_idx = Md_inst.val.index

	elif Md_inst is None:
		if not isinstance(val, (pd.Series, pd.DataFrame)):
			raise TypeError("Pls. pass a Pandas Series or DataFrame to val.")

		date_idx = val.index
	else:
		raise ValueError("Incorrect parameter usage.")

	# Calc delta time's
	dts = pd.Series(date_idx).diff()
	# Get rid of the na for the first data point
	dts = dts[pd.notnull(dts)]

	# Verify that time is inc -- no auto sorting is done here, to make this explicit
	# TODO - DEC TIME - either *-1 to dts or make it dts < zero_s for all. Take care of the implications of
	#  supporting both inc and dec time, esp. in .__prelim_time_analysis() and .tsloc(). This has not been
	#  extensively studied (05.12.19)
	if not (dts > zero_s).all():
		raise TimeIndexError("Time-oriented data structures must be INC in time. Currently, dec time is not allowed, "
		                     "because .tsloc() would not work.")

	# After .value_counts(), now indexed as time step : no of occurence
	dt_count = dts.value_counts()
	dt_count = dt_count / dt_count.sum() * 100

	return dt_count


# ------------------------------------------------ Auxiliary Functios ----------------------------------------------- #
def Sers_toDF(*MDs):
	"""Convenience function to aggregate underlying Series (typically with the same index) in the passed Metadata
	instances as one DataFrame (returned). No checking is done."""
	return pd.DataFrame({md.val.name: md.val for md in MDs})


def askwargs(kwargs_str):
	"""Convenience function to read function args as string, all passed via keyword, and returned as the resulting
	namespace as dictionary"""
	toEval = lambda **kwargs : kwargs
	return eval("toEval({})".format(kwargs_str))


#todo  -- similar in nature to calc_timestep_dist(), define a helper function that would yield the data points
#      wherein you have sub res and time gaps
# --------------------------------------------- End of Metadata  -------------------------------------------- #
__defexceptions()
