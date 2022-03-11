"""This module is standalone in the project, meaning it has no dependency within the project.

Author:         Pang Teng Seng, D. Kayanan
Create date:    Sep. 17, 2019
Revised date:   Feb. 22, 2022
Version:        2.0
Release date:   TBA

- Implemented GSF mode
"""
# Python 3rd parth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Python
import warnings
import logging

# DK library
#import DK_Numerical as dkn
from DK_Numerical import is_xymonotonic
from DK_Collections import fmtdict, fuzzyLookUp

class GenUnit():
	"""
	Base class for all power generation units (i.e. a power plant can have multiple generation units, which can
	possibly use different tech and fuels).

	CLASS PUBLIC ATTRIBUTES:
		settings        Dictionary of GenUnit method settings, some of which are also optional parameters (which
						default to the current settings) that can be set per call.

						USED BY     		         KEY                DEFINITION
						self.set_bidquantities()    'minbid_size'       Minimum bid size in MW
													'min_Nbids'         Minimum number of bids calculated
													'max_Nbids'         Maximum number of bids calculated
													'bid_precision'     Significant decimal places of bid sizes


						.self.calc_TotalCosts(), self.calc_SRMC(), self.calc_StartupCosts(), self.calc_bids()

													'currency_precision'   Significant decimal places of prices and costs


	CLASS PRIVATE ATTRIBUTES:
		__PPdb          The read power plant database (by cls.set_PPdb()) will be bound to this.
		__fleet         Series of {rowID: GenUnit instance}. Contains the active units in the fleet.

		__simul         Indicates the simulation index when a simulation is run (0-indexed). The methods of GenUnit
						behave slightly differently during simulation mode (e.g. effects of live random parameter). If
						not in simulation mode, this attr is set to -1.

		__dynamic_attr_defaults
						Dict of {attr: initial value}. Read by self.reset_dynamic_attrs(), it contains initialization
						values of dynamic attributes of GenUnit that are not None (every other dynamic attr is
						initialized/reset to None).

	# TODO TODO TODO Gen Unit attributes docu


	"""
	# ---------------------------------------------------------------------------------------- CLASS ATTRS
	# The power plant database underpinning the GenUnits is bound to this attr
	__PPdb = None
	__fleet = pd.Series(dtype='object')
	__simul = -1
	__dynamic_attr_defaults = {}
	#'Availability':     {key: None for key in ('is_available_P', 'planned_outages', 'unplanned_outages')},

	dispatch = None

	settings = {
		# self.set_bidquantities()
		'minbid_size':      1,
		'min_Nbids':        1,
		'max_Nbids':        10,
		'bid_precision':    1,

		# self.calc_TotalCosts(), self.calc_SRMC(), self.calc_StartupCosts(), self.calc_bids()
		'currency_precision':  2,
		'currency':         'USD',
	}

	# ---------------------------------------------------------------------------------------- CLASS METHODS
	@classmethod
	def set_PPdb(cls, filepath, readprms):
		"""Reads the power plant database given by filepath. set_PPdb() must first be called, before instantiating
		members of GenUnit, because the source database is read in doing so. This resets the fleet (removes all
		the active GenUnit members).

		Returns the read database (also bound to GenUnit.__PPdb).

		Note:
			PP DATABASE COL NAMES ARE EXPLICITLY ENCODED HERE.

			1) Changes to the power plant database are ILLEGAL and NOT CHECKED (changes might cause problems). The
			power plant database can only be complete reset via using set_PPdb() again.

			2) The expected sheets (a master and parameters sheet) and columns (in local variables master_cols and
			params_cols, respectively) are defined here.

		PARAMETERS:
			filepath        File path of the power plant database Excel file.
			readprms        Parameters that facilitate the read, as a dictionary with the ff. key, values pairs:

							'master'        The master sheet
							'params'        Plant parameters sheet

		The power plant database is stored in private attribute GenUnit.__PPdb, but this is also returned by
		GenUnit.set_PPdb(), and expects read-only uses.

		e.g. Set the database and store it
			PPdb = GenUnit.set_PPdb(PATHS['PP database'], readprms = {'master': 'Stations 2016',
									'params': 'Plant Parameters', 'seeds': 'Random Seeds'})

		"""
		# ----------------------------------------------------------------------- 0) Declarations
		master_cols = {'Power Plant', 'Owner/Operator', 'Main Power Producer', 'Vested (2004)', 'Vested (LNG)', 'Registration Status',
		               'Commissioned', 'Registered Capacity [MW]', 'Technology', 'Fuel', 'Fuel ID*', 'Cooling System',
		               'Stack height [m]', 'Condenser exhaust height [m]', 'Unit Name', 'Units', 'lat (°)',
		               'long (°)', 'Comments'}

		params_cols = {'Power Plant', 'Unit Name', 'Registered Capacity [MW]', 'Technology', 'Fuel ID*',
		               'Full Load Efficiency [%]', 'Efficiency rand var*', "Efficiency Curve*",  'Cogen Total '
		               'Efficiency',
		               'Min Stable Generation [MW]', #'Ramping Limit [MW/min]',
		               'Average Failures per year [yr-1]', 'Mean Time to Repair [wks]',
		               }

		randseed_cols = {'Power Plant', 'Unit Name',
		                 'Full Load Efficiency', 'Start Online', 'UP Time Duration', 'DOWN Time Duration'}

		common_cols = master_cols & params_cols
		# ----------------------------------------------------------------------- 1) Read PPdb from disk
		#                                                     IMPORTANT: Excel db column 'ID' is ignored.
		#                                                                Use default Pandas range index
		PPdb = {
			'master': pd.read_excel(filepath, sheet_name=readprms['master'], usecols=master_cols),

			'params': pd.read_excel(filepath, sheet_name=readprms['params'], usecols=params_cols),

			'randseeds': pd.read_excel(filepath, sheet_name=readprms['seeds'], usecols=randseed_cols),
		}
		# ----------------------------------------------------------------------- 2) Check common columns
		# Todo -- currently, just compares the master and params common cols
		# Assert that sheets have corresponding entries
		# equalnull is a modified == comparison of Series, wherein nan values are treated as equal.
		equalnull = lambda s1, s2: (s1.isnull() == s2.isnull()).all() and (s1[s1.notnull()] == s2[s2.notnull()]).all()

		commoncolcomp = {ccol: equalnull(PPdb['master'][ccol], PPdb['params'][ccol])
		                 for ccol in common_cols}

		if not all(commoncolcomp.values()):
			try:
				from DK_Collections import fmtdict
			except ModuleNotFoundError:
				fmtdict = str

			logging.error("Power plant database -- inconsistent columns. Pls. check the common col comparison:\n"
			              "{}\n".format(fmtdict(commoncolcomp)))
			raise ValueError("Common cols in the power plant database do not have identical values.")

		# ----------------------------------------------------------------------- 3) Name the indices
		# Call the index as 'rowID', because this is linked to the GenUnit attr, self.id['rowID']
		for db in PPdb.values():
			db.index.name = 'rowID'

		# ----------------------------------------------------------------------- 4) Set class attributes
		cls.__PPdb = PPdb                          # bind to class
		cls.__fleet = pd.Series(dtype='object')    # reset the fleet
		return PPdb

	@classmethod
	def get_PPdb(cls):
		"""Returns the PPdb"""
		return cls.__PPdb


	@classmethod
	def init_fleet(cls, rowIDs='all'):
		"""Convenient function to initialize the units in the power plant database. Pass an iterable of indices of
		the power plant database. Defaults to all units."""
		# First, reset the fleet
		cls.__fleet = pd.Series(dtype='object')

		if rowIDs == 'all':
			rowIDs = cls.__PPdb['master'].index

		# Remember __init__ updates cls.__fleet
		for idx in rowIDs:
			GenUnit(idx)
		return


	@classmethod
	def get_fleet(cls):
		"""Returns the GenUnit fleet (used to be a shallow copy, but on second thought, this had no advantages. You
		couldn't change what object cls.__fleet is bound to, but you could still change the underlying anyway)."""
		#return cls.__fleet.copy(deep=False)
		return cls.__fleet


	@classmethod
	def del_GenUnit(cls, rowID):
		"""Delete an active GenUnit from the fleet, identified by its database index, rowID."""
		if rowID not in cls.__fleet.index:
			raise ValueError("GenUnit to delete does not exist.")

		cls.__fleet = cls.__fleet.loc[cls.__fleet.index != rowID]
		return


	# ---------------------------------------------------------------------------------------- INIT
	def __init__(self, rowID):
		"""
		Instantiation of GenUnit is intimately tied with reading the power plant database, which must first be bound
		to PowerPlants.GenUnit before the first GenUnit is instantiated (use method .set_PPdb()). Each new instance
		is bound to GenUnit.__fleet (Series of rowID : GenUnit instance).

		PARAMETERS:
			rowID       Row label of GenUnit.__PPdb, referring to which record to initialize as a GenUnit.

		"""
		if GenUnit.__PPdb is None:
			raise RuntimeError('No power plant database defined. Must set the power plant database via '
			                   'GenUnit.set_PPdb() before instantiating GenUnit.')

		# Fetch record from the power plant database. Do it this way to ride on panda's unique row labels for Excel read
		# (the ID columns in the Excel db are ignored)
		# todo - note that the init process just reads GenUnit.__PPdb, regardless of how it was set
		master_row = GenUnit.__PPdb['master'].loc[rowID, :]
		params_row = GenUnit.__PPdb['params'].loc[rowID, :]
		rseeds_row = GenUnit.__PPdb['randseeds'].loc[rowID, :]

		# --------------------------------------------------------------------------------- 1) Set ATTRIBUTES
		# ........................................................ a) Identification
		self.id = {
			'Unit Name':    master_row.at['Unit Name'],
			'Units':        master_row.at['Units'],
			'Plant Name':   master_row.at['Power Plant'],
			'Gen Co':       master_row.at['Owner/Operator'],
			'rowID':        rowID,
		}

		# ........................................................ b) Bools
		True_if_Y = lambda YorN: {'Y': True}.get(YorN, False)
		self.is_ = {
			'Main Producer': True_if_Y(master_row.at['Main Power Producer']),
			'Vested 2004'  : True_if_Y(master_row.at['Vested (2004)']),
			'Vested LNG'   : True_if_Y(master_row.at['Vested (LNG)']),
			'Cogen'        : 'cogen' in str(params_row.at['Technology']).lower(),
			'WtE'          : params_row.at['Technology'] == 'WtE ST' or master_row.at['Fuel'].lower() == 'biomass',
			'GSF'          : master_row.at['Registration Status'] == 'GSF',
			#'Biomass'      : master_row.at['Fuel'].lower() == 'biomass',
		}

		# ........................................................ c) Gen tech, fuel and cogen
		self.gentech = {
			'Prime Mover':          params_row.at['Technology'],
			'Capacity [MW]':        params_row.at['Registered Capacity [MW]'],
			'Pmin [MW]':            params_row.at['Min Stable Generation [MW]'],
			'FL efficiency [%]':    params_row.at['Full Load Efficiency [%]'],
			'Efficiency rand var*': params_row.at['Efficiency rand var*'],
			'Efficiency Curve*':    params_row.at['Efficiency Curve*'],
			'Fuel':                 master_row.at['Fuel'],
			'Fuel ID*':             params_row.at['Fuel ID*'],
			'Fuel Price':           None,
			'Fuel Conversion to MWh':   None,
			'Fuel latent heat per HHV': None,
			#'Ramping Limit [MW/min]':   params_row.at['Ramping Limit [MW/min]'],
			'Commissioned [yr]':        master_row.at['Commissioned'],

		}

		# Parameters for cogen only
		if self.is_['Cogen']:
			self.cogentech = {
				'Heat & power FL efficiency [%]':       params_row.at['Cogen Total Efficiency'],
				'Alternative boiler efficiency [%]':    None,
			}
			n_e = self.gentech['FL efficiency [%]']
			n_tot = self.cogentech['Heat & power FL efficiency [%]']
			n_h = n_tot-n_e

			self.cogentech['HPR [-]'] = n_h/n_e

		# Aliases (NONE's will be filled-in after reading the inputs)
		self.bind_aliases()

		# ........................................................ d) Reliability
		self.reliability = {
			'Average Failures per year [yr-1]':     params_row.at['Average Failures per year [yr-1]'],
			'Mean Time to Repair [wks]':            params_row.at['Mean Time to Repair [wks]'],
			'Limiting Availability':                None,
			'T lambda':                             None,
			'D mu':                                 None,
			'Mean Time to Fail [wks]':              None,
		}

		# ........................................................ d) Thermo -- prime mover and cooling system
		self.thermo = None
		self.heatstreams_byoutlet = None        # Stack, condenser, misc+aux loads
		self.heatstreams_bykind = None          # Sensible air, latent air, seawater
		# TODO cooling tech
		self.coolingsys = master_row.at['Cooling System']


		# ........................................................ e) Dynamic (all None)
		self.fuelprice_D = None
		self.Po_P, self.TotalCosts_P = None, None
		self.SRMarginalCosts_P, self.bids_P, self.Schedule = None, None, None
		# Pending attrs -- StartupCosts

		# ........................................................ f) Random parameters
		# selv.rv is initialized to a dict mapped to None
		# The stats random vars are initialized by gd_core.subinit5_preprocess()
		self.rv = {key: None for key in ('n_e_FL', 'T', 'D')}
		self.rsamples = {key: None for key in self.rv.keys()}

		self.rseeds = {
			'n_e_FL':   rseeds_row['Full Load Efficiency'],
			'start ol': rseeds_row['Start Online'],
			'T':        rseeds_row['UP Time Duration'],
			'D':        rseeds_row['DOWN Time Duration'],
		}

		# ........................................................ g) Misc
		self.misc = {
			'lat, long': (master_row.at['lat (°)'], master_row.at['long (°)']),
		}

		# --------------------------------------------------------------------------------- 2) Bind to __fleet
		GenUnit.__fleet.at[rowID] = self
		return


	def bind_aliases(self):
		"""Redeclares the alias attributes, to ensure consistency. Called during __init__()"""
		self.Pmin = self.gentech['Pmin [MW]']
		self.GenCap = self.gentech['Capacity [MW]']
		self.FuelPr = self.gentech['Fuel Price']
		self.fuel_conversion = self.gentech['Fuel Conversion to MWh']
		self.latheat_factor = self.gentech['Fuel latent heat per HHV']
		return


	# ---------------------------------------------------------------------------------------- DYNAMIC METHODS
	def get_fuelprice_D(self, day, forex_D=1.0, Scenarios=None, mode=0):
		"""Reads the fuel price vector of the GenUnit's fuel (in self.gentech['Fuel Price']), and sets
		self.fuelprice_D (in local currency per fqty) to the price for the day.

		PARAMETERS:
			day         Simulation day
			forex_D     (float) The effective local currency per USD rate on that day. Defaults to 1.0.

			mode        (int) Controls how the fuel price vector is read. Defaults to 0.

			Scenarios   (default) As in PPdispatch 'Scenario' module-level name.


		MODES:
			mode=0       Default
							- fit for all plants (has a default procedure + special cases)
							- direct querry via .tsloc(day, col=scenario)
							- WtE assumes fuel cost = gate fees for now
							# Todo - must work on WtE cost structure

		(Modes 10 amd above are old implementations)
			mode=10      Direct querry via .tsloc(day)
			mode=11      Read parameter (i.e. fixed price), keyed by plant name and assumed a Pint quantity in local
						currency

		Sets:
			self.fuelprice_D

		Resets:
			self.TotalCosts_P
			self.SRMarginalCosts_P
			self.bids_P

		Returns:
			self.fuelprice_D
		"""
		# Goal is to set one of these:
		fuelprice_LOC_D, fuelprice_USD_D = None, None

		# Prep ScenarioCol
		if Scenarios:
			ScenarioCol = Scenarios.get(self.gentech['Fuel ID*'])
		else:
			ScenarioCol = None


		# --------------------------------------------------------------------------- 1) Read self.gentech['Fuel Price']
		if mode == 0:
			# ......................................................... a) Exception 1 - WtE
			if self.gentech['Fuel ID*'] == 'Waste':
				# Both would raise an exception in the standard cost calculation
				# fuelprice_LOC_D = -1 * self.gentech['Fuel Price'][self.id['Plant Name']].magnitude
				# fuelprice_LOC_D = 0
				# Todo - improve WtE cost structure
				fuelprice_LOC_D = self.gentech['Fuel Price'][self.id['Plant Name']].magnitude

			# ......................................................... b) Default
			else:
				# Todo - time lag
				fuelprice_USD_D = self.gentech['Fuel Price'].tsloc(day, ScenarioCol)


		elif mode == 10:
			# Assumes a price time series as DataHandler.Metadata is bound
			fuelprice_USD_D = self.gentech['Fuel Price'].tsloc(day)

		elif mode == 11:
			# Price is stored in parameters, with key=gen.id['Plant Name']
			fuelprice_LOC_D = self.gentech['Fuel Price'][self.id['Plant Name']].magnitude
		else:
			raise NotImplementedError("Undefined mode '{}' passed.".format(mode))

		# --------------------------------------------------------------------------- SET - RESET
		# Set
		if fuelprice_LOC_D is not None:
			self.fuelprice_D = fuelprice_LOC_D
		elif fuelprice_USD_D is not None:
			self.fuelprice_D = fuelprice_USD_D * forex_D
		else:
			raise RuntimeError('Failed to fetch the fuel price.')

		# Reset
		self.TotalCosts_P, self.SRMarginalCosts_P, self.bids_P = None, None, None

		return self.fuelprice_D


	def set_bidquantities(self, mode, minbid_size=None, min_Nbids=None, max_Nbids=None, bid_precision=None,
	                      runchecks=True, **kwargs):
		"""Todo Update docstring (from pd to np)
		Sets the plant output tranches (bid steps), which is the independent variable of the supply curve of the
		GenUnit. This serves as an index (self.Po_P) to self.bidquantities_P, which calculates the additional MW
		the plant is bidding per output level.

		self.bidquantities_P is a Pandas Series of {Po [MW] : bid quantity [MW]}, with the ff. properties:

			- Po index starts at 0 MW (0 MW maps to NaN)
			- monotonically increasing
			- Except for Po=0, Po is in [self.Pmin, self.GenCap] (further tightened by ramping constraints)
			- bid quantities is the first difference of the Po index

		PARAMETERS:
			mode                    Selects the algorithm for computing the bid steps. See 'MODE' below.

		OPTIONAL PARAMETERS:
			minbid_size             (defaults to GenUnit settings (default 1 MW)) The minimum bid size in MW.

			min_Nbids, max_Nbids    (defaults to GenUnit settings (default 1 to 10)) Inclusive range of the
									bounds of the number of bid steps.

			bid_precision           (defaults to GenUnit settings (default 1)) Number of significant decimal
									places in the bid steps.

			runchecks               (Optional; defaults to True) If True, then assertions are made.

			kwargs                  Mode-specific parameters. Pls. see section 'MODE' for a full description of what
									additional parameters can be passed to each mode.


		MODE:
			Let
				Pmin        self.gentech['Pmin [MW]']
				GenCap      self.gentech['Capacity [MW]']

			mode=0      Random no of bids and bid quantities, ignoring ramp limits.

			mode=1      Set bids from Pmin to GenCap (exclusive) in steps of minbid_size (last step capped until
						GenCap), ignoring the parameters min_Nbids and max_Nbids; and ramp limits. Numpy arange is used,
						which recommends an integer value for minbid_size. This is useful for plotting the
						semi-continuous version of the costs.

			mode=2      Sets bids from Pmin to GenCap (inclusive) in an evenly-spaced sequence of a given number of
						bids (Nbids).

						kwargs:
							Nbids           The number of bids to set.

		Sets:
			self.Po_P
			self.bids_P (Sets cols 0: Bid ID and 1: Bid Qty;)
				col 0 [Bid ID]      calculated
				col 1 [Bid qty]     calculated
				col 2 [Price]       init to 0
				col 3 [Acc]         init to 0

		Resets:
			self.TotalCosts_P
			self.SRMarginalCosts_P

		Returns:
			self.Po_P

		# ------------------------------------------------------------------------------------------------------------ #
		DEV NOTES
			Po_idx (and array source Po_idx_np) always starts at 0
			Nbids == len(Po_idx)-1
			Pmin < Po_idx < GenCap (except for Po = 0 MW)

			Addt'l notes for mode=0
			# ub_arr[0] is for 1st bid --> which has Nbids-1 ahead of it, and thus we reserve:
			#   minbid_size*(Nbids-1)   [MW] of capacity for the later bids.
			# This is an arithmetic series with common difference =
			#   (stop-start)/(num-1) = minbid_size


		"""
		Pmin, GenCap = self.Pmin, self.GenCap

		# Optional parameters to default
		if minbid_size is None:
			minbid_size = GenUnit.settings['minbid_size']

		# For modes that use Nbid limits
		if mode not in (1,):
			if min_Nbids is None:
				min_Nbids = GenUnit.settings['min_Nbids']
			if max_Nbids is None:
				max_Nbids = GenUnit.settings['max_Nbids']

			if max_Nbids * minbid_size > GenCap - Pmin:
				warnings.warn("Might not be able to satisfy the minbid_size and max_Nbids constraint.  "
				              "Pls. consider lowering at least one of them.")

		# For modes that do not use Nbid limits
		else:
			min_Nbids, max_Nbids = None, None

		if bid_precision is None:
			bid_precision = GenUnit.settings['bid_precision']


		# -------------------------------------------------------------------------- 1) CALCULATE Nbids and Po_arr
		#                                                                               (Nbids is int no of bids)

		# ..................................................................... [MODE 0]
		if mode == 0: # Random (no of steps, steps, no ramping limtits)
			Nbids = np.random.randint(min_Nbids, max_Nbids+1)
			Po_arr = np.zeros(Nbids+1)

			randfs = np.random.random_sample(Nbids)
			ub_arr = np.linspace(start=GenCap-minbid_size*(Nbids-1),
			                     stop=GenCap, num=Nbids, endpoint=True)
			for idx in range(Nbids):
				ub = ub_arr[idx]
				if idx == 0:
					lb = Pmin
				else:
					lb = Po_arr[idx] + minbid_size
				Po_arr[idx+1] = lb + (ub-lb) * randfs[idx]


		# ..................................................................... [MODE 1]
		elif mode == 1:
			Po_arr = np.append(0.0, np.arange(Pmin, GenCap, minbid_size))
			Nbids = len(Po_arr) - 1

		# ..................................................................... [MODE 2] Evenly spaced [lb, ub] w/ Nbids
		elif mode == 2:
			# Fetch required kwargs
			try:
				Nbids = kwargs['Nbids']
			except KeyError:
				raise TypeError("Missing keyword argument 'Nbids'.")

			Po_arr = np.append(0.0, np.linspace(Pmin, GenCap, num=Nbids))

		else:
			raise NotImplementedError("Undefined mode '{}' passed.".format(mode))

		# --------------------------------------------------------------------------- 2) Round to Po_rd
		# Round to bid precision
		Po_rd = np.around(Po_arr, decimals=bid_precision)
		# Correction -- do not round Pmin and GenCap in the original
		Po_rd[np.where(Po_arr == Pmin)[0]] = Pmin
		Po_rd[np.where(Po_arr == GenCap)[0]] = GenCap

		# --------------------------------------------------------------------------- 3) Init bids table
		# COL 0 - Bids id
		Bid_ids = np.fromiter((100*self.id['rowID'] + idx for idx in range(Nbids)), dtype='f8')

		# COL 1 - Bid quantities (delta P)
		Bid_qtys = Po_rd[1:] - Po_rd[:-1]
		# Required assertion -- bids must be increasing
		# assert all(Bid_qtys > 0), "Set Po schedule is invalid. Bid_qtys: {}".format(Bid_qtys)

		# bids_P table
		bids_P = np.zeros((Nbids, 4))
		bids_P[:, 0] = Bid_ids
		bids_P[:, 1] = Bid_qtys

		# --------------------------------------------------------------------------- 4) Assertions
		if runchecks:
			# a) Within capacity
			assert all(Pmin <= Po_rd[1:]) and all(Po_rd[1:] <= GenCap), Po_rd

			# b) No of bids
			assert Nbids == len(Po_rd) - 1, (Nbids, Po_rd)
			if (min_Nbids, max_Nbids) != (None, None):
				assert min_Nbids <= Nbids <= max_Nbids, "The Nbids bounds was violated: {}".format(Po_rd)

			# d) Min bid step
			assert Bid_qtys.min() > minbid_size - 10**-3, Bid_qtys

		# --------------------------------------------------------------------------- 4) SET - RESET
		self.Po_P = Po_rd
		self.bids_P = bids_P

		self.TotalCosts_P, self.SRMarginalCosts_P = None, None
		return Po_rd


	def effcurve(self, Po_MW=None, normalized=False):
		"""Implements the efficiency curve (Po in MW --> absolute/normalized efficiency). Returns an array* of the
		efficiency values (always within [0,1]).

		*Even for a scalar input, the result is a 1-item array.

		PARAMETERS:
			Po_MW       (Optional; defaults to self.Po_P set by self.set_bidquantities()) Output power in MW,
						array-like. Passing values outside the permitted range of [Pmin, GenCap] would raise ValueError,
						except for Po = 0, which maps to nan.

			normalized  (Optional; defaults to False) Return the normalized efficiency if True, which is in per unit
						of the full load efficiency. That is,

						normalized=True     GenCap --> 1
						normalized=False    GenCap --> FL efficiency <= 1
		"""
		# .............................................................. 0) Check and condition Po_MW
		if 'unbounded eff curve interpolant' not in self.misc:
			raise RuntimeError("This method requires that the normalized efficiency curves as a callable interpolant* "
			                   "of the per-unit output power is bound to self.misc['unbounded eff curve interpolant']."
			                   " *Such as the one returned by scipy.interpolate.interp1d")

		# Po_MW default -- use self.Po_P
		if Po_MW is None:
			if self.Po_P is None:
				raise RuntimeError("If Po_MW is not used, then  bid steps should be set by self.set_bidquantities()")
			Po_MW = self.Po_P

		# Else, coerce Po_MW into an np.array
		else:
			# 0 --> np.nan bypass
			if Po_MW is 0:
				return  np.array([np.nan], dtype='f8')

			try:
				if hasattr(Po_MW, "__len__"):
					Po_MW = np.array(Po_MW, dtype='f8')
				else:
					Po_MW = np.array([Po_MW], dtype='f8')
			except:
				print("Could not typecast input into a numpy array of dtype='f8'.")
				raise

		# .............................................................. 1) Bound to Pmin, GenCap
		out_of_bounds = np.fromiter((P for P in Po_MW if (P < self.Pmin or P > self.GenCap) and P != 0.0), dtype='f8')

		#if len(out_of_bounds) > 0:
			#raise ValueError("Po values for gen ({}: {}) are outside the limits of [Pmin, GenCap] and != 0: {} "
			#                 "MW".format(self.id['rowID'], self.id['Unit Name'], out_of_bounds))

		# .............................................................. 2) Apply pu conversions
		eff_out = self.misc['unbounded eff curve interpolant'](Po_MW/self.GenCap)

		if not normalized:
			# Todo -- test added rv
			if GenUnit.__simul == -1:
				eff_out = eff_out * self.gentech['FL efficiency [%]']/100
			else:
				# Simulation mode; add random parameter
				# Alternative way: via DxP_index and self.Schedule['n_e_pot']. But doing it via GenUnit.__simul
				# allows us to distinguish simulation mode.
				eff_out = eff_out * self.rsamples['n_e_FL'][GenUnit.__simul]

		# Force NaN for P = 0 MW
		eff_out[np.where(Po_MW == 0.0)] = np.nan

		return eff_out


	def calc_TotalCosts(self, currency_precision=None, runchecks=True):
		"""Calculates the Total Costs of the output levels in self.Po_bidbins_P. The results are written to
		self.TotalCosts_P as a series {Po: Total Cost} (Po_idx_P as index).

		-------------------------------------------------------------------------------------
		Total Cost = Fuel Cost + Non-Fuel Cost

		FC(Po)        =   fuelpr_D     *    fuel_conv     *     Eff^-1      *     Po           (element-wise mult)
		[loc_cur/h]    [loc_cur/fqty]     [fqty/MWh_f]       [MW_f/MW_e]        [MW_e]
		(vector; dep)     (scalar)          (scalar)       (vector; dep)   (vector; indep)
		                                      [HHV]          [usu. LHV]
		Where,
			loc_cur         Local currency
			fqty            units of fuel (arbitrary unit; can be different units of energy, mass or vol)
			_e, _f          Electricity, fuel. For combusted fuels, always use HHV (for fairness across energy
							conversion processes)

		-------------------------------------------------------------------------------------
		PSEUDOCODE:
			0) Checks (must explicitly set these)
				- fuelpr_D
				- Po bid steps

			1) Fetch fuel_conv (based on fuel qty; purely numeric as unit conversion has to be done externally)

			CALCULATIONS -- as np arrays. Indexing with bid steps done at the end.
			2) Calculate FC
				- calc eff (absolute) vector
				- FC equation

			3) Calculate the non-FC

			4) TC = FC + non-FC

			5) Assertions

			6.1) Set        self.TotalCosts_P  (Series of {Po bid steps : TC}, in the local currency per h)
			6.2) Reset      self.SRMarginalCosts_P
			6.3) Return     self.TotalCosts_P

		"""
		# --------------------------------------------------------------------------- 0) PREP + CHECKS
		# ....................................................... a) Prep settings
		if currency_precision is None:
			currency_precision = GenUnit.settings['currency_precision']

		# ....................................................... b) Fuel price must be set
		if self.fuelprice_D is None:
			raise RuntimeError('Pls. set the daily fuel price (.get_fuelprice_D()) first before calculating the Total '
			                   'Costs')

		# ....................................................... c) Po bid steps must be set
		if self.Po_P is None:
			raise RuntimeError('Pls. set the bid steps (.set_bidquantities()) first before calculating the Total '
			                   'Costs')

		# --------------------------------------------------------------------------- 1) FUEL COSTS
		Eff_ofPo = self.effcurve(Po_MW=None, normalized=False)
		fuel_conv = 1/self.fuel_conversion.magnitude
		# self.fuel_conversion is a Pint quantity as [MWh/fqty]

		FuelCosts = np.array(self.fuelprice_D * fuel_conv * (1/Eff_ofPo) * self.Po_P, dtype='f8')
		# [loc_cur/h]          [loc_cur/fqty] [fqty/MWh_f]  [MW_f/MW_e]         [MW_e]

		# Overwrite Po = 0: NaN to 0, because Eff_ofPo has NaN here
		FuelCosts[0] = 0.0

		# --------------------------------------------------------------------------- 2) NON-fuel costs
		# TODO -- plant-specific (e.g. cogen)
		NonFuelCosts = 0.0

		# --------------------------------------------------------------------------- 3) TOTAL COSTS
		# This is where the Series conversion used to take place
		TotalCosts_P = np.around(FuelCosts + NonFuelCosts, decimals=currency_precision)

		# --------------------------------------------------------------------------- 4) ASSERTIONS
		if runchecks:
			if len(self.Po_P) > 2:
				is_xymonotonic(self.Po_P, TotalCosts_P, slope='pos', as_assertion=True)

		# --------------------------------------------------------------------------- 5) SET - RESET
		self.TotalCosts_P = TotalCosts_P
		self.SRMarginalCosts_P = None

		return TotalCosts_P


	def calc_SRMC(self, currency_precision=None, runchecks=True):
		"""Calc SRMC allows self.TotalCosts_P to be None (it attempts to call calc_TotalCosts())
			1) Set        self.SRMarginalCosts_P
			2) Reset      -
			3) Return     self.SRMarginalCosts_P

		"""
		# --------------------------------------------------------------------------- 0) Prep
		if self.TotalCosts_P is None:
			self.calc_TotalCosts(runchecks=runchecks)

		if currency_precision is None:
			currency_precision = GenUnit.settings['currency_precision']

		# --------------------------------------------------------------------------- 1) Calc
		# Bid_qtys = Po_rd[1:] - Po_rd[:-1]  -- from .set_bidquantities()
		# SRMC = self.TotalCosts_P.diff() / self.bidquantities_P
		_TC = self.TotalCosts_P
		_bidqty = self.bids_P[:, 1]

		SRMC = (_TC[1:] - _TC[:-1]) / _bidqty


		# --------------------------------------------------------------------------- 2) Assertions, THEN polish
		if runchecks:
			# SRMC * delta P should sum up to the Total Cost
			assert abs((SRMC * _bidqty).sum() - _TC[-1]) < 10**-3

		# Polishing
		SRMC = np.around(SRMC, decimals=currency_precision)

		# --------------------------------------------------------------------------- 3) SET - RESET
		self.SRMarginalCosts_P = SRMC
		return SRMC


	def calc_bids(self, main_mode='SRMC', sub_mode='shift all', **kwargs):
		"""Calculates the GenUnit bids.

		MODES:
			'SRMC'          The GenUnit bids at marginal cost. self.SRMarginalCosts_P is used and smoothened to have
							strictly-increasing bids. Only this mode is implemented in the first version of genDispatch.

			'WtE'           WtE plant simplifcation
							As a power plant not based on energy commodities, WtE plants costs are not modelled. They
							are set to bid near-0 at an output level set by their monthly schedule. Pls. pass the ff.
							kwarg:

							WtE_sched=dFundamentals['WtE Sched']
							D_day=D_day in NEMS.simulate() (time['D_index'] members)



		-------------------------------------------------------------------------------------
		PSEUDOCODE for SRMC mode:

		1) Apply monotonically inc requirement to SRMC

		2) If cogen, then shift by the value of heat (= MC of direct combustion of the same fuel)

		3) If vested, then ...

		4) Assertions

		5) Exit
			5.1) Set        self.bids_P
			5.2) Reset      n/a
			5.3) Return     self.bids_P

		BUGS (solved):

			In sub_mode == 'shift all', originally it was if delta < 0:
			This failes if calculated SRMC is the same for subsequent steps. Thus, change to <= 0. In these cases, every
			subsequent bid is increased by min_inc


		fuel_conv = 1/self.fuel_conversion.magnitude
		# self.fuel_conversion is a Pint quantity as [MWh/fqty]

		FuelCosts = np.array(self.fuelprice_D * fuel_conv * (1/Eff_ofPo) * self.Po_P, dtype='f8')
		# [loc_cur/h]          [loc_cur/fqty] [fqty/MWh_f]  [MW_f/MW_e]         [MW_e]


		"""
		# raise NotImplementedError('Only the SRMC-based bidding is implemented for now.')

		if main_mode == 'SRMC':
			# --------------------------------------------------------------------------- 0) CHECKS + PREP SERIES
			if self.Po_P is None:
				raise RuntimeError('Pls. set the bid steps (.set_bidquantities()) first before calculating the bids.')
			if self.SRMarginalCosts_P is None:
				raise RuntimeError('Pls. calculate the marginal costs first (.calc_SRMC()) first before '
				                   'calculating the bids.')

			# Init arrays [loc_cur / MWh]
			SRMC_raw = self.SRMarginalCosts_P
			MC_shift = np.zeros(SRMC_raw.shape)

			# --------------------------------------------------------------------------- 1) MONOTONICALLY INC
			min_inc = 10**-GenUnit.settings['currency_precision']
			if sub_mode == 'shift all':
				# i) Traverse delta_MC to calculate MC_shift
				for idx, delta in enumerate(SRMC_raw[1:] - SRMC_raw[:-1]):
					# For dec MC, must shift that item and all thereafter
					if delta <= 0:
						MC_shift[idx+1:] += min_inc - delta

				# ii) Shift MC
				bidPrices = SRMC_raw + MC_shift

			elif sub_mode == 'true min':
				raise NotImplementedError
				bidPrices.iloc[0:2] = SRMC_raw.iloc[0:2]

				# Just shift the next, and then choose the larger of the previous MC and the current MC
				for idx, _SRMC in enumerate(SRMC_raw.iloc[2:], start=2):
					bidPrices.iloc[idx] = max(SRMC_raw.iloc[idx], bidPrices.iloc[idx - 1] + min_inc)

			# Assert increasing - can consider removing
			assert all((bidPrices[1:] - bidPrices[:-1]) > 0)

			# --------------------------------------------------------------------------- 2) COGEN (heat discount)
			if self.is_['Cogen']:
				# Single shift downward by cogen heat value
				HPR = self.cogentech['HPR [-]']
				n_boiler = self.cogentech['Alternative boiler efficiency [%]']/100
				fuel_conv = 1 / self.fuel_conversion.magnitude

				bidPrices -= round(HPR * (1/ n_boiler) * fuel_conv * self.fuelprice_D,
				#            [MWh_h/MWh_e] [MWh_f/MWh_h]   [fqty/MWh_f] [loc_cur/fqty]
				                   GenUnit.settings['currency_precision'])

			# --------------------------------------------------------------------------- 5) SET-RESET
			# Only bidPrices is calculated
			# Col 3: acc is initialized to all 0
			self.bids_P[:, 2] = bidPrices

			return self.bids_P

		elif main_mode == 'WtE':
			# In this mode, the work done by self.set_bidquantities() is also performed,
			# and costs calculations in self.calc_TotalCosts() and self.calc_SRMC() are not required.

			# .................................................................... a) Init
			WtE_sched, D_day = kwargs['WtE_sched'], kwargs['D_day']
			Pmin, GenCap = self.Pmin, self.GenCap

			# .................................................................... b) Get schedule
			AveLoad_MW = WtE_sched.val.at[D_day.month, self.id['Unit Name']]
			assert Pmin-10**-3 <= AveLoad_MW <= GenCap+10**-3

			# .................................................................... c) Construct bids_P
			Nbids = 2
			Bid_ids = np.fromiter((100 * self.id['rowID'] + idx for idx in range(Nbids)), dtype='f8')
			Bid_qtys = np.array([Pmin, AveLoad_MW-Pmin])
			bidPrices = np.array([0, 0.1])

			# bids_P table
			bids_P = np.zeros((Nbids, 4))
			bids_P[:, 0] = Bid_ids
			bids_P[:, 1] = Bid_qtys
			bids_P[:, 2] = bidPrices

			self.bids_P = bids_P
			return self.bids_P
        
		elif main_mode == 'GSF':
			# In this mode, the work done by self.set_bidquantities() is also performed,
			# and costs calculations in self.calc_TotalCosts() and self.calc_SRMC() are not required.

			# .................................................................... a) Init
			#WtE_sched, D_day = kwargs['WtE_sched'], kwargs['D_day']
			Pmin, GenCap = self.Pmin, self.GenCap

			# .................................................................... b) Get schedule
			#AveLoad_MW = WtE_sched.val.at[D_day.month, self.id['Unit Name']]
			#assert Pmin-10**-3 <= AveLoad_MW <= GenCap+10**-3

			# .................................................................... c) Construct bids_P
			Nbids = 2
			Bid_ids = np.fromiter((100 * self.id['rowID'] + idx for idx in range(Nbids)), dtype='f8')
			Bid_qtys = np.array([Pmin,GenCap])
			bidPrices = np.array([0, 0.1])

			# bids_P table
			bids_P = np.zeros((Nbids, 4))
			bids_P[:, 0] = Bid_ids
			bids_P[:, 1] = Bid_qtys
			bids_P[:, 2] = bidPrices

			self.bids_P = bids_P
			return self.bids_P
            
		else:
			raise NotImplementedError('Only the SRMC-based, WtE and GSF bidding are implemented for now.')



	# ---------------------------------------------------------------------------------------- OTHER INSTANCE METHODS
	def __repr__(self):
		return "Gen {rowID}: {UnitName} \nStation: {PlantName}\n{GenCap} MW, {PM}".format(
			rowID=self.id['rowID'], UnitName=self.id['Unit Name'], PlantName=self.id['Plant Name'], GenCap=self.GenCap,
			PM=self.gentech['Prime Mover'])


	def info(self, omit=('Fuel Price', 'Commissioned [yr]', 'Ramping Limit [MW/min]')):
		"Functions as an extended __repr__, it also prints out info on the generator features."
		print("Gen {rowID}: {UnitName} \nStation: {PlantName}\n\n{geninfo}".format(
			rowID=self.id['rowID'],
			UnitName=self.id['Unit Name'],
			PlantName=self.id['Plant Name'],
			geninfo=fmtdict({key: val for key, val in self.gentech.items() if key not in omit})))
		return


	def getPPdb_record(self, *features, sheet='master'):
		"""Fetches the corresponding PPdb record from which the GenUnit was instantiated. Pass select fields in
		var-pos parameter features to subset the columns. Invalid columns would return nans."""
		if len(features) == 0:
			features = GenUnit.__PPdb[sheet].columns
		else:
			features = pd.Index(features)
		return GenUnit.__PPdb[sheet].loc[self.id['rowID']].reindex(features)


	def get_CostsTable(self):
		"""Returns a DataFrame summarizing the power plant costs over the bid quantities."""
		_df =  pd.DataFrame(index=self.Po_P,
		                    data={
			                    'Total Costs [per h]': self.TotalCosts_P,
		                    })
		# Must separate these, because lengths are not the same
		_df['Marginal Costs [per MWh]'] = pd.Series(index=self.Po_P[1:], data=self.SRMarginalCosts_P)
		_df['Bid Price [per MWh]'] = pd.Series(index=self.Po_P[1:], data=self.bids_P[:, 2])
		return _df


	def plot_Costs(self, getTable=False, **kwargs):
		"""Plots the GenUnit costs (total costs, marginal costs) over its output.

		See: https://matplotlib.org/gallery/api/two_scales.html
		"""
		dfCosts = self.get_CostsTable()

		# Scale TC to thousands
		dfCosts['Total Costs [per h]'] = dfCosts['Total Costs [per h]']/1000

		# Init plot
		fig, ax1 = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

		# ........................................................................................ 1) Plot TC
		TC_color = kwargs.get('TC_color', '#2471A3')
		L1 = plt.plot(dfCosts.index, dfCosts['Total Costs [per h]'], color=TC_color, label='VC')

		# ........................................................................................ 2a) Plot SRMC
		ax2 = ax1.twinx()
		SRMC_color = kwargs.get('SRMC_color', '#B03A2E')
		L2 = plt.step(dfCosts.index, dfCosts['Marginal Costs [per MWh]'], color=SRMC_color, label='SRMC')

		# ........................................................................................ 2b) Plot bids
		Bids_color = kwargs.get('Bids_color', '#8E44AD')
		L3 = plt.step(dfCosts.index, dfCosts['Bid Price [per MWh]'], color=Bids_color, label='Bids')


		# ........................................................................................ Polish plots
		# Polish TC axis
		ax1.tick_params(axis='y', labelcolor=TC_color)
		ax1.set_ylim(0.9 * dfCosts['Total Costs [per h]'].min(), 1.1 * dfCosts['Total Costs [per h]'].max())

		ax1.set_ylabel("Variable Costs \n [k{}/h]".format(GenUnit.settings['currency']), fontsize=14)


		# Polish SRMC axis
		ax2.tick_params(axis='y', labelcolor=SRMC_color)

		if 'SRMClims' in kwargs:
			ax2.set_ylim(kwargs['SRMClims'])
		elif dfCosts['Marginal Costs [per MWh]'].iloc[1:].between(0, 150).all():
			ax2.set_ylim(0, 150)
		else:
			ax2.set_ylim(dfCosts['Marginal Costs [per MWh]'].min(), dfCosts['Marginal Costs [per MWh]'].max())

		ax2.set_ylabel("SRMC / Bids\n [{}/MWh]".format(GenUnit.settings['currency']), fontsize=14)

		# Common to both
		ax1.set_xlim(0, self.GenCap * 1.05)
		ax1.grid(True)
		ax1.legend()
		ax2.legend(loc='lower right')
		ax1.set_title("Power plant cost curves", fontweight='bold', fontsize=16)
		ax1.set_xlabel("Plant output [MW]", fontsize=14)

		plt.show()

		if getTable:
			return dfCosts


def get_plantclass(PPclass, GenFleet, PPdb, synonyms=None, as_ids=False):
	"""Returns a subset of GenFleet (Series) of the gen units matching the PPclass. Can be returned as a set of gen
	IDs or a subset of GenFleet (use param as_ids).

	DISCLAIMER -- synonyms are hard-coded. Synonyms replace the arguments.

	PARAMETERS:
		PPclass             Two-tuple of string keys as (Gen Tech, Fuel)
							If either Gen Tech or Fuel key is '', then this means all of them.

		GenFleet, PPdb      As in PPdispatch

		synonyms            (Optional) Is a dictionary of {key: synonym}. This allows for flexibility in both the
							fuel and tech keys, by providing keys. that are deemed synonymous.

							e.g. A synonym for 'Crude' can be 'Oil'. In this case, if externally, the latter term is
							preferred, then we can pass synonyms={'Oil': 'Crude'}.

		as_ids              (Optional; defaults to False). If True, then a set of the gen IDs is returned. Else,
							a Series of the corresponding gens is returned.

	DEV NOTES:
		Query process works as:
		1) try exact
		2) if TypeError, then there is just 1 match
		3) if KeyError, then try a fuzzy match

	"""
	# Get keys with synonyms
	if synonyms is None:
		synonyms = {}

	techkey = synonyms.get(PPclass[0], PPclass[0])
	fuelkey = synonyms.get(PPclass[1], PPclass[1])

	# ------------------------------------------
	_SerTech = pd.Series(index=PPdb['master']['Technology'].values, data=PPdb['master'].index)

	#if fuelkey == 'Waste':
	if fuelkey == 'Waste' or fuelkey == 'Biomass' or fuelkey.lower() == 'nat gas':                                 # newly added
		# Note WtE plants now have blank Fuel ID*, but the Fuel column still indicates 'waste'
		_SerFuel = pd.Series(index=PPdb['master']['Fuel'].values, data=PPdb['master'].index)
	else:
		_SerFuel = pd.Series(index=PPdb['master']['Fuel ID*'].values, data=PPdb['master'].index)

	# ---------------------------------------------------------------------------- 1) Querry gen tech
	if techkey == '':
		# get all
		tech_matches = set(_SerTech)
	else:
		try:
			tech_matches = set(_SerTech[techkey])
		except TypeError:  # single match
			tech_matches = {_SerTech[techkey]}
		except KeyError:   # no matches; use fuzzy
			tech_matches = set()
			tech_key_matches = set(key for key, score in fuzzyLookUp(techkey, _SerTech, matcheslim=None))

			for key in tech_key_matches:
				try:
					tech_matches.update(set(_SerTech[key]))
				except TypeError:
					tech_matches.update({_SerTech[key]})

	# ---------------------------------------------------------------------------- 2) Querry fuel (same structure)
	if fuelkey == '':
		# get all
		fuel_matches = set(_SerFuel)
	else:
		try:
			fuel_matches = set(_SerFuel[fuelkey.lower()])
		except TypeError:  # single match
			fuel_matches = {_SerFuel[fuelkey.lower()]}
		except KeyError:   # no matches; use fuzzy
			fuel_matches = set()
			fuel_key_matches = set(key for key, score in fuzzyLookUp(fuelkey.lower(), _SerFuel, matcheslim=None))

			for key in fuel_key_matches:
				try:
					fuel_matches.update(set(_SerFuel[key]))
				except TypeError:
					fuel_matches.update({_SerFuel[key]})

	# ---------------------------------------------------------------------------- 3) Get intersection of matches
	final_matches = tech_matches & fuel_matches

	if as_ids:
		return final_matches
	else:
		return GenFleet.loc[final_matches]

# Note to self: Why HHV-basis?
# 1) Fuels are quoted on an HHV-basis
# 2) HHV represents the total energy (not 100% heat as there could be latent heat) available from the fuel (it is
# based on the combustion products achieving equilibrium with the environment at 25C)
# 3) We need latent heat for WRF.

def calc_CCGT_FLflows(net_HHVeff, GT_ST_ratio, eff_ST=0.3, HHV_LHV_ratio=1.105, aux_load=0.05, x_GT_misc=0.005,
                      x_HRSG_misc=0.002, x_ST_misc=0.005, min_Stack_StoL=1.0):
	"""Calculates the full load energy flows of a CCGT plant, per unit fuel HHV.
	(Currently a special case based on the required parameters; eff_GT is implied from these)

	DESIGN SPECIFICATIONS:
	(all within [0,1], and not a percentage)

		net_HHVeff      HHV net electrical efficiency of the plant (considering house loads)
						Note: I suspect that the typical 40%-60% efficiency is LHV and gross

		GT_ST_ratio     Ratio of gas turbine output to steam turbine output in the net electricity produced (after
						electricity losses and house loads). Recommended value: at least 2.

		eff_ST          Target rankine cycle efficiency (soft requirement; will be adjusted if solution violates the
						minimum stack sensible heat loss). The rankine efficiency is the ratio of the HRSG recovered
						heat to the ST output (typically ~0.3-0.45).

		HHV_LHV_ratio   Ratio of HHV to LHV of the fuel (1.10-1.11 for natural gas).

		aux_load        House / auxiliary loads of the power plant (typically 0.05; up to 0.1)


	OPTIONAL PARAMETERS:
		min_Stack_StoL  Minimum sensible heat to latent heat ratio in the stack heat.
						(Recommended value based on Handbook for Cogen and CC plants example of highly efficient CCGT)

		x_GT_misc
		x_HRSG_misc
		x_ST_misc


	RETURNS:
		*Note: Stack heat includes the latent heat of combustion
		energy flows    Series of the energy flows of the CCGT plant
		losses          Series of CCGT losses
		metrics         Series of calculated efficiencies of major components.


	DEV NOTES:
		Let variables to solve be prepended with x_

		--- Gas Turbine stage ---
		x_GT            GT elec output (gross; before house loads)
		x_GT_exh        GT exhaust heat
		x_GT_misc       Misc. GT losses (e.g. radiation)

		--- Heat Recovery stage ---
		x_HRSG          HRSG recovered heat (input to Rankine cycle)
		x_Stack         Stack heat (must be larger than the latent heat of combustion, since it also has sensible heat)
		x_HRSG_misc     Misc. HRSG losses (e.g. radiation)

		--- Steam Turbine stage ---
		x_ST            ST elec output (gross; before house loads)
		x_Condenser     Heat rejected by condenser
		x_ST_misc       Misc. losses of Rankine cycle (e.g. radiation)

	REFERENCES:
		OCGT efficiency 35%-40%, but upt to 46% (proxy for Brayton cycle efficiency)
		(http://www.ipieca.org/resources/energy-efficiency-solutions/power-and-heat-generation/open-cycle-gas-turbines/)


	"""
	# ............................................................. 1) Overall plant output
	gross_elec = net_HHVeff + aux_load

	# ............................................................. 2) GT balance
	# a) Assume GT:ST
	x_ST = 1 / (GT_ST_ratio+1) * gross_elec
	x_GT = gross_elec - x_ST

	# b) calc exhaust
	x_GT_exh = 1 - x_GT - x_GT_misc

	assert abs(x_GT + x_GT_misc + x_GT_exh - 1) < 10**-6

	# ............................................................. 3) HRSG balance (stack loss)
	# a) Assume ST efficiency
	x_HRSG = x_ST/eff_ST

	# b) calc stack and check if ST efficiency is valid
	x_Stack = x_GT_exh - x_HRSG - x_HRSG_misc

	LH_combustion = 1-1/HHV_LHV_ratio

	if x_Stack < (1+min_Stack_StoL)*LH_combustion:
		warnings.warn('Calculated stack heat is too small to have sufficient sensible heat. Increasing the overall '
		              'efficiency of the steam cycle.')
		x_Stack = (1+min_Stack_StoL)*LH_combustion
		x_HRSG = x_GT_exh - x_Stack - x_HRSG_misc

	# ............................................................. 4) ST balance and losses (condenser)
	# calc condenser
	x_Condenser = x_HRSG - x_ST - x_ST_misc

	# total misc losses
	x_misc = x_GT_misc + x_HRSG_misc + x_ST_misc

	# ............................................................. 5) Assertions
	# GT balance
	assert abs(1 - (x_GT + x_GT_misc + x_GT_exh)) < 10 ** -6, 'GT energy balance not satistfied'
	# HRSG balance
	assert abs(x_GT_exh - (x_HRSG + x_HRSG_misc + x_Stack)) < 10 ** -6, 'HRSG energy balance not satistfied'
	# ST balance
	assert abs(x_HRSG - (x_ST + x_Condenser + x_ST_misc)) < 10 ** -6, 'ST energy balance not satistfied'
	# Plant overall balance
	TotalElec = x_GT + x_ST - aux_load
	TotalLoss = x_Stack + x_Condenser + x_misc + aux_load
	assert abs(1 - (TotalElec + TotalLoss)) < 10 ** -6, 'Overall energy balance not satistfied'

	# ............................................................. 6) Results
	energy_flows = pd.Series({
		'GT elec': x_GT,
		'GT misc losses': x_GT_misc,
		'GT exhaust': x_GT_exh,
		'HRSG recovered heat': x_HRSG,
		'HRSG misc losses': x_HRSG_misc,
		'Stack': x_Stack,
		'ST elec': x_ST,
		'ST misc losses': x_ST_misc,
		'Condenser': x_Condenser,
		'Aux loads': aux_load,
	})

	losses = pd.Series({
		'Stack': x_Stack,
		'Condenser': x_Condenser,
		'Misc and aux loads': x_misc + aux_load,
	})

	metrics = pd.Series({
		'η Brayton': x_GT,
		'η HRSG': x_HRSG / x_GT_exh,
		'η Rankine': x_ST / x_HRSG,
		'η Overall(HHV)': net_HHVeff,
	})

	return energy_flows, losses, metrics


def calc_CogenCCGT_BPST_FLflows(net_HHVeff, HPR_net, HPR_bp=4, HHV_LHV_ratio=1.105, aux_load=0.05, x_GT_misc=0.005,
                      x_HRSG_misc=0.002, x_ST_misc=0.005, min_Stack_StoL=1.0):
	"""Calculates the full load energy flows of a Cogeneration CCGT plant with a back-pressure ST, per unit fuel HHV.

	RETURNS:
		*Note: Stack heat includes the latent heat of combustion
		energy flows    Series of the energy flows of the CCGT plant
		losses          Series of CCGT losses
		metrics         Series of calculated efficiencies of major components and HPRs.
	"""
	# ............................................................. 1) Calc cogen heat
	x_Cogen = net_HHVeff * HPR_net

	# ............................................................. 2) Rankine balance
	# a) ST output from HPR_ext
	x_ST = x_Cogen / HPR_bp

	# b) Rankine balance
	x_HRSG =  x_ST + x_Cogen + x_ST_misc

	# ............................................................. 3) GT balance
	# a) GT output from ST and net HHV eff
	x_GT = net_HHVeff + aux_load - x_ST

	# b) GT balance
	x_GT_exh = 1 - x_GT - x_GT_misc

	# ............................................................. 4) HRSG balance -- check stack
	x_Stack = x_GT_exh - x_HRSG - x_HRSG_misc
	LH_combustion = 1 - 1 / HHV_LHV_ratio

	if x_Stack < (1 + min_Stack_StoL) * LH_combustion:
		raise ValueError("Pls. assign a higher HPR_bp to fulfill minimum stack flow (typical: 4-14). Otherwise, "
		                 "too much power and/or cogen heat is being generated.")
		# Raised error because it's not straightforward to adjust the parameters


	# ............................................................. 5) Assertions
	# total misc losses
	x_misc = x_GT_misc + x_HRSG_misc + x_ST_misc

	# GT balance
	assert abs(1 - (x_GT + x_GT_misc + x_GT_exh)) < 10 ** -6, 'GT energy balance not satistfied'
	# HRSG balance
	assert abs(x_GT_exh - (x_HRSG + x_HRSG_misc + x_Stack)) < 10 ** -6, 'HRSG energy balance not satistfied'
	# ST balance
	assert abs(x_HRSG - (x_ST + x_Cogen + x_ST_misc)) < 10 ** -6, 'ST energy balance not satistfied'
	# Plant overall balance
	TotalElec = x_GT + x_ST - aux_load
	TotalLoss = x_Stack + x_misc + aux_load
	assert abs(1 - (TotalElec + x_Cogen + TotalLoss)) < 10 ** -6, 'Overall energy balance not satistfied'

	# ............................................................. 6) Results
	energy_flows = pd.Series({
		'GT elec'            : x_GT,
		'GT misc losses'     : x_GT_misc,
		'GT exhaust'         : x_GT_exh,
		'HRSG recovered heat': x_HRSG,
		'HRSG misc losses'   : x_HRSG_misc,
		'Stack'              : x_Stack,
		'ST elec'            : x_ST,
		'ST misc losses'     : x_ST_misc,
		'BP cogen'           : x_Cogen,
		'Aux loads'          : aux_load,
	})

	losses = pd.Series({
		'Stack'             : x_Stack,
		'Condenser'         : 0.0,
		'Misc and aux loads': x_misc + aux_load,
	})

	metrics = pd.Series({
		'η Brayton'     : x_GT,
		'η HRSG'        : x_HRSG / x_GT_exh,
		'η Rankine'     : x_ST / x_HRSG,
		'η Overall(HHV)': net_HHVeff,
		'HPR(net)'    : x_Cogen/TotalElec,
		'HPR(BP)'     : x_Cogen/x_ST,
	})

	return energy_flows, losses, metrics


def calc_OCGT_FLflows(net_HHVeff, HHV_LHV_ratio=1.105, aux_load=0.05, x_GT_misc=0.005, min_Stack_StoL=1.0):
	""""""
	# ............................................................. 1) GT balance (stack)
	x_GT = net_HHVeff + aux_load
	x_Stack = 1.0 - x_GT - x_GT_misc

	# Stack sensible heat check
	LH_combustion = 1 - 1/HHV_LHV_ratio

	if x_Stack < (1 + min_Stack_StoL) * LH_combustion:
		warnings.warn('Calculated stack heat is too small to have sufficient sensible heat. Adjusting the overall '
		              'efficiency of the plant.')
		# Adjust stack
		x_Stack = (1 + min_Stack_StoL) * LH_combustion
		# Adjust GT balance
		x_GT = 1.0 - x_Stack - x_GT_misc


	# ............................................................. 2) Assertions
	# GT balance
	assert abs(1 - (x_GT + x_GT_misc + x_Stack)) < 10 ** -6, 'GT energy balance not satistfied'

	# ............................................................. 6) Results
	energy_flows = pd.Series({
		'GT elec':          x_GT,
		'GT misc losses':   x_GT_misc,
		'Stack':            x_Stack,
		'Aux loads':        aux_load,
	})

	losses = pd.Series({
		'Stack':                x_Stack,
		'Condenser':            0.0,
		'Misc and aux loads':   x_GT_misc + aux_load,
	})

	metrics = pd.Series({
		'η Brayton'     : x_GT,
		'η Overall(HHV)': x_GT-aux_load,
	})

	return energy_flows, losses, metrics


def calc_ST_FLflows(net_HHVeff, eff_ST, x_SG_otherlosses, HHV_LHV_ratio=1.05, aux_load=0.05, x_ST_misc=0.005,
                    min_Stack_StoL=1.0, ):
	"""Calculates the full load energy flows of a steam power plant (coal, oil, waste, natural gas), per unit fuel
	HHV.

	DESIGN SPECIFICATIONS:
	(all within [0,1], and not a percentage)

		net_HHVeff      HHV net electrical efficiency of the plant (considering house loads)

		eff_ST          Target electrical efficiency of the Rankine cycle (gross electricity produced). This is a
						soft requirement; will be adjusted if solution violates the minimum stack sensible heat
						loss. The rankine efficiency is the ratio of the SG steam enthalpy increase to the ST output
						(typically ~0.3-0.45).

		x_SG_otherlosses    Other steam generator losses. This includes miscellaneous losses such as radiation and
							convection from the boiler surface, but may also include more significant losses, such as
							bottom ash for coal.

							Recommended values [1-3]:
								coal            0.01 - 0.03
								oil             0.007 - 0.008
								natural gas     0.006 - 0.007
								waste           0.03


		HHV_LHV_ratio   Ratio of HHV to LHV of the fuel (1.10-1.11 for natural gas).

		aux_load        House / auxiliary loads of the power plant (typically 0.05; up to 0.1)


	OPTIONAL PARAMETERS:
		min_Stack_StoL      Minimum sensible heat to latent heat ratio in the stack heat.

	RETURNS:
		energy_flows

		losses              Stack losses just include the latent and sensible heat in exhaust gases. Other combustion
							losses, such as bottom ashes (if any) are accounted in the misc. losses.

		metrics


	References:
		[1] For oil and natural gas boiler losses.
		https://www.nrcan.gc.ca/energy/efficiency/industry/technical-info
		/tools/boilers/5429

		[2] For coal boiler losses. https://beeindia.gov.in/sites/default/files/2Ch7.pdf

		[3] For W2E boiler losses. Search for 'radiation'
		https://ec.europa.eu/environment/waste/framework/pdf/guidance.pdf
	"""
	# ............................................................. 1) Calc ST elec out
	x_ST = net_HHVeff + aux_load

	# ............................................................. 2) ST balance
	x_SG = x_ST / eff_ST

	x_Condenser = x_SG - x_ST - x_ST_misc

	# ............................................................. 3) SG balance
	x_Stack = 1 - x_SG - x_SG_otherlosses

	# Check stack latent heat
	LH_combustion = 1 - 1/HHV_LHV_ratio

	if x_Stack < (1 + min_Stack_StoL) * LH_combustion:
		warnings.warn('Calculated stack heat is too small to have sufficient sensible heat. Adjusting the overall '
		              'efficiency of the steam cycle.')
		x_Stack = (1 + min_Stack_StoL) * LH_combustion
		x_SG = 1 - x_Stack - x_SG_otherlosses
		x_Condenser = x_SG - x_ST - x_ST_misc

	# ............................................................. 4) Assertions
	x_otherlosses = x_SG_otherlosses + x_ST_misc

	# SG balance
	assert abs(1 - (x_SG + x_Stack + x_SG_otherlosses)) < 10**-6, 'SG energy balance not satistfied'
	# ST balance
	assert abs(x_SG - (x_ST + x_Condenser + x_ST_misc)) < 10**-6, 'ST energy balance not satistfied'
	# Plant overall balance
	TotalElec = x_ST - aux_load
	TotalLoss = x_Stack + x_Condenser + x_otherlosses + aux_load
	assert abs(1 - (TotalElec + TotalLoss)) < 10**-6, 'Overall energy balance not satistfied'

	# ............................................................. 5) Form results
	energy_flows = pd.Series({
		'SG steam ΔH':      x_SG,
		'SG other losses':  x_SG_otherlosses,
		'Stack':            x_Stack,
		'ST elec':          x_ST,
		'ST misc losses':   x_ST_misc,
		'Condenser':        x_Condenser,
		'Aux loads':        aux_load,
	})

	losses = pd.Series({
		'Stack':            x_Stack,
		'Condenser':        x_Condenser,
		'Misc and aux loads': x_otherlosses + aux_load,
	})

	metrics = pd.Series({
		'η SG':             x_SG,
		'η Rankine':      x_ST / x_SG,
		'η Overall(HHV)': net_HHVeff,
	})

	return energy_flows, losses, metrics


def calc_CogenBPST_FLflows(net_HHVeff, x_SG_otherlosses, HPR_net=None, HPR_bp=None, HHV_LHV_ratio=1.05,
                           aux_load=0.05, x_ST_misc=0.005, min_Stack_StoL=1.0,):
	"""Calculates the full load energy flows of a cogeneration steam power plant (coal, oil, waste, natural gas) with a
	back-pressure ST for cogenerating heat, per unit fuel HHV.

	DESIGN SPECIFICATIONS:
	(all within [0,1], and not a percentage)

		net_HHVeff      HHV net electrical efficiency of the plant (considering house loads)

		HPR_net, HPR_bp     Heat-to-power ratio overall, and within the BP-ST, respectively. The overall ratio is
							based on the net electrical output, whereas the BP ratio is based on the ST output (gross).
							(recommended HPR_bp 4-14). Must provide only one.

		x_SG_otherlosses    Other steam generator losses. This includes miscellaneous losses such as radiation and
							convection from the boiler surface, but may also include more significant losses, such as
							bottom ash for coal.

							Recommended values [1-3]:
								coal            0.01 - 0.03
								oil             0.007 - 0.008
								natural gas     0.006 - 0.007
								waste           0.03


		HHV_LHV_ratio   Ratio of HHV to LHV of the fuel (1.10-1.11 for natural gas).

		aux_load        House / auxiliary loads of the power plant (typically 0.05; up to 0.1)


	OPTIONAL PARAMETERS:
		min_Stack_StoL      Minimum sensible heat to latent heat ratio in the stack heat.


	RETURNS:
		energy_flows

		losses              Stack losses just include the latent and sensible heat in exhaust gases. Other combustion
							losses, such as bottom ashes (if any) are accounted in the misc. losses.
		metrics

	"""
	if HPR_net is None == HPR_bp is None:
		raise ValueError('Must use only one of HPR_net and HPR_ext.')

	# ............................................................. 1) BP-ST power & heat
	# a) ST output
	x_ST = net_HHVeff + aux_load

	# b) Cogen
	if HPR_net:
		x_Cogen = HPR_net * net_HHVeff
	if HPR_bp:
		x_Cogen = HPR_bp * x_ST

	# c) balance
	x_SG = x_ST + x_ST_misc + x_Cogen

	# ............................................................. 2) SG balance (stack)
	# Stack and latent heat check
	x_Stack = 1.0 - x_SG - x_SG_otherlosses

	# Check stack latent heat
	LH_combustion = 1 - 1/HHV_LHV_ratio

	if x_Stack < (1 + min_Stack_StoL) * LH_combustion:
		warnings.warn('Calculated stack heat is too small to have sufficient sensible heat. Adjusting the HPR to '
		              'achieve the minimal stack heat.')
		x_Stack = (1 + min_Stack_StoL) * LH_combustion

		# Adjust SG balance
		x_SG = 1.0 - x_Stack - x_SG_otherlosses
		# Adjust ST balance
		x_Cogen = x_SG - x_ST - x_ST_misc

	# ............................................................. 3) Assertions
	x_otherlosses = x_SG_otherlosses + x_ST_misc

	# SG balance
	assert abs(1 - (x_SG + x_Stack + x_SG_otherlosses)) < 10**-6, 'SG energy balance not satistfied'
	# ST balance
	assert abs(x_SG - (x_ST + x_Cogen + x_ST_misc)) < 10**-6, 'ST energy balance not satistfied'
	# Plant overall balance
	TotalElec = x_ST - aux_load
	TotalLoss = x_Stack + x_otherlosses + aux_load
	assert abs(1 - (TotalElec + x_Cogen + TotalLoss)) < 10**-6, 'Overall energy balance not satistfied'


	# ............................................................. 4) Results
	energy_flows = pd.Series({
		'SG steam ΔH':      x_SG,
		'SG other losses':  x_SG_otherlosses,
		'Stack':            x_Stack,
		'ST elec':          x_ST,
		'BP cogen':         x_Cogen,
		'ST misc losses':   x_ST_misc,
		'Aux loads':        aux_load,
	})

	losses = pd.Series({
		'Stack'             : x_Stack,
		'Condenser'         : 0.0,
		'Misc and aux loads': x_otherlosses + aux_load,
	})

	metrics = pd.Series({
		'η SG':           x_SG,
		'η Rankine':      x_ST / x_SG,
		'η Overall(HHV)': net_HHVeff,
		'HPR(net)':     x_Cogen / TotalElec,
		'HPR(BP)':      x_Cogen / x_ST,
	})

	return energy_flows, losses, metrics


def calc_CogenExtST_FLflows(net_HHVeff, eff_ST, x_SG_otherlosses, HPR_net=None, HPR_ext=None, HHV_LHV_ratio=1.05,
                            aux_load=0.05, x_ST_misc=0.005, min_Stack_StoL=1.0, ):
	"""Calculates the full load energy flows of a cogeneration steam power plant (coal, oil, waste, natural gas) with a
	back-pressure ST for cogenerating heat, per unit fuel HHV.

	DESIGN SPECIFICATIONS:
	(all within [0,1], and not a percentage)

		net_HHVeff      HHV net electrical efficiency of the plant (considering house loads)

		eff_ST          Power efficiency of the Rankine cycle (gross electricity output). Recommended 0.20-0.30 for
						an Ext-ST cogen.

		HPR_net, HPR_ext    Heat-to-power ratio overall, and within the Extraction-ST, respectively. The overall
							ratio is based on the net electrical output, whereas the BP ratio is based on the ST
							output (gross). (recommended HPR_ext 2-10). Must use only one.

		x_SG_otherlosses    Other steam generator losses. This includes miscellaneous losses such as radiation and
							convection from the boiler surface, but may also include more significant losses, such as
							bottom ash for coal.

							Recommended values [1-3]:
								coal            0.01 - 0.03
								oil             0.007 - 0.008
								natural gas     0.006 - 0.007
								waste           0.03


		HHV_LHV_ratio   Ratio of HHV to LHV of the fuel (1.10-1.11 for natural gas).

		aux_load        House / auxiliary loads of the power plant (typically 0.05; up to 0.1)


	OPTIONAL PARAMETERS:
		min_Stack_StoL      Minimum sensible heat to latent heat ratio in the stack heat.


	RETURNS:
		energy_flows

		losses              Stack losses just include the latent and sensible heat in exhaust gases. Other combustion
							losses, such as bottom ashes (if any) are accounted in the misc. losses.
		metrics

	"""
	if HPR_net is None == HPR_ext is None:
		raise ValueError('Must use only one of HPR_net and HPR_ext.')

	# ............................................................. 1) Ext-ST power & heat
	# a) ST output
	x_ST = net_HHVeff + aux_load

	# b) Cogen
	if HPR_net:
		x_Cogen = HPR_net * net_HHVeff
	if HPR_ext:
		x_Cogen = HPR_ext * x_ST

	# ............................................................. 2) ST balance
	# a) Apply Rankine efficiency
	x_SG = x_ST/eff_ST

	# b) calc condenser
	x_Condenser = x_SG - (x_ST + x_ST_misc + x_Cogen)


	# ............................................................. 3) SG balance (stack)
	# Stack and latent heat check
	x_Stack = 1.0 - x_SG - x_SG_otherlosses

	# Check stack latent heat
	LH_combustion = 1 - 1 / HHV_LHV_ratio

	if x_Stack < (1 + min_Stack_StoL) * LH_combustion:
		warnings.warn('Calculated stack heat is too small to have sufficient sensible heat. Adjusting the Rankine '
		              'efficiency to achieve the minimal stack heat.')
		x_Stack = (1 + min_Stack_StoL) * LH_combustion

		# Adjust SG balance
		x_SG = 1.0 - x_Stack - x_SG_otherlosses
		# Adjust ST balance
		x_Condenser = x_SG - (x_ST + x_ST_misc + x_Cogen)

		if x_Condenser < 0:
			raise ValueError('Parameters lead to an infeasible design. Must lower the electrical and/or cogen outputs.')

	# ............................................................. 4) Assertions
	x_otherlosses = x_SG_otherlosses + x_ST_misc

	# SG balance
	assert abs(1 - (x_SG + x_Stack + x_SG_otherlosses)) < 10 ** -6, 'SG energy balance not satistfied'
	# ST balance
	assert abs(x_SG - (x_ST + x_Cogen + x_Condenser + x_ST_misc)) < 10 ** -6, 'ST energy balance not satistfied'
	# Plant overall balance
	TotalElec = x_ST - aux_load
	TotalLoss = x_Stack + x_Condenser + x_otherlosses + aux_load
	assert abs(1 - (TotalElec + x_Cogen + TotalLoss)) < 10 ** -6, 'Overall energy balance not satistfied'


	# ............................................................. 5) Results
	energy_flows = pd.Series({
		'SG steam ΔH':      x_SG,
		'SG other losses':  x_SG_otherlosses,
		'Stack':            x_Stack,
		'ST elec':          x_ST,
		'Ext cogen':        x_Cogen,
		'Condenser':        x_Condenser,
		'ST misc losses':   x_ST_misc,
		'Aux loads':        aux_load,
	})

	losses = pd.Series({
		'Stack' :           x_Stack,
		'Condenser':        x_Condenser,
		'Misc and aux loads': x_otherlosses + aux_load,
	})

	metrics = pd.Series({
		'η SG'          : x_SG,
		'η Rankine'     : x_ST / x_SG,
		'η Overall(HHV)': net_HHVeff,
		'HPR(net)'    : x_Cogen / TotalElec,
		'HPR(Ext)'     : x_Cogen / x_ST,
	})
	
	return energy_flows, losses, metrics
