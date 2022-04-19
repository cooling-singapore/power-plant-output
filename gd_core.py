"""genDispatch core. Used to initialize power plants, inputs and underlying constructs.

DEV NOTES:
	The only changed done in refactoring the code is to make the PPdispatchError a local variable in __init_sequence(
	) and __check_PPdispatchError(), because PPdispatchError is a global variable in genDispatch.

	Now, we can further nest subinit5_preprocess() within this module, and would not complicate the structure of the
	top-level module.


Author:         Pang Teng Seng, D. Kayanan
Create date:    Dec. 9, 2019
Revised date:   Feb. 22, 2022
Version:        2.0
Release date:   TBA

"""
# Project modules
import PowerPlants as pp

# Python
import os
import datetime as dttm
import configparser
import warnings
import logging
import atexit
import json
from scipy import stats

# DK modules
import DataHandler as dh
import DK_Numerical as dkn
#from DK_Collections import fmtdict
from DK_Collections import *

with warnings.catch_warnings():
	# Suppress the warning upon importing fuzzywuzzy (no speedup)
	warnings.simplefilter("ignore")
	# from DK_Collections import fuzzyLookUp

# 3rd party and extensions
import pandas as pd
from scipy.interpolate import interp1d

log_time = lambda: dttm.datetime.now().strftime("%d.%m.%y, %H:%M")


def __init_sequence():
	"""This is the initialization sequence that must be called when the program is launched.

	DEV NOTES:
		The initialization consists of multiple sub steps, each preparing module-level names.

		The control is designed such that if an exception is encountered in any of these steps, __init_sequence() still
		reaches its return statement, thereby providing any available outputs. Sub steps that encountered no problems
		would have their outputs ready, while the rest (the one that failed and the subsequent) will have their outputs
		remaining as None.

		However, it is still desirable to raise the exception. This is done via the module-level PPdispatchError,
		which acts as a temporary binding to the exception. After __init_sequence() finishes (w/o fulfilling all the
		normal steps), any exception in PPdispatchError is raised.
	"""
	# ----------------------------------------------------------------------------------------- DECLARATIONS
	# Register exit handler
	atexit.register(normalexit)

	# Declare names and set to None.
	PATHS, config, metadata, = None, None, None
	dFundamentals, dGenParameters, dParameters, dStats = None, None, None, None
	time =  None
	PPdb, GenFleet, GenCos = None, None, None
	Scenarios = {}
	PPdispatchError = None
	# Control
	success = [False]*7
	#success = [True]*7
    
	# ----------------------------------------------------------------------------------------- MAIN INIT SEQUENCE
	# 1) Initialize __PATHS (w/ paths from config.ini) and __config. Asserts that these paths exist.
	try:
		PATHS, config = subinit1_initpaths_config_log()
		success[sum(success)] = True
		print('Success 1: Initialize __PATHS (w/ paths from config.ini) and __config. Asserts that these paths exist.')
	except Exception as err:
		#raise
		PPdispatchError = err


	# 2) Initialize metadata
	if success[0]:
		try:
			metadata = configparser.ConfigParser()
			metadata.read(PATHS['metadata'])
			success[sum(success)] = True
			print('Success 2 Initialize metadata')            
		except Exception as err:
			PPdispatchError = err



	# 3) Read the power plant database and check if the required input files can be found in the project directory
	if success[1]:
		try:
			PPdb = subinit2_readPPdb_checkkeys(PATHS, config, metadata)
			print(PPdb) 
			success[sum(success)] = True
			print('Success 3 Read the power plant database and check if the required input files can be found in the project directory')            
		except Exception as err:
			PPdispatchError = err

	# 4) Read (required) fundamentals and gen parameters
	if success[2]:
		try:
#			print("Checkpoint 4:Read (required) fundamentals and gen parameters")            
			dFundamentals, dGenParameters, dParameters = subinit3_readinputs(PATHS, config, metadata, PPdb)
#			print("dFundamentals : ")   
#			print(dFundamentals)            
            
			success[sum(success)] = True
			print('Success 4 Read (required) fundamentals and gen parameters"')               
		except Exception as err:
			PPdispatchError = err

	# 5) Preprocess input data
	if success[3]:
		try:
			GenFleet, GenCos = subinit4_initGenUnits(PATHS, PPdb, config)
			success[sum(success)] = True
		except Exception as err:
			PPdispatchError = err

	# 6) Initialize GenUnits
	if success[4]:
		try:
			time, dStats = subinit5_preprocess(config, dFundamentals, dGenParameters, dParameters, GenFleet, PPdb)
			success[sum(success)] = True
		except Exception as err:
			PPdispatchError = err

	# 7) Patches
	# Note - here lies any experimental code
	if success[5]:
		try:
			subinit6_patches(PATHS, metadata, dFundamentals)
			success[sum(success)] = True
		except Exception as err:
			PPdispatchError = err


	# ------------------------------------------------------------------------------------ EXIT - Check success
	success_res = sum(success)
        

	if success_res == len(success):
		# Successful init
		logging.info("Initialization successful.\n---------------------------------------------------------------")
		print("Initialization successful -- pls. set the simulation period to proceed.")

	else:
		# Oh no. Something wrong happened in step success+1
		success_res += 1

		print("ERROR: Initialization step {} failed.".format(success_res))
		logging.critical("[PROGRAM FAILED EXIT] Initialization step {} failed at {}.".format(success_res, log_time()))
		logging.shutdown()

	__check_PPdispatchError(PPdispatchError, config)
	return PATHS, config, metadata, PPdb, dFundamentals, dGenParameters, dParameters, dStats, time, Scenarios, \
	       GenFleet, GenCos, PPdispatchError


# -------------------------------------------------------------------------- Init sequence sub routines
def subinit1_initpaths_config_log():
	"""
	Initializes the paths (stored in global __PATHS):
		1 Finds the project location
		2 Reads config.ini
		3 Reads the paths defined in config.ini
		4 Checks that the paths exist
	"""
	# -------------------------------------------------------------------------------- 1) FUNDAMENTAL PATHS
	#                                                                                  - project root + directory
	PATHS = {'Proj': os.path.dirname(os.path.abspath(__file__)),}
	# Proj directory is well-defined. All paths are relative to the root (Paths['Proj'])
	toAbsPath = lambda PATHS_key, relpath: os.path.join(PATHS[PATHS_key], relpath)

	# ............................................................... a) Subdirs of root
	PATHS['Inputs and resources'] = toAbsPath('Proj', 'Inputs and resources')
	PATHS['Results'] = toAbsPath('Proj', 'Results')
	PATHS['To WRF'] = toAbsPath('Proj', 'To WRF')

	# ............................................................... b) Subdirs of Inputs and resources
	PATHS['Fundamentals'] = toAbsPath('Inputs and resources', 'Fundamentals')
	PATHS['Gen Parameters'] = toAbsPath('Inputs and resources', 'Gen Parameters')
	PATHS['Other Parameters'] = toAbsPath('Inputs and resources', 'Other Parameters')

	PATHS['Fundamentals', 'fuels'] = toAbsPath('Fundamentals', 'fuels')
	PATHS['Gen Parameters', 'efficiency curves'] = toAbsPath('Gen Parameters', 'efficiency curves')


	# ............................................................... c) Subdirs of To WRF
	PATHS['WRF resources'] = toAbsPath('To WRF', 'Resources')



	# -------------------------------------------------------------------------------- 2) Read CONFIG
	PATHS['config'] = toAbsPath('Inputs and resources', 'config.ini')
	config = configparser.ConfigParser()
	config.read(PATHS['config'])

	# -------------------------------------------------------------------------------- 3) Start log
	PATHS['log'] = toAbsPath('Inputs and resources', config['log']['file_name'])
	logging.basicConfig(filename=PATHS['log'], level=eval(config['log']['level']), filemode='w')
	logging.info("[PROGRAM START] at {}.\nInitialization commencing. \n "
	             "---------------------------------------------------------------".format(log_time()))

	# -------------------------------------------------------------------------------- 4) __PATHS from CONFIG
	# Q: Why is the metadata file configurable?
	# A: If all inputs are configurable, and the metadata is part of the input, then rightfully so.
	PATHS['PP database'] = toAbsPath('Inputs and resources', config['paths']['fp_powerplant_database'])
	PATHS['metadata']    = toAbsPath('Inputs and resources', config['paths']['fp_metadata'])
	PATHS['pint defn']   = toAbsPath('Inputs and resources', config['data import settings']['pint_unitdefn'])


	# -------------------------------------------------------------------------------- 5) Check that all dir/file exists
	donotexist = tuple(key for key, fp in PATHS.items() if not os.path.exists(fp))

	if donotexist:
		strgen = ("\t{}: '{}'".format(key, PATHS[key]) for key in donotexist)
		raise FileNotFoundError("The ff. paths or files were not found: \n{}\n\nPls. double check that "
		                        "config.ini (section 'paths') points to these required paths in the project "
		                        "directory, and that the project directory system was not altered.".format(
								'\n'.join(strgen)))

	return PATHS, config


def subinit2_readPPdb_checkkeys(PATHS, config, metadata):
	"""
	Reads the power plant database and determines the required input files (fundamentals and parameters):
		1) Read the power plant database from disk

		2) Read the database and check the required input files for:
			- fuels
			- efficiency curves
			(could add more in the future)

			This is done by:
				2.1) Collect key requirements from PPdb (e.g. fuels, params, etc.)
				2.2) Check if these keys are in metadata.ini
				2.3) Use the corresponding instructions in metadata.ini to check if the required input files are in
				./Inputs and resources

		If 2.2 or 2.3 fails, raise appropriate exception.
	"""
	# ----------------------------------------------------------------------- 1) Read PPdb from disk
	PPdb = pp.GenUnit.set_PPdb(PATHS['PP database'], readprms={key: config['power plant database'][key]
	                                                           for key in config['power plant database']})

	# ----------------------------------------------------------------------- 2.1) Collect the required keys
	print('Checkpoint 1: ')
	print(PPdb['params'])
	prmssh = PPdb['params']

	# NOTE - When you have more metadata sections, this is where you add them
	# note - dropna() here to allow WtEs to have no fuel key
	# {metadata section : df[['in metadata', 'file found']]
	checkKeys = {
		'fuels':             pd.DataFrame(index=pd.Index(prmssh['Fuel ID*'].dropna().unique())),
		'efficiency curves': pd.DataFrame(index=pd.Index(prmssh['Efficiency Curve*'].dropna().unique())),
	}
	#                       df.index = required keys as in sheet (apply key.lower() to check in metadata)
    
	print('Checkpoint 2: checkKeys')
	print(checkKeys)
    
	# Prep for 2.3
	extract_fname = lambda **args: args['filename'] # Use Python's kwargs parser :)
	PATHS_tupkey = {key[1]: key for key in PATHS.keys() if isinstance(key, tuple)}
    
	print("extract_fname: ")
	print(str(extract_fname))
	print(type(extract_fname))    
	print("PATHS_tupkey: ")
	print(PATHS_tupkey)  
	print(checkKeys.items())
    
	for mdsection, df in checkKeys.items():
		# ------------------------------------------------------------ 2.2) Check if the keys are in metadata.ini
		# logical series for filtering items that have to be checked further
		df['in metadata'] = pd.Series(index=df.index,
		                              data=(key.lower() in metadata[mdsection] for key in df.index))
		sub_idx = df['in metadata'].loc[df['in metadata'] ].index

		print("mdsection                          :"+mdsection)
		print(df)        
		print("\n"+"df['in metadata']             :")
		print(df['in metadata'])        

		# ------------------------------------------------------ 2.3) Check if input files are in the project directory
		#                                                             (only for keys found in metadata)
		#                                                        2.3.1) Build the check df's
		df['file found'] = pd.Series(index=df.index)
        
		print("df['file found']                   :")
		print(df['file found'])        

		print("sub_idx:")
		print(sub_idx)        
		for key in sub_idx:
			mdkey = key.lower()

			print(mdkey)            
            
			# a) Extract the filename
			try:
				fname = eval("extract_fname({})".format(metadata[mdsection][mdkey]))
			except SyntaxError:
				print("SyntaxError encountered while evaluating the metadata['{mdsection}']['{mdkey}'] instructions. "
				      "Pls. check that the following encoded argument in the metadata file is a valid expression to "
				      "pass to DataHandler.Metadata(): \n\n '{arg}'\n\n".format(
					mdsection=mdsection, mdkey=mdkey, arg=metadata[mdsection][mdkey]))
				raise

			if fname is None:
				raise NotImplementedError("This implies that dh.Metadata() will be called with values passed. Current "
				                          "implementation only expects file reads.")

			# b) Get the path
			fp = os.path.join(PATHS[PATHS_tupkey.get(mdsection, mdsection)], fname)

			# c) Check if exists and assign to series
			df.loc[key, 'file found'] = os.path.exists(fp)

			print("fname                     :"+fname)
			print("fp                        :"+fp)
			#print("os.path.exists(fp)        :"+os.path.exists(fp))
            

	# ------------------------------------------------------ 2.3.2) Summarize the results
	# Do this by looking for the failed keys
	err_msg = "Error in checking the parameter and input keys in the power plant database: \n\n"

	# a, b) Not in metadata, In metadata but file not found
	Failed_metadata, Failed_file = {}, {}

	for mdsection, df in checkKeys.items():
		_md = tuple(key for key in df.index if not df.loc[key, 'in metadata'])
		_file = tuple(key for key in df.index if not df.loc[key, 'file found'])

		if _md: Failed_metadata[mdsection] = _md
		if _file: Failed_file[mdsection] = _file

	# c) Report
	if Failed_metadata:
		err_msg += "The ff. keys were not found in the metadata file: \n\n{}\n\n".format(
			"\n".join("\t{}: {}".format(mdsection, ", ".join(keys)) for mdsection, keys in Failed_metadata.items()))

	if Failed_file:
		err_msg += "The ff. keys were not found in the appropriate project input directories: \n\n{}\n\n".format(
			"\n".join("\t{}: {}".format(mdsection, ", ".join(keys)) for mdsection, keys in Failed_file.items()))

	if Failed_metadata or Failed_file:
		logging.debug("\n\n".join("\n{}\n{}".format(key.upper(), val) for key, val in checkKeys.items()))
		raise RuntimeError(err_msg)

	return PPdb


def subinit3_readinputs(PATHS, config, metadata, PPdb):
	"""Reads ALL the fundamentals and parameter inputs as specified by the config file and power plant database.

	Note: Unit-handling support via Pint 0.9 of dh.Metadata is used.

	FUNDAMENTALS:
		- system demand
		- forex (market currency per USD)
		- fuel prices

	GEN PARAMETERS:
		- efficiency curves

	OTHER PARAMETERS:
		- higher heating values
		- fuel densities
		- cogen alternative boiler efficiencies

	"""
	# ------------------------------------------------------------------------------- 0) Preparations (Pint here)
	dFundamentals, dGenParameters, dParameters = {}, {}, {}
	# ............................................................... a) Metadata options
	dh.Metadata.opt.update({key: eval(val) for key, val in dict(config['data import settings']).items()
	                        if key in dh.Metadata.opt})

	print("Test : ")   
	print(fmtdict(dh.Metadata.opt))      
	logging.info("dh.Metadata options set upon data import:\n{}\n".format(fmtdict(dh.Metadata.opt)))

	# ............................................................... b) Explicit warning log
	Md_warning_notes = []
	if not dh.Metadata.opt['warn_no_units']:
		Md_warning_notes.append("NO unit check")
	if dh.Metadata.opt['allow_subres']:
		Md_warning_notes.append("sub res ALLOWED")
	if dh.Metadata.opt['allow_timegaps']:
		Md_warning_notes.append("time gaps ALLOWED")

	if len(Md_warning_notes) > 0:
		logging.warning("dh.Metadata: {}\n".format(", ".join(Md_warning_notes)))

	print("Test : ")   
	print(Md_warning_notes)   
        
	# ............................................................... c) Metadata unit handling via Pint
	kwargs = {'own_defnfile': PATHS['pint defn']}

	print("Test : ")   
	print(kwargs)     
    
	# Define local currency in Pint
	local_cur = config['market settings']['currency'].upper()
	print("Test : ")   
	print(local_cur)     
        
	if local_cur != 'USD':
		kwargs['direct_defn'] = "{cur} = [{cur}] = {cur_low}".format(cur=local_cur, cur_low=local_cur.lower())

	#dh.set_UnitHandler('Pint 0.9', silent=True, **kwargs)
	print("Test dh.set_UnitHandler : ")   
	print(dh.set_UnitHandler('Pint 0.9', silent=True, **kwargs))      
    
	logging.info("Unit defn file read: {}".format(config['data import settings']['pint_unitdefn']))

	# ............................................................... d) Others
	sh_params = PPdb['params']

	print("Test : ")   
	print(sh_params)      
    
	def _Md(md_section, md_key):
		"""Wrapper for instantiating dh.Metadata from metadata instructions. This abstracts the phrases 'dh.Metadata()'
		and 'metadata' from the code below"""
		# todo - if you want explicit units, this is where to do it
		py_str = "dh.Metadata(report='log', {})".format(metadata[md_section][md_key])
		try:
			return eval(py_str)
		except:
			logging.error("Metadata instantiation failed for metadata['{}']['{}']".format(md_section, md_key))
			raise

	# ------------------------------------------------------------------------------- 1) FUNDAMENTALS
	#                                                              1.1 System demand
	logging.info("--------------------------------------------------------------- Reading fundamentals")
	dh.set_readpath(PATHS['Fundamentals'])
	dFundamentals['sys demand'] = _Md('demand', md_key='system_electricity_mw')

	#                                                              1.2 Forex
	if config['market settings']['currency'] != 'USD':
		dFundamentals['forex'] = _Md('forex', md_key=config['market settings']['currency'])


	#                                                              1.3 Fuel prices
	# note -- again, dropna() for WtE plants
	logging.info("Reading fundamentals -- fuels")
	dh.set_readpath(PATHS['Fundamentals', 'fuels'])
	dFundamentals['fuel prices'] = {key: _Md('fuels', md_key=key) for key in
	                                sh_params['Fuel ID*'].dropna().unique()}

	# Store the fuel quantity denomination (e.g. MMBtu, bbl, ton, etc.) as more_md
	ureg = dh.Metadata.get_UHaux('ureg')
	for key, fuelmd in dFundamentals['fuel prices'].items():
		if key == 'Waste': continue
		fuelmd.more_md.update({'fuel qty': ureg(str(fuelmd.units).split(' / ')[1])})


	# ------------------------------------------------------------------------------- 2) GEN PARAMETERS
	#                                                              2.1 Efficiency curves
	logging.info("--------------------------------------------------------------- Reading gen parameters -- "
	             "efficiency curves")
	dh.set_readpath(PATHS['Gen Parameters', 'efficiency curves'])

	dGenParameters['efficiency curves'] = {key: _Md('efficiency curves', md_key=key) for key in
	                                       sh_params['Efficiency Curve*'].unique()}

	# ------------------------------------------------------------------------------- 3) OTHER PARAMETERS
	dh.set_readpath(PATHS['Other Parameters'])
	#                                                              3.1 HHV table
	logging.info("--------------------------------------------------------------- Reading HHV table")
	dParameters['HHV'] = _Md('constants', md_key='HHV_table')
	#                                                              3.2 LHV table
	logging.info("--------------------------------------------------------------- Reading LHV table")
	dParameters['LHV'] = _Md('constants', md_key='LHV_table')
	#                                                              3.3 Fuel densities
	logging.info("--------------------------------------------------------------- Reading fuel densities")
	dParameters['fuel densities'] = _Md('constants', md_key='fuel_densities')
	#                                                              3.4 Fuel densities
	logging.info("------------------------------------------------------ Reading cogen alternative boiler efficiencies")
	dParameters['boiler efficiencies'] = _Md('constants', md_key='cogen_alt_boiler_eff')


	# ...................................................................... Exit
	logging.info("Read input complete.\n")
 
	print("dFundamentals End : ")   
	print(dFundamentals)      
	return dFundamentals, dGenParameters, dParameters


def subinit4_initGenUnits(PATHS, PPdb, config):
	"""Initializes the GenUnit instances. Returns two containers:
		1) GenFleet, a Series of GenUnit idx : GenUnit instance
		2) GenCos, a dictionary of generator company (str) : GenUnit indices* belonging to that gen co

		*Used to subset GenFleet and PPdb.
	"""
	# ---------------------------------------------------------------------------- 1) Initialize the GenUnit instances
	# -- set_PPdb() was called in _subinit2
	pp.GenUnit.init_fleet()
	GenFleet = pp.GenUnit.get_fleet()

	# ---------------------------------------------------------------- 2) GenCos {gen co : iterable of GenUnit indices}
	# These indices --> PPdb, GenUnit
	Ser_gencos = PPdb['master']['Owner/Operator']
	GenCos = {genco: (Ser_gencos.loc[Ser_gencos == genco]).index for genco in Ser_gencos.unique()}

	print("Market has {} gen cos with {} GenUnit(s) total.".format(len(GenCos), len(GenFleet)))
	logging.info("Initialized {} GenUnit(s) from the power plant database at '{}'.\n".format(
		len(GenFleet), PATHS['PP database']))

	# ------------------------------------------------------------------------------ 3) GenUnit settings from config
	# All settings except for 'currency' have to use eval()
	pp.GenUnit.settings['currency'] = config['market settings']['currency']

	GenUnit_configsettings = {key: eval(config['market settings'][key]) for key in pp.GenUnit.settings.keys()
	                          if key in config['market settings'] and key != 'currency'} # 'currency' in prev line

	if GenUnit_configsettings:
		pp.GenUnit.settings.update(GenUnit_configsettings)
		logging.info("GenUnit settings from config:\n{}\n".format(fmtdict(GenUnit_configsettings)))

	return GenFleet, GenCos


def subinit5_preprocess(config, dFundamentals, dGenParameters, dParameters, GenFleet, PPdb):
	"""Preprocessing of input data:
		Data                                To Do
		1) Efficiency curves                Interpolate and if necessary, clip to [0, 1]

		2) fundamentals, metadata.ini       Prep __time:  Get common time of fundamentals; process market period info
											Start dStats

		3) Fuel conversion units            Calculate

		4) Gen efficiency & reliability     Create rand vars (n_e_FL, T, D)

		5) latent heat factors              Calculate

		6) Bindings to GenUnits             eff curve interpolants,
											fuel prices Metadata obj
											fconv Pint
											cogen alt boiler eff
											rand vars
											latent heat factors

	DEV NOTES:
		The interpolant, bound to more_md of the efficiency curves data, has Po (pu) --> normalized efficiency,
		both of which are bound [0, 1].

	"""
	# ------------------------------------------------------------------------------- 1) Efficiency curves
	interpolate_efficiency_curves(effcurves=dGenParameters['efficiency curves'])

	# ------------------------------------------------------------------------------- 2) Prepare __time
	time, dStats = calc_timeresources(config, dFundamentals)

	# ------------------------------------------------------------------------------- 3) Fuel conversion factors
	logging.info("Calculating fuel conversion factors (fqty to MWh)")
	dParameters['fuel qty to MWh'], dfdiag = convertFuelUnits(dFundamentals['fuel prices'], to='MWh',
	                                                          Prms_HHV=dParameters['HHV'],
	                                                          Prms_density=dParameters['fuel densities'],
	                                                          get_diagnostics=True, log=True)
	logging.info("Successfully converted the ff. fuels as:\n\n{}".format(dfdiag))

	# ------------------------------------------------------------------------------- 4) Process stochastic elements

	# ........................................................... i) Full load eff
	for gen in GenFleet:
		gen.rv['n_e_FL'] = create_rand_var(vartype=gen.gentech['Efficiency rand var*'],
		                                   typical=gen.gentech['FL efficiency [%]']/100)

	# .......................................................... ii) T and D (up and down time duration)
	process_outage_info(config, GenFleet)

	# ------------------------------------------------------------------------------- 5) Latent heat percentage calc
	fuel_presets = {
		'Orimulsion'    : 'Crude',
		'oil'           : 'Crude',
		'clean coal'    : 'WB Coal (Australia)',
		'heavy fuel oil': 'HSFO',
	}
	calc_latentheat_factors(dParameters, PPdb['master']['Fuel'], fuel_presets)

	# ------------------------------------------------------------------------------- 6) Bindings to GenUnits
	for gen in GenFleet:
		# a) fuel price (WtE's have no fuel price, hence .get())
		if not gen.is_['WtE']:
			gen.gentech['Fuel Price'] = dFundamentals['fuel prices'][gen.gentech['Fuel ID*']]

		# b) eff curve unbounded interpolant
		effkey = gen.gentech['Efficiency Curve*']
		gen.misc['unbounded eff curve interpolant'] = dGenParameters['efficiency curves'][effkey].more_md[
			'interpolant']

		# c) fconv factors
		if not gen.is_['WtE']:
			gen.gentech['Fuel Conversion to MWh'] = dParameters['fuel qty to MWh'][gen.gentech['Fuel ID*']]

		# d) alternative boiler efficiencies
		if gen.is_['Cogen']:
			gen.cogentech['Alternative boiler efficiency [%]'] = fuzzyLookUp(gen.gentech['Fuel ID*'],
			                                                                 dParameters['boiler efficiencies'].val)

		# e) latent heat perc
		lheat_key = gen.gentech['Fuel'].split(',')[0]
		gen.gentech['Fuel latent heat per HHV'] = dParameters['latent heat factors'][lheat_key]

		# Final - re-bind aliases
		gen.bind_aliases()


	return time, dStats


def reinit_fleet():
	"""Used in the calibration process to re-initialize the fleet with new parameters. Must update the PPdb first,
	before calling this routine. Returns the new fleet (updates the underlying fleet in PowerPlants.GenUnit,
	so the returned fleet is just for convenience.).

	Calls the necessary routines from gd_core subinit 4 and 5. Updates the GenFleet module name. This routine
	changes GenFleet only (all other work space names are untouched).
	"""
	# ------------------------------------------------------------------------ subinit4 step 1) Init fleet
	# Renews the global name GenFleet, but earlier imports of gd_core would still have the previous fleet.
	# Thus, the modules that borrow from gd_core should call pp.GenUnit.get_fleet()
	global GenFleet
	pp.GenUnit.init_fleet()
	GenFleet = pp.GenUnit.get_fleet()

	# ------------------------------------------------------------------------ subinit5 step 4) Init rand vars
	# ........................................................... i) Full load eff
	for gen in GenFleet:
		gen.rv['n_e_FL'] = create_rand_var(vartype=gen.gentech['Efficiency rand var*'],
		                                   typical=gen.gentech['FL efficiency [%]'] / 100)

	# .......................................................... ii) T and D (up and down time duration)
	process_outage_info(config, GenFleet)

	# ------------------------------------------------------------------------ subinit5 step 5) Bindings to GenUnits
	for gen in GenFleet:
		# a) fuel price (WtE's have no fuel price)
		if not gen.is_['WtE']:
			gen.gentech['Fuel Price'] = dFundamentals['fuel prices'][gen.gentech['Fuel ID*']]

		# b) eff curve unbounded interpolant
		effkey = gen.gentech['Efficiency Curve*']
		gen.misc['unbounded eff curve interpolant'] = dGenParameters['efficiency curves'][effkey].more_md[
			'interpolant']

		# c) fconv factors
		if not gen.is_['WtE']:
			gen.gentech['Fuel Conversion to MWh'] = dParameters['fuel qty to MWh'][gen.gentech['Fuel ID*']]

		# d) alternative boiler efficiencies
		if gen.is_['Cogen']:
			gen.cogentech['Alternative boiler efficiency [%]'] = fuzzyLookUp(gen.gentech['Fuel ID*'],
			                                                                 dParameters['boiler efficiencies'].val)

		# Final - re-bind aliases
		gen.bind_aliases()

	print('GenFleet re-initialized.')
	return GenFleet


def subinit6_patches(PATHS, metadata, dFundamentals):
	"""Experimental code

	Currently, this will read the WtE monthly ave load schedule. WtE plant costs are not modelled in this version.

	Note: Currently, reinit_fleet() doesn't have to call this.
	"""
	# Monthly ave MW of WtE plants. This Po is bid at ~0$.
	#dFundamentals['WtE Sched'] = pd.read_pickle(os.path.join(PATHS['Fundamentals'], 'WtE Monthly Ave Load.pkl'))
	dh.set_readpath(PATHS['Fundamentals'])
	dFundamentals['WtE Sched'] = eval("dh.Metadata(report='log', {})".format(metadata['others']['waste_sched']))
	return




# -------------------------------------------------------------------------- Routines used by subinit5_preprocess()
def interpolate_efficiency_curves(effcurves):
	"""Interpolates the efficiency curves in dGenParameters['efficiency curves'], and binds the interpolant to
	more_md"""
	logging.info("Interpolating the efficiency curves.\n")
	for key, effc_md in effcurves.items():
		effc = effc_md.val
		# --------------------------------------------------- 1.1) Data checks
		if not ((0 <= effc['Load pu']).all() and (effc['Load pu'] <= 1).all()):
			raise ValueError("'Load pu' column in efficiency curves must be within [0, 1]. "
			                 "Failed curve: {}".format(key), key)

		if not ((0 <= effc['Part load efficiency']).all() and (effc['Part load efficiency'] <= 1).all()):
			raise ValueError("'Part load efficiency' column in efficiency curves must be within [0, 1]. "
			                 "Failed curve: {}".format(key), key)

		if effc.loc[(effc['Load pu'] == 1) & (effc['Part load efficiency'] == 1)].shape != (1, 2):
			raise ValueError("The efficiency curves should have the point (1.0, 1.0). "
			                 "Failed curve: {}".format(key), key)

		if not dkn.is_xymonotonic(effc['Load pu'], effc['Part load efficiency'], slope='pos')[0]:
			raise ValueError("Efficiency curves have to be monotonically increasing. "
			                 "Failed curve: {}".format(key), key)

		# --------------------------------------------------- 1.2) Interpolate
		interpolant = interp1d(x=effc['Load pu'], y=effc['Part load efficiency'],
		                       kind='linear', fill_value='extrapolate', bounds_error=False, assume_sorted=True)

		# Clip if lowest P goes out of bounds
		if interpolant(0) < 0:
			interpolant = dkn.clip(interpolant, lb=0)
			logging.warning("Efficiency curve'{}' goes out of bounds when extrapolating to 0. Interpolant "
			                "clipped.".format(key))

		# --------------------------------------------------- 1.3) FINAL -- bind to more_md
		effc_md.more_md['interpolant'] = interpolant
	return


def calc_timeresources(config, dFundamentals):
	"""Calculates the time resources used in the simulation. Returns time and dStats"""
	time = {}
	dStats = {}
	# -------------------------------------------------------------------- 1) Get common period
	#                                                                           __time['common period']
	logging.info("Seeking the common period in fundamentals input.")
	# Filter time-oriented price data via time_params attr
	ts, te = dh.get_common_period(*(fund for fund in dkn.get_dictvals(dFundamentals) if fund.time_params), report=None)

	logging.info("dh.get_common_period() returns: ({}, {})".format(ts, te))

	if (ts, te) == (None, None):
		print("ERROR: No common period found in the input fundamentals. Pls. double check your input data.")
		raise RuntimeError
	else:
		if not isinstance(ts, pd.Timestamp) or not isinstance(te, pd.Timestamp):
			raise NotImplementedError("Assumes that DataHandler.get_common_period() returns pandas.Timestamp. "
			                          "Pls. check date coercions that follow.")

		# Note that __time['common period'] and __time['simulation period'] describe durations in DISCRETE DAYS (
		# datetime.date). They are INCLUSIVE.
		# The equivalent, continuous version is [__time['common period'] [0], __time['common period'] [1]+1day)

		ts_dt, te_dt = ts.date(), te.date()  # tx_dt <= tx;  pd.Timestamp(D) --> continuous D

		# ts -- if ts in (D,D+t), shift to D+1
		if pd.Timestamp(ts_dt) < ts:
			ts_dt += dttm.timedelta(1)

		# te -- if te in [D, D+1-mp), shift to D-1
		if te < (pd.Timestamp(te_dt) + pd.Timedelta('1d') - pd.Timedelta(config['market settings']['period_duration'])):
			te_dt -= dttm.timedelta(1)

		msg = "Found common period ({} days) in the fundamentals data.\n\tFrom: \t{}\n\tTo: \t{}\n".format(
			(te_dt - ts_dt).days + 1, ts_dt.strftime("%Y %b %d"), te_dt.strftime("%Y %b %d"))
		print(msg)
		logging.info(msg)

		time['common period'] = ts_dt, te_dt

	# -------------------------------------------------------------------- 2) Period duration and no
	try:
		DxP_freq = config['market settings']['period_duration']
		P_duration = pd.Timedelta(DxP_freq)
	except:
		warnings.warn("Failed to interpret config['market settings']['period_duration']. Use the default 1-hour "
		              "market period duration instead.")
		DxP_freq = '1H'
		P_duration = pd.Timedelta(DxP_freq)

	N_periods = pd.Timedelta('1D') / P_duration

	if not N_periods.is_integer():
		raise ValueError("The number of periods in one day implied by config['market settings']['period_duration'] = "
		                 "{} is not whole.".format(config['market settings']['period_duration']))

	time.update({
		'periods per D'   : int(N_periods),
		'periods duration': P_duration,
		'DxP_freq'        : DxP_freq,
	})
	logging.info("market period duration: {}".format(time['periods duration']))


	# -------------------------------------------------------------------- 3) User-set variables
	time.update({
		'simulation period': [None, None],
		'D_index'          : None,
		'DxP_index'        : None,
	})

	# -------------------------------------------------------------------- 4) Time starts and ends in dStats

	timefunds = tuple(fund for fund in dkn.get_dictvals(dFundamentals) if fund.time_params)
	_df = pd.DataFrame(index=(itm.defn_short for itm in timefunds))
	_df['Start'] = pd.Series(data=(itm.time_params['t_start'] for itm in timefunds), index=_df.index)
	_df['End'] = pd.Series(data=(itm.time_params['t_end'] for itm in timefunds), index=_df.index)
	_df.loc['COMMON', ('Start', 'End')] = time['common period']
	dStats['Fundamentals Time'] = _df

	return time, dStats


def convertFuelUnits(fueldict, to, Prms_HHV, Prms_density, get_diagnostics=False, log=False):
	"""Calculates fuel unit conversion factors. Performs the ff. translation:
		{fuel: fuel quantity basis (str) or Metadata of fuel price*} --> {fuel: conversion factor = common unit / fuel
		quantity basis}

	This is used by subinit5_preprocess() but can also be used externally.

	DEPENDENCIES:
		DataHandler.Metadata
		Pint 0.9

	PARAMETERS:
		fueldict       {fuel: fuel quantity basis (str) or Metadata of fuel price*}
		to             Common unit as str.
		Prms_HHV       Table of HHV values, as a Metadata parameters object (explicit units)**
		Prms_density   Table of fuel densities, as a Metadata parameters object (explicit units)**

	 Note:
	    *If Metadata, then if the HHV and/or density are needed, more_md is first checked and takes priority over
	    matches in Prms_HHV and Prms_density, respectively.

	    **These tables support a fuzzy lookup, meaning the fuel key does not have to exactly match the keys in the
	    parameters table. This is implemented with the FuzzyWuzzy process module.

	 RETURNS:
	 	fuel conversion factors {fuel: conversion factor = common unit / fuel quantity basis}

        if get_diagnostics=True, then a DataFrame of diagnostics is returned as the 2nd value. Indexed by the fuels,
        it contains information on:
            - the detected dimensionality of the fuel quantitiy (either energy, mass or volume)
            - if an HHV was sought, then which key in the HHV table matched it
            - if a density was sought, then which key in the densities table matched it


	Example:
		res = convertFuelUnits(myfuels, to='MWh', dParameters['HHV'], dParameters['fuel densities'])
	"""
	# --------------------------------------------------------------------------------------- 0a) Init output
	fconv = {}
	dfdiag = pd.DataFrame(index=fueldict.keys(), columns=('Fuel denominated by', 'HHV Used', 'rho Used'))
	dfdiag.index.name = 'Fuel ID'

	# --------------------------------------------------------------------------------------- 0b) Get Pint and check to
	ureg, DimensionalityError = dh.Metadata.get_UHaux('ureg', 'DimensionalityError')

	try:
		ureg(to).to('J')
	except DimensionalityError:
		raise ValueError("Must pass a unit of energy to parameter 'to'.")


	# --------------------------------------------------------------------------------------- 1) Fuel conversion
	for fuelID, value in fueldict.items():
		# ........................................................... 1.1) Parse value to get units
		isMd = isinstance(value, dh.Metadata)

		if isMd:
			if 'fuel qty' in value.more_md:
				# unit str has been prepared
				fqty = value.more_md['fuel qty']
			else:
				fqty = ureg(str(value.units).split(' / ')[1])

		elif isinstance(value, str):
			fqty = ureg(value)

		else:
			raise TypeError('Cannot interpret values in fueldict')

		# ........................................................... 1.2a) Try as if per energy
		try:
			fconv[fuelID] = fqty.to(to) / fqty      # Past this conversion means success
			dfdiag.loc[fuelID, 'Fuel denominated by'] = 'energy'

			continue

		except DimensionalityError:
			pass

		# ........................................................... 1.2b) Try as if per mass
		try:
			# Calc HHV - Fuzzy lookup from HHV and density tables (standard), but allows parameters to be
			# encoded in the Metadata object (checks .more_md first)
			# Note -- must use __getitem__ of Metadata to return Pint
			HHV = None
			if isMd and 'HHV' in value.more_md:
				HHV = ureg(value.more_md['HHV'])
				match = 'more_md'

			if HHV is None:
				match = fuzzyLookUp(fuelID, Prms_HHV.val, get='key', NoMatch='KeyError')
				HHV = Prms_HHV[match]

			dfdiag.loc[fuelID, 'HHV Used'] = match

			fconv[fuelID] = (fqty * HHV).to(to) / fqty
			dfdiag.loc[fuelID, 'Fuel denominated by'] = 'mass'

			continue

		except KeyError:
			msg_p1 = "Need the HHV value (per kg) of fuel '{}'".format(fuelID)
			if log: logging.error("{}, but it is not available in the HHV table.".format(msg_p1))
			print("ERROR: {}".format(msg_p1))
			raise

		except DimensionalityError:
			pass

		# ........................................................... 1.2c) Try as if per volume
		try:
			# Fuzzy lookup of density, with more_md try. Note -- must use __getitem__ of Metadata to return Pint
			rho = None

			if isMd and 'density' in value.more_md:
				rho = ureg(value.more_md['density'])
				match = 'more_md'

			if rho is None:
				match = fuzzyLookUp(fuelID, Prms_density.val, get='key', NoMatch='KeyError')
				rho = Prms_density[match]

			dfdiag.loc[fuelID, 'rho Used'] = match

			fconv[fuelID] = (fqty * rho * HHV).to(to) / fqty
			dfdiag.loc[fuelID, 'Fuel denominated by'] = 'volume'

			continue

		except KeyError:
			msg_p1 = "Need the density of fuel '{}'".format(fuelID)
			if log: logging.error("{}, but it is not available in the densities table.".format(msg_p1))
			print("ERROR: {}".format(msg_p1))
			raise

		except DimensionalityError:
			# This time, this is really an error.
			raise NotImplementedError(
				'Fuel prices can only be denominated in terms of a) energy content (e.g. HHV MMBtu), b) mass (e.g. '
				'ton) or c) volume (e.g. bbl). Also, pls. check if the unit was properly defined.')

	# --------------------------------------------------------------------------------------- 2) Return
	if get_diagnostics:
		return fconv, dfdiag
	else:
		return fconv


def process_outage_info(config, GenFleet):
	"""
	Used by subinit5, this does two things:
		i) Creates the T and D random vars                  - self.rv['T'], self.rv['D']
		ii) Calculates the limiting availability            - self.Reliability['Limiting Availability']

	All changes are written to GenFleet.

	Note:
		The T and D random vars are the up and down time duration exponential variables, respectively. Their
		lambda parameters are obtained from the 'Average Failures per year [yr-1]' and 'Mean Time to Repair [wks]'
		values in self.reliability, respectively. These are created per GenUnit, and sampled at pre-simulation to set
		the availability schedule. These are sampled with the random seed in the PPdb rand seed sheet (columns 'UP Time
		Duration' and 'DOWN Time Duration').

	Reference:
		Introduction to Repairable Systems Modeling (Cassady, C Richard; Pohl, Edward A)
		Homogeneous Poisson Process (HPP) https://www.itl.nist.gov/div898/handbook/apr/section1/apr171.htm
	"""
	# ------------------------------------------------------------------------------- 1) Get Pint and set market period
	Q_ = dh.Metadata.get_UHaux('Q_')
	mp_hrs = config['market settings']['period_duration'].rsplit('H', maxsplit=1)[0]
	mp = Q_('{} hr'.format(mp_hrs))

	# ------------------------------------------------------------------------------- 2) Loop through fleet
	for gen in GenFleet:
		# ................................................................ 2.1) Calculate lambda_mp, mu_mp and
		#                                                                       availability
		lambda_mp, mu_mp, Alim = calc_lambda_mu_A(AF_yr_prm=gen.reliability['Average Failures per year [yr-1]'],
		                                          MTTR_wk_prm=gen.reliability['Mean Time to Repair [wks]'],
		                                          mp=mp)

		# todo -- possibly add MTTF in weeks
		gen.reliability['T lambda]'] = lambda_mp
		gen.reliability['D mu'] = mu_mp
		gen.reliability['Limiting Availability'] = Alim
		gen.reliability['Mean Time to Fail [wks]'] = Alim/(1-Alim)*gen.reliability['Mean Time to Repair [wks]']
		

		# ................................................................ 2.1) Create the rand vars
		gen.rv['T'] = create_rand_var(vartype='exponential', exp_lambda=lambda_mp)
		gen.rv['D'] = create_rand_var(vartype='exponential', exp_lambda=mu_mp)

	return


def create_rand_var(vartype, **kwargs):
	"""Creates a random variable (scipy.stats variable) from a list of predefine variables. These are parameterized
	by the keyword arguments. The ff. table lists these variables and the necessary parameters

	DEFINED VARIABLES:

		|     VARTYPE     |        DESCRIPTION & PARAMETERS
		...............................................................................................................
		eff_skewed              A random variable representing efficiency, described by a typical / very good
								efficiency but skewed to allow lower efficiencies. This is modelled with a skewed
								normal distribution (implemented with scipy.stats.skewnorm)

								typical         The typical / good efficiency (accepts 0.05-0.85) at full load. This
												would be larger than the median. Under the default parameters,
												the median and 99th percentile will just be 2% efficiency points apart.

								alpha           (Optional; defaults to -10) Alpha parameter of stats.skewnorm
								scale           (Optional; defaults to 0.02) Scale parameter of stats.skewnorm

		...............................................................................................................
		exponential             A generic exponential random variable, parametrized by lambda. Implemented with
								scipy.stats.expon.

								exp_lambda      The lambda parameter of the exponential distribution.

		...............................................................................................................
	"""
	if vartype == 'eff_skewed':
		if 'typical' not in kwargs:
			raise TypeError("Pls. specify the typical good efficiency via parameter 'typical'.")
		if not 0.05 <= kwargs['typical'] <= 0.85:
			raise ValueError("The entered full load typical efficiency is not realistic.")

		return stats.skewnorm(a=kwargs.get('alpha', -10), loc=kwargs['typical'], scale=kwargs.get('scale', 0.02))

	elif vartype == 'exponential':
		if 'exp_lambda' not in kwargs:
			raise TypeError("Pls. specify the 'lambda' parameter of the exponential distribution.")

		return stats.expon(scale=1/kwargs['exp_lambda'])

	else:
		raise ValueError("Undefined vartype.")


def calc_lambda_mu_A(AF_yr_prm, MTTR_wk_prm, mp):
	"""Calculates lambda, mu (in market periods, w/o units) and the availability limit, and used by subinit5 /
	process_outage_info(). Note that mp is a Pint quantity in hours. See 'SG Plant Dispatch Model Blueprints.docx',
	Ch3 Sec Plant outages for details."""
	Q_ = dh.Metadata.get_UHaux('Q_')
	# ------------------------------------------------------- 1) Params as pint
	AF_yr = Q_('{}/year'.format(AF_yr_prm))
	MTTR_wk = Q_('{} week'.format(MTTR_wk_prm))

	# ------------------------------------------------------- 2) mu as pint [market period-1]
	mu_mp = (1/MTTR_wk.to('hr')) * mp

	# ------------------------------------------------------- 3) lambda as pint [market period-1]
	MTTR_yr = MTTR_wk.to('year')
	MTTF_yr = 1/AF_yr - MTTR_yr
	lambda_mp = (1/ MTTF_yr.to('hr')) * mp

	# ------------------------------------------------------- 4) limiting availability as pint [-]
	Alim = MTTF_yr / (MTTF_yr + MTTR_yr)
	assert 0 <= Alim <= 1, "Computed probability is out of bounds"

	# ------------------------------------------------------- 5) return (magnitudes only)
	return lambda_mp.magnitude, mu_mp.magnitude, Alim.magnitude


def calc_latentheat_factors(dParameters, PPdb_fuels, fuel_presets=None):
	"""Calculates the latent heat factors [latent heat per unit HHV fuel] as (HHV-LHV)/HHV

	PARAMETERS:
		dParameters         As in module-level name
		fuel_presets        (Optional; defaults to empty dict)Synonyms {fuel A: fuel B} to apply to PPdb_fuels prior to
							looking them up in the HHV & LHV tables.
		PPdb_fuels          'Fuels' column of the master sheet of PPdb

	RETURNS:
		None. Results are written to dParameters['latent heat factors']
	"""
	# 1) Get set of fuels from PPdb_fuels
	# For multi fuels separated by commas, get first
	getfuels = set(fuel.split(',')[0] for fuel in PPdb_fuels)

	# 2) Match fuel keys with presets and the HHV/LHV tables
	fuel_PPdb_to_table = {fuel: fuzzyLookUp(fuel_presets.get(fuel, fuel), dParameters['LHV'].val, get='key')
	                      for fuel in getfuels}

	# 3) Calculate factors
	_HHV = dParameters['HHV']
	_LHV = dParameters['LHV']
	dParameters['latent heat factors'] = {f_PPdb: 1.0 - (_LHV[f_Tbl]/_HHV[f_Tbl]).magnitude
	                                      for f_PPdb, f_Tbl in fuel_PPdb_to_table.items()}

	return


# -------------------------------------------------------------------------- Other routines
def normalexit():
	"""Exit sequence. Logs the time."""
	logging.info("[PROGRAM NORMAL EXIT] at {}".format(log_time()))
	return


def __check_PPdispatchError(PPdispatchError, config):
	"""Aux function to raise PPdispatchError from anywhere. This is so designed so that in debug mode (defined by
	__config['system']['debug'] == 'True', the exception is only announced, and the variables can be inspected
	(including the actual exception raised, which is stored in PPdispatchError)."""
	if PPdispatchError is not None:
		try:
			if config['system']['debug'] == 'True':
				print("Exception caught and bound to PPdispatchError.")
			else:
				raise PPdispatchError
		except: # This happens when there is a problem with config
			raise PPdispatchError
	return


# -------------------------------------------------------------------------- Startup
PATHS, config, metadata, \
PPdb, dFundamentals, dGenParameters, dParameters, dStats, \
time, Scenarios, GenFleet, GenCos, PPdispatchError = __init_sequence()

# Write PATHS.pkl for convenient access
with open(os.path.join(PATHS['Proj'], 'PATHS.json'), 'w') as f:
	json.dump(json.loads(pd.Series(PATHS).to_json()), f, indent=4)


if config['system']['debug'] == 'True':
	# Explicit reminders during development
	notetoself = ('No Non-Fuel Costs in .calc_TotalCosts()',
	              'if Po < Pmin, assume that ne is that of Pmin',
	              'Cogen HPR assumed constant',
	              'WtE cost structure not implemented',
	              'Fuel presets hardcoded when calculating latent heat factors'
	              )
	notetoself = "".join("\t- {}\n".format(msg) for msg in notetoself)
	logging.warning("Development Notes: \n" + notetoself)


