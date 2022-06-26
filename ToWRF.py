"""Scripts for the thermal modelling of SG power plants and preparing WRF inputs. These scripts are not meant for
general use, but only within the SG power plant dispatch model.

Author: Pang Teng Seng, D. Kayanan
Created: April 17, 2020
Revised: Feb. 22, 2022

- Implemented height in WRF output
- Implemented output to DUCT
"""
from PowerPlants import *
from genDispatch import *

import pandas as pd

from os import path, mkdir
import pickle
import warnings
import configparser
import sys

# --------------------------------------------------- NEW mandatory init
PathProj = path.dirname(path.abspath(__file__))
PATHS = pd.read_json(path.join(PathProj, 'PATHS.json'), typ='series', orient='records')

# equivalent to gd_analysis.c_
config = configparser.ConfigParser()
config.read(PATHS['config'])

c_ = {
	'MWh to ktoe':  8.598452278589855e-05,
	'period hr':    pd.Timedelta(config['market settings']['period_duration']).seconds/3600,
}



# --------------------------------------------------------------------------- genDispatch init
# Todo If we contain this in a function, we need to pass the required names to the module namespace
# GenFleet, dParameters

def init_genDispatch():
	""""""
	# Define calc_allheat() here, so that it is available to read_results()
	global calc_allheat
	# Note: Setting the scenario is not necessary, because this is used only during market simulation in NEMS.py
	print('Initializing genDispatch\n')

	# deleted sys.path.append()

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		import gd_core as __core
		from gd_analysis import calc_allheat
		from genDispatch import set_period

	GenFleet = GenUnit.get_fleet()
	PPdb = __core.PPdb
	dParameters = __core.dParameters

	# Call set period to prep __time (used in calc_allheat())
	set_period()
	return GenFleet, PPdb, dParameters


def init_genDispatch_old():
	""""""
	# Note: Setting the scenario is not necessary, because this is used only during market simulation in NEMS.py
	print('Initializing genDispatch\n')
	import sys
	PathProj = path.dirname(path.abspath(__file__))
	sys.path.append(PathProj)


	with warnings.catch_warnings():
		# Suppress the warning upon importing fuzzywuzzy (no speedup)
		warnings.simplefilter("ignore")
		#from genDispatch import *
		# Todo cant import * at the sub-module level
		# NOTE Do NOT overwrite c_
		from DK_Collections import fuzzyLookUp

	# Call set period to prep __time
	set_period()
	return
# --------------------------------------------------------------------------- #

def thermal_analysis(GenFleet, dParameters):
	"""Computes the energy flows of thermal power plants. Sets the ff. attributes:
		self.thermo['flows']
		self.thermo['losses']
		self.thermo['metrics']

	This is need only once, and is done after initializaing genDispatch
	"""
	HHV_LHV_ratios = dParameters['HHV'].val/dParameters['LHV'].val

	fuel_presets = {
		'Orimulsion'    : 'Crude',
		'oil'           : 'Crude',
		'clean coal'    : 'WB Coal (Australia)',
		'heavy fuel oil': 'HSFO',
	}

	compiled_prms = {}



	for gen in GenFleet:
		# Get relevant gen info
		_pm = gen.gentech['Prime Mover']
		n_e = gen.gentech['FL efficiency [%]'] / 100

		if gen.is_['Cogen']:
			n_tot = gen.cogentech['Heat & power FL efficiency [%]'] / 100
			HPR_net = (n_tot - n_e) / n_e
		else:
			n_tot, HPR_net = None, None

		gen_fuel = gen.gentech['Fuel']
		# fuzzyLookUp() is brought in via PowerPlants.py
		HHV_LHV_r = fuzzyLookUp(fuel_presets.get(gen_fuel, gen_fuel), HHV_LHV_ratios, get='val')

		# Initialize gen.thermo
		gen.thermo = {}

		if _pm == 'CCGT':
			# i) Adjust aux_load if highly efficient
			if n_e > 0.51:
				_args = {
					'aux_load': 0.01,
					'x_GT_misc': 0.002,
					'x_HRSG_misc': 0.001,
					'x_ST_misc': 0.002,
				}
			else:
				_args = {}

			# ii) thermal calc
			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_CCGT_FLflows(
				net_HHVeff=n_e,
				GT_ST_ratio=2,
				eff_ST=0.45,
				HHV_LHV_ratio=HHV_LHV_r,
				**_args
			)

		elif _pm == 'Cogen CCGT':
			# Note: Pushing down losses bec 83% thermal eff is too high (probably LHV-basis)
			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_CogenCCGT_BPST_FLflows(
				net_HHVeff=n_e,
				HPR_net=HPR_net,
				HPR_bp=14,
				aux_load=0.01,
				x_GT_misc=0.001,
				x_HRSG_misc=0.001,
				x_ST_misc=0.001,
				HHV_LHV_ratio=HHV_LHV_r,
				min_Stack_StoL=0.6,
			)

		elif _pm == 'Cogen Extraction ST':
			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_CogenExtST_FLflows(
				net_HHVeff=n_e,
				HPR_net=HPR_net,
				eff_ST=0.47,
				aux_load=0.03,
				x_SG_otherlosses=0.025,
				HHV_LHV_ratio=HHV_LHV_r,
			)

		elif _pm == 'OCGT':
			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_OCGT_FLflows(
				net_HHVeff=n_e,
				HHV_LHV_ratio=HHV_LHV_r,
			)

		elif _pm == 'Oil-fired ST':
			if n_e > 0.33:
				_args = {
					'eff_ST': 0.46,
					'x_SG_otherlosses': 0.006,
					'x_ST_misc': 0.001,
					'aux_load': 0.02,
				}
			else:
				_args = {
					'eff_ST': 0.45,
					'x_SG_otherlosses': 0.007,
				}

			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_ST_FLflows(
				net_HHVeff=n_e,
				HHV_LHV_ratio=HHV_LHV_r,
				**_args
			)

		elif _pm == 'WtE ST':
			gen.thermo['flows'], gen.thermo['losses'], gen.thermo['metrics'] = calc_ST_FLflows(
				net_HHVeff=n_e,
				eff_ST=0.35,
				x_SG_otherlosses=0.03,
				HHV_LHV_ratio=HHV_LHV_r,
				aux_load=0.08, # TIP WtE consumes 35% of the elec it produces; this is a closer est.
			)
		else:
			raise NotImplementedError

		# Assertions
		assert abs(gen.thermo['metrics'].at['η Overall(HHV)'] - n_e) < 10**-6
		assert not gen.is_['Cogen'] or abs(gen.thermo['metrics'].at['HPR(net)'] - gen.cogentech['HPR [-]']) < 10**-6

		# Compile results into tables for inspection
		if (_pm, 'flows') not in compiled_prms:
			compiled_prms[_pm, 'flows'] = pd.DataFrame()
			compiled_prms[_pm, 'losses'] = pd.DataFrame()
			compiled_prms[_pm, 'metrics'] = pd.DataFrame()

		for key in ('flows', 'losses', 'metrics'):
			compiled_prms[_pm, key][gen.id['rowID']] = gen.thermo[key]

	return compiled_prms

from DK_Collections import fmtdict
import pickle

def read_results(GenFleet):
	"Reads dispatch results and writes schedules to gen.Schedule. Also calls calc_allheat(Results)"
	fname_choices = {
		'Base'  : 'Baseline demand and proxy fuel prices (27.02.20).pkl',
		'FullEV': 'Full road transport electrification (04.03.20).pkl',
	}

	fname = fname_choices[input("\nChoose among:\n{}\n\n\tAns:".format(fmtdict(fname_choices)))]

	#print(fname_choices)
	#print(fmtdict(fname_choices))
	#print("\nChoose among:\n{}\n\n\tAns:".format(fmtdict(fname_choices)))
	#print(fname_choices[input()])
	#fname=fname_choices[input()]
    
	print("file name:"+fname)
	print("Paths:"+PATHS['Results'])
	print("Paths   :"+path.join(PATHS['Results'], fname))
	print('\n')
	fp = path.join(PATHS['Results'], fname)
	print(fp)
	#new_data={}
	#print(type(new_data)) 
	#with open(fp, 'rb') as f: 
		#print(f)
		#print(type(f))        
		#print(pickle.load(f))
		#print(type(pickle.load(f)))
		#new_data = pickle.load(f)
		#data1 = f.readline()
	#f.close()    
    
	#print(pickle.load(open(fp, "rb")))
	#pickle.load(open(fp, "rb")
	Results = pickle.load(open(fp, "rb"))
	calc_allheat(Results)

	# Results['Schedules'] = tuple(gen.Schedule for gen in GenFleet) <--- in post_simulation()
	for idx, sched in enumerate(Results['Schedules']):
		GenFleet[idx].Schedule = sched
                
	return Results, GenFleet


def calc_heatstreams(GenFleet, Results, by='outlet', latentheatfactors=None):
	"""Calculates the power plant heat streams by outlet or by kind [MW per period]

	Requirements:
		1) Dispatch results as encoded in gen.Schedule
		2) Power plant energy flows as calculated by thermal_analysis()

	PARAMETERS:
		GenFleet        as in genDispatch
		Results         results directory as in genDispatch

		by              (Optional; defaults 'outlet') Use to specify which heat streams to calculate.
							by='outlet'         Stack, condenser, misc+aux loads
							by='kind'           Sensible air, latent air, seawater

		latentheatfactors   (Required only if by='kind') Latent heat factors in dParameters['latent heat factors']

	RETURNED:
		None. Results are written to GenFleet members (attribute .heatstreams_byoutlet or .heatstreams_bykind),
		in MW values per period.


	PSEUDOCODE:
		0) gen.Schedule is a 2D array with:
			axis 0 as time
			axis 1 as _Available, _Po, _ne, _ne_pot (and _Qo, if cogen)

		1) If the plant has Po=0, then there is also no heat. So the first step is to filter the periods that the plant
		is active,
			Lf_active = gen.Schedule[:, _Po] != 0

		2) Extract the Po and the achieved efficiency ne (not ne_pot) via Lf_active.
		Calculate these:
			input fuel HHV
			scalar for gen.thermo['losses']	= (1-losses, actual)/(1-losses, nominal)
				losses pertains to waste heat only (i.e. if cogen, must account for cogen heat)

		2) The results container 'res' will be initialized to all 0, with the same axis 0 as gen.Schedule, and with
		axis 1 as indicated in the 'by' param above.



	"""
	# gen.Schedule identifiers
	_Available, _Po, _ne, _ne_pot, _Qo = 0, 1, 2, 3, 4

	by_outlet = by == 'outlet'
	DxP_index = Results['dispatch [MW]'].index

	# check if calc_allheat() has been called
	if 'Total PP waste heat [ktoe]' not in Results:
		raise RuntimeError('Pls. call gd_analysis.calc_allheat() on the results before running this function.')


	for gen in GenFleet:
		# 1) Init results to zero (will stay 0 if Po=0 in those indices)
		res = np.zeros((gen.Schedule.shape[0], 3), dtype='f8')

		# 2) get non-zero Po arrays
		Lf_active = gen.Schedule[:, _Po] != 0

		# Proceed only if dispatched
		if Lf_active.sum() != 0:
			ne_FL = gen.gentech['FL efficiency [%]'] / 100
			# Po, ne (achieved), Fuel input HHV [MW]
			gen_Po = gen.Schedule[Lf_active, _Po]
			gen_ne = gen.Schedule[Lf_active, _ne]
			gen_FuelHHV = gen_Po / gen_ne

			# Cogen particulars
			if gen.is_['Cogen']:
				gen_Qo = gen.Schedule[Lf_active, _Qo]
				gen_losses = 1.0 - gen_ne*(1+gen.cogentech['HPR [-]'])
				gen_nomlosses = 1.0 - gen.cogentech['Heat & power FL efficiency [%]']/100
			else:
				gen_Qo = 0.0
				gen_losses = 1.0 - gen_ne
				gen_nomlosses = 1.0 - ne_FL


			# 3) Calculate scaling factor
			k = gen_losses/gen_nomlosses

			# 4) Apply scaling factor to nominal gen.thermo['losses'] and convert to MW
			heatstr_byoutlet_per_fuelHHV = np.outer(k, gen.thermo['losses'].to_numpy())
			heatstr_MW = np.zeros_like(heatstr_byoutlet_per_fuelHHV)

			if by_outlet:
				for idx in range(3):
					heatstr_MW[:, idx] = gen_FuelHHV * heatstr_byoutlet_per_fuelHHV[:, idx]
			else:
				# calc heatstr_bykind_per_fuelHHV from heatstr_byoutlet_per_fuelHHV
				fuel_key = gen.gentech['Fuel'].split(',')[0]
				LH_factor = latentheatfactors[fuel_key]

				# _toheatstr_bykind()
				heatstr_bykind_per_fuelHHV = np.apply_along_axis(_toheatstr_bykind,
					axis=1, arr=heatstr_byoutlet_per_fuelHHV, coolingsys=gen.coolingsys,
					latentheatfactor=LH_factor)

				for idx in range(3):
					heatstr_MW[:, idx] = gen_FuelHHV * heatstr_bykind_per_fuelHHV[:, idx]


			# Compare sum of heat streams per period == total heat
			assert np.allclose(heatstr_MW.sum(axis=1), gen_FuelHHV-gen_Po-gen_Qo), gen.id['rowID']

			# 5) Write results
			# First, to subset
			res[Lf_active, :] = heatstr_MW

		# Then, to entire space
		if by_outlet:
			gen.heatstreams_byoutlet = pd.DataFrame(data=res, index=DxP_index, columns=gen.thermo['losses'].index)
		else:
			gen.heatstreams_bykind = pd.DataFrame(data=res, index=DxP_index,
			                                      columns=['sensible, air', 'latent, air', 'seawater'])


	# ------------------------------------------------------------------------------ Assertions
	if by_outlet:
		_attr = 'heatstreams_byoutlet'
	else:
		_attr = 'heatstreams_bykind'

	Total_wasteheat_here = sum(getattr(gen, _attr).sum().sum() * c_['period hr'] * c_['MWh to ktoe']
	                           for gen in GenFleet)
	assert abs(Total_wasteheat_here - Results['Total PP waste heat [ktoe]']) < 10**-6

	return


def calc_heatstreams1(GenFleet, Results, by='outlet', latentheatfactors=None):
    # remove the assert at last few lines
    # abs(Total_wasteheat_here - Results['Total PP waste heat [ktoe]']) < 10**-6
	"""Calculates the power plant heat streams by outlet or by kind [MW per period]

	Requirements:
		1) Dispatch results as encoded in gen.Schedule
		2) Power plant energy flows as calculated by thermal_analysis()

	PARAMETERS:
		GenFleet        as in genDispatch
		Results         results directory as in genDispatch

		by              (Optional; defaults 'outlet') Use to specify which heat streams to calculate.
							by='outlet'         Stack, condenser, misc+aux loads
							by='kind'           Sensible air, latent air, seawater

		latentheatfactors   (Required only if by='kind') Latent heat factors in dParameters['latent heat factors']

	RETURNED:
		None. Results are written to GenFleet members (attribute .heatstreams_byoutlet or .heatstreams_bykind),
		in MW values per period.


	PSEUDOCODE:
		0) gen.Schedule is a 2D array with:
			axis 0 as time
			axis 1 as _Available, _Po, _ne, _ne_pot (and _Qo, if cogen)

		1) If the plant has Po=0, then there is also no heat. So the first step is to filter the periods that the plant
		is active,
			Lf_active = gen.Schedule[:, _Po] != 0

		2) Extract the Po and the achieved efficiency ne (not ne_pot) via Lf_active.
		Calculate these:
			input fuel HHV
			scalar for gen.thermo['losses']	= (1-losses, actual)/(1-losses, nominal)
				losses pertains to waste heat only (i.e. if cogen, must account for cogen heat)

		2) The results container 'res' will be initialized to all 0, with the same axis 0 as gen.Schedule, and with
		axis 1 as indicated in the 'by' param above.



	"""
	# gen.Schedule identifiers
	_Available, _Po, _ne, _ne_pot, _Qo = 0, 1, 2, 3, 4

	by_outlet = by == 'outlet'
	DxP_index = Results['dispatch [MW]'].index

	# check if calc_allheat() has been called
	if 'Total PP waste heat [ktoe]' not in Results:
		raise RuntimeError('Pls. call gd_analysis.calc_allheat() on the results before running this function.')


	for gen in GenFleet:
		# 1) Init results to zero (will stay 0 if Po=0 in those indices)
		res = np.zeros((gen.Schedule.shape[0], 3), dtype='f8')

		# 2) get non-zero Po arrays
		Lf_active = gen.Schedule[:, _Po] != 0

		# Proceed only if dispatched
		if Lf_active.sum() != 0:                                   #sum across 17568 periods
			ne_FL = gen.gentech['FL efficiency [%]'] / 100
			# Po, ne (achieved), Fuel input HHV [MW]
			gen_Po = gen.Schedule[Lf_active, _Po]
			gen_ne = gen.Schedule[Lf_active, _ne]
			gen_FuelHHV = gen_Po / gen_ne                          #calculate  gen_FuelHHV

			# Cogen particulars
			if gen.is_['Cogen']:
				gen_Qo = gen.Schedule[Lf_active, _Qo]
				gen_losses = 1.0 - gen_ne*(1+gen.cogentech['HPR [-]'])
				gen_nomlosses = 1.0 - gen.cogentech['Heat & power FL efficiency [%]']/100
			else:
				gen_Qo = 0.0
				gen_losses = 1.0 - gen_ne
				gen_nomlosses = 1.0 - ne_FL


			# 3) Calculate scaling factor
			k = gen_losses/gen_nomlosses

			# 4) Apply scaling factor to nominal gen.thermo['losses'] and convert to MW
			heatstr_byoutlet_per_fuelHHV = np.outer(k, gen.thermo['losses'].to_numpy())  # heatstr_byoutlet_per_fuelHHV = k*gen.thermo['losses']
			heatstr_MW = np.zeros_like(heatstr_byoutlet_per_fuelHHV)

			if by_outlet:
				for idx in range(3):
					heatstr_MW[:, idx] = gen_FuelHHV * heatstr_byoutlet_per_fuelHHV[:, idx]
			else: #by 'kind'
				# calc heatstr_bykind_per_fuelHHV from heatstr_byoutlet_per_fuelHHV
				fuel_key = gen.gentech['Fuel'].split(',')[0]
				LH_factor = latentheatfactors[fuel_key]

				# _toheatstr_bykind()
				heatstr_bykind_per_fuelHHV = np.apply_along_axis(_toheatstr_bykind,
					axis=1, arr=heatstr_byoutlet_per_fuelHHV, coolingsys=gen.coolingsys,
					latentheatfactor=LH_factor)

				for idx in range(3):
					heatstr_MW[:, idx] = gen_FuelHHV * heatstr_bykind_per_fuelHHV[:, idx]


			# Compare sum of heat streams per period == total heat
			assert np.allclose(heatstr_MW.sum(axis=1), gen_FuelHHV-gen_Po-gen_Qo), gen.id['rowID']

			# 5) Write results
			# First, to subset
			res[Lf_active, :] = heatstr_MW

		# Then, to entire space
		if by_outlet:
			gen.heatstreams_byoutlet = pd.DataFrame(data=res, index=DxP_index, columns=gen.thermo['losses'].index)
		else:
			gen.heatstreams_bykind = pd.DataFrame(data=res, index=DxP_index,
			                                      columns=['sensible, air', 'latent, air', 'seawater'])


	# ------------------------------------------------------------------------------ Assertions
	if by_outlet:
		_attr = 'heatstreams_byoutlet'
	else:
		_attr = 'heatstreams_bykind'

	Total_wasteheat_here = sum(getattr(gen, _attr).sum().sum() * c_['period hr'] * c_['MWh to ktoe']
	                           for gen in GenFleet)
	#assert abs(Total_wasteheat_here - Results['Total PP waste heat [ktoe]']) < 10**-6
	abs(Total_wasteheat_here - Results['Total PP waste heat [ktoe]']) < 10**-6

	return


def _toheatstr_bykind(heatstr_byoutlet, coolingsys, latentheatfactor):
	"""Processes a single vector of heat streams by outlet (shape (3,): stack, condenser, misc+aux) into a heat
	stream by kind (shape (3,): air_sensible, air_latent, seawater ). The results is still in per unit fuel HHV.

	This is a utility function needed by calc_heatstreams()
	"""
	# col indices
	sens_air, latent_air, seawater = 0, 1, 2
	# Heat streams by outlet
	stack = heatstr_byoutlet[0]
	condenser = heatstr_byoutlet[1]
	miscaux = heatstr_byoutlet[2]
	out = np.zeros(3)

	# --------------------------------------------------------------------- 1) Stack
	out[latent_air] += min(latentheatfactor, stack)
	out[sens_air] += stack - out[latent_air]

	# --------------------------------------------------------------------- 2) Condenser
	if coolingsys == 'wet-recirculating':
		out[latent_air] += 0.75*condenser
		out[sens_air] += 0.25*condenser

	elif coolingsys == 'once-through':
		out[seawater] += condenser

	elif coolingsys == 'dry-cooling':
		out[sens_air] += condenser

	elif coolingsys =='BP-cogen':
		pass
	else:
		raise ValueError

	# if cogen BP-ST, do nothing
	# --------------------------------------------------------------------- 3) Misc and aux loads
	out[sens_air] += miscaux

	# This statement makes the execution of the np.apply_along_axis() call 10x longer
	#assert np.allclose(out.sum(), heatstr_byoutlet.sum())
	return out


def summarize_heatstreams(GenFleet):
	"""This function summarizes the heat streams calculated (first over time, and next over all gens) in ktoe. The
	returned DataFrame has the ff. structure:
		index: heat streams as 'Stack', 'Condenser', 'misc+aux', 'sensible, air', 'latent, air', 'seawater',
		'total waste heat'

		columns: 'Total' (represents all units), 0, 1, 2... (as GenUnit indices)

	This is an auxiliary check function for calc_heatstreams()
	"""
	_res = pd.DataFrame(
		data=0.0,
		index=['Stack', 'Condenser', 'Misc+aux', 'sensible, air', 'latent, air', 'seawater', 'total waste heat'],
		columns=['Total']+list(range(len(GenFleet))),
		dtype='f8'
	                    )

	for idx, gen in enumerate(GenFleet):
		# Get heat streams and summarize over time
		heatstr_byoutlet = gen.heatstreams_byoutlet.sum(axis=0) * c_['period hr'] * c_['MWh to ktoe']
		heatstr_bykind = gen.heatstreams_bykind.sum(axis=0) * c_['period hr'] * c_['MWh to ktoe']
		total_wasteheat = heatstr_byoutlet.sum()
		assert abs(total_wasteheat-heatstr_bykind.sum()) < 10**-6

		# Write
		_res.at['Stack', idx] = heatstr_byoutlet.at['Stack']
		_res.at['Condenser', idx] = heatstr_byoutlet.at['Condenser']
		_res.at['Misc+aux', idx] = heatstr_byoutlet.at['Misc and aux loads']

		_res.at['sensible, air', idx] = heatstr_bykind.at['sensible, air']
		_res.at['latent, air', idx] = heatstr_bykind.at['latent, air']
		_res.at['seawater', idx] = heatstr_bykind.at['seawater']

		_res.at['total waste heat', idx] = total_wasteheat


	# Summarize over all GenUnits
	for _q in _res.index:
		_res.at[_q, 'Total'] = _res.loc[_q, :].sum()

	return _res


def prep_WRF_inputs(GenFleet, day, WRF_cell_area=300*300, scenario=None, unit='W/m^2', write_to_disk=False,
                    PPcells_only=False, With_height=True):
	"""Prepares the WRF input files, with the ff. specification:

	Domain          As in defined by the WRF grid prepared by Vivek (04.2020)
	Time            24-h with hourly resolution

	Requirements:
		Heat streams by kind have been calculated for all generators
		Option for Height of Exhaust Stack for sensible and latent heat 

	PARAMETERS:
		GenFleet

		day                 The day ('YYYY MMM DD') of the 24-h profile (i.e. a day in the set period in genDispatch).

		WRF_cell_area       (Optional; defaults to 300 m x 300 m = 9*10^4) The area of the WRF grid cell, in m^2.

		scenario            (Optional; required if writing to disk) Name of the scenario. The files would be written in
							the directory "Input_<scenario>" in the PATHS['To WRF'] directory (will create a new directory if
							it doesn't exist).

		write_to_disk       (Optional; defaults to False) Writes the WRF input files (as csvs) to disk.

		PPcells_only        (Optional; defaults to False). If True, then only the WRF grid cells with power plants
							will be in the returned (and possibly written) data structures, as a sparse format.

							Note: Can still have zero rows if power plants were not dispatched. This parameter was
							introduced when studying the variation of a 24-h profile.


	"""
	# --------------------------------------------------------------------------- 1) Initialize input files (pickled)
	# Read the WRF grid info
	WRF_grid = pd.read_pickle(path.join(PATHS['WRF resources'], 'WRF grid.pkl'))

	# Set the columns
	SH_24h_cols = ['Sensible_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
	LH_24h_cols = ['Latent_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
	Sea_24h_cols = ['Sea_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]

	SH_cols = ['Longitude', 'Latitude'] + SH_24h_cols
	LH_cols = ['Longitude', 'Latitude'] + LH_24h_cols
	Sea_cols = ['Longitude', 'Latitude'] + Sea_24h_cols

	# Init the dataframes
	WRF_SH = pd.DataFrame(data=0.0, columns=SH_cols, index=WRF_grid.index, dtype='f8')
	WRF_LH = pd.DataFrame(data=0.0, columns=LH_cols, index=WRF_grid.index, dtype='f8')
	WRF_Sea = pd.DataFrame(data=0.0, columns=Sea_cols, index=WRF_grid.index, dtype='f8')

	# Write the grid info
	for col in ('Longitude', 'Latitude', ):
		WRF_SH[col] = WRF_grid[col]
		WRF_LH[col] = WRF_grid[col]
		WRF_Sea[col] = WRF_grid[col]

	# --------------------------------------------------------------------------- 2) Read the PP->WRF grid mapping file
	PPlocs = pd.read_pickle(path.join(PATHS['WRF resources'], 'Final PP mapping to WRF.pkl'))


	if PPcells_only:
		# if write_to_disk:
		# 	warnings.warn('Writing sparse format WRF input files.')

		PPcells = PPlocs['WRF cell id'].unique()
		WRF_SH = WRF_SH.loc[PPcells, :]
		WRF_LH = WRF_LH.loc[PPcells, :]
		WRF_Sea = WRF_Sea.loc[PPcells, :]

	# --------------------------------------------------------------------------- 3) Resolve scalar
	if unit == 'W/m^2':
		_scalar = 10**6 / WRF_cell_area
	elif unit == 'MW':
		_scalar = 1.0
	else:
		raise ValueError('Unexpected unit: {}'.format(unit))

	# --------------------------------------------------------------------------- 4) Write heat streams
	total_MWh = pd.Series(data=0.0, index=['sensible, air', 'latent, air', 'seawater'])

	for idx, gen in enumerate(GenFleet):
		# a) Get the 24-h profile of heat streams [W/m^2]
		sens_air, lath_air, seawater = toWRF_vector(gen.heatstreams_bykind[day], _scalar)

		# b) Get the WRF cell id
		WRF_cell_id = PPlocs.at[idx, 'WRF cell id']

		# c) Write to the WRF input file
		WRF_SH.loc[WRF_cell_id, SH_24h_cols] += sens_air
		WRF_LH.loc[WRF_cell_id, LH_24h_cols] += lath_air
		WRF_Sea.loc[WRF_cell_id, Sea_24h_cols] += seawater

		# d) Check -- accumulate the total energy (MWh)
		total_MWh += gen.heatstreams_bykind[day].sum() * c_['period hr']

	# Check that the energies match
	if not abs(WRF_SH.loc[:, SH_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['sensible, air']) < 10**-6:
		warnings.warn('Sensible heat did not balance.')

	if not abs(WRF_LH.loc[:, LH_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['latent, air']) < 10**-6:
		warnings.warn('Latent heat did not balance.')

	if not abs(WRF_Sea.loc[:, Sea_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['seawater']) < 10**-6:
		warnings.warn('Seawater heat did not balance.')

	# --------------------------------------------------------------------------- 5) Change col names if MW
	if unit == 'MW':
		WRF_SH.columns = ['Longitude', 'Latitude'] + ['Sensible_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
		WRF_LH.columns = ['Longitude', 'Latitude'] + ['Latent_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
		WRF_Sea.columns = ['Longitude', 'Latitude'] + ['Sea_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]

	# --------------------------------------------------------------------------- 6) Append Height         
	PPlocs['Height [m]']=PPdb['master']['Stack height [m]']
	height = PPlocs[['WRF cell id','Height [m]']]
	height.reset_index(drop=True, inplace=True)
	height = height.drop_duplicates(subset='WRF cell id')
	height = height.rename(columns={'WRF cell id': 'cell ID'})
	height = height.set_index('cell ID')

	if With_height:     
		WRF_SH=pd.concat([WRF_SH,height], axis=1).fillna(0) 
		WRF_LH=pd.concat([WRF_LH,height], axis=1).fillna(0)

	# --------------------------------------------------------------------------- 7) Write to disk
	if write_to_disk:
		#dir_name_SH = path.join(PATHS['To WRF'], 'SH Input - {}'.format(scenario))
		#dir_name_LH = path.join(PATHS['To WRF'], 'LH Input - {}'.format(scenario))
		#dir_name_Sea = path.join(PATHS['To WRF'], 'Sea Input - {}'.format(scenario))
		dir_name_SH = path.join(PATHS['To WRF'], '{}'.format(scenario), 'SH Input' )
		dir_name_LH = path.join(PATHS['To WRF'], '{}'.format(scenario), 'LH Input')
		dir_name_Sea = path.join(PATHS['To WRF'], '{}'.format(scenario), 'Sea Input')      

		if not path.isdir(path.join(PATHS['To WRF'], '{}'.format(scenario))):
			mkdir(path.join(PATHS['To WRF'], '{}'.format(scenario)))
                          
		if not path.isdir(dir_name_SH):
			mkdir(dir_name_SH)
		if not path.isdir(dir_name_LH):
			mkdir(dir_name_LH)
		if not path.isdir(dir_name_Sea):
			mkdir(dir_name_Sea)            

		WRF_SH.to_csv(path.join(dir_name_SH, '{} - {}_sensible_air.csv'.format(scenario, day)), index=True, header=True)
		WRF_LH.to_csv(path.join(dir_name_LH, '{} - {}_latent_air.csv'.format(scenario, day)), index=True, header=True)
		WRF_Sea.to_csv(path.join(dir_name_Sea, '{} - {}_seawater.csv'.format(scenario, day)), index=True, header=True)
		#print('Files written to: {}'.format(dir_name_SH))
		print('Files written to: {}'.format(path.join(PATHS['To WRF'], '{}'.format(scenario))))


	return WRF_SH, WRF_LH, WRF_Sea, total_MWh


def prep_DUCT_inputs(GenFleet, day, WRF_cell_area=300*300, scenario=None, unit='W/m^2', write_to_disk=False,
                    PPcells_only=False, With_height=True):
	"""Prepares the DUCT input files, with the ff. specification:

	Domain          As in defined by the WRF grid prepared by Vivek (04.2020)
	Time            24-h with hourly resolution

	Requirements:
		Heat streams by kind have been calculated for all generators
		Option for Height of Exhaust Stack for sensible and latent heat 

	PARAMETERS:
		GenFleet

		day                 The day ('YYYY MMM DD') of the 24-h profile (i.e. a day in the set period in genDispatch).

		WRF_cell_area       (Optional; defaults to 300 m x 300 m = 9*10^4) The area of the WRF grid cell, in m^2.

		scenario            (Optional; required if writing to disk) Name of the scenario. The files would be written in
							the directory "Input_<scenario>" in the PATHS['To WRF'] directory (will create a new directory if
							it doesn't exist).

		write_to_disk       (Optional; defaults to False) Writes the WRF input files (as csvs) to disk.

		PPcells_only        (Optional; defaults to False). If True, then only the WRF grid cells with power plants
							will be in the returned (and possibly written) data structures, as a sparse format.

							Note: Can still have zero rows if power plants were not dispatched. This parameter was
							introduced when studying the variation of a 24-h profile.


	"""
	# --------------------------------------------------------------------------- 1) Initialize input files (pickled)
	# Read the WRF grid info
	WRF_grid = pd.read_pickle(path.join(PATHS['WRF resources'], 'WRF grid.pkl'))

	# Set the columns
	SH_24h_cols = ['Sensible_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
	LH_24h_cols = ['Latent_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
	Sea_24h_cols = ['Sea_(W/m2)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]

	SH_cols = ['Longitude', 'Latitude'] + SH_24h_cols
	LH_cols = ['Longitude', 'Latitude'] + LH_24h_cols
	Sea_cols = ['Longitude', 'Latitude'] + Sea_24h_cols

	# Init the dataframes
	WRF_SH = pd.DataFrame(data=0.0, columns=SH_cols, index=WRF_grid.index, dtype='f8')
	WRF_LH = pd.DataFrame(data=0.0, columns=LH_cols, index=WRF_grid.index, dtype='f8')
	WRF_Sea = pd.DataFrame(data=0.0, columns=Sea_cols, index=WRF_grid.index, dtype='f8')

	# Write the grid info
	for col in ('Longitude', 'Latitude', ):
		WRF_SH[col] = WRF_grid[col]
		WRF_LH[col] = WRF_grid[col]
		WRF_Sea[col] = WRF_grid[col]

	# --------------------------------------------------------------------------- 2) Read the PP->WRF grid mapping file
	PPlocs = pd.read_pickle(path.join(PATHS['WRF resources'], 'Final PP mapping to WRF.pkl'))


	if PPcells_only:
		# if write_to_disk:
		# 	warnings.warn('Writing sparse format WRF input files.')

		PPcells = PPlocs['WRF cell id'].unique()
		WRF_SH = WRF_SH.loc[PPcells, :]
		WRF_LH = WRF_LH.loc[PPcells, :]
		WRF_Sea = WRF_Sea.loc[PPcells, :]

	# --------------------------------------------------------------------------- 3) Resolve scalar
	if unit == 'W/m^2':
		_scalar = 10**6 / WRF_cell_area
	elif unit == 'MW':
		_scalar = 1.0
	else:
		raise ValueError('Unexpected unit: {}'.format(unit))

	# --------------------------------------------------------------------------- 4) Write heat streams
	total_MWh = pd.Series(data=0.0, index=['sensible, air', 'latent, air', 'seawater'])

	for idx, gen in enumerate(GenFleet):
		# a) Get the 24-h profile of heat streams [W/m^2]
		sens_air, lath_air, seawater = toWRF_vector(gen.heatstreams_bykind.loc[day], _scalar)

		# b) Get the WRF cell id
		WRF_cell_id = PPlocs.at[idx, 'WRF cell id']

		# c) Write to the WRF input file
		WRF_SH.loc[WRF_cell_id, SH_24h_cols] += sens_air
		WRF_LH.loc[WRF_cell_id, LH_24h_cols] += lath_air
		WRF_Sea.loc[WRF_cell_id, Sea_24h_cols] += seawater

		# d) Check -- accumulate the total energy (MWh)
		total_MWh += gen.heatstreams_bykind.loc[day].sum() * c_['period hr']

	# Check that the energies match
	if not abs(WRF_SH.loc[:, SH_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['sensible, air']) < 10**-6:
		warnings.warn('Sensible heat did not balance.')

	if not abs(WRF_LH.loc[:, LH_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['latent, air']) < 10**-6:
		warnings.warn('Latent heat did not balance.')

	if not abs(WRF_Sea.loc[:, Sea_24h_cols].sum().sum() / _scalar
	           - total_MWh.at['seawater']) < 10**-6:
		warnings.warn('Seawater heat did not balance.')

	# --------------------------------------------------------------------------- 5) Change col names if MW
	if unit == 'MW':
		WRF_SH.columns = ['Longitude', 'Latitude'] + ['Sensible_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
		WRF_LH.columns = ['Longitude', 'Latitude'] + ['Latent_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]
		WRF_Sea.columns = ['Longitude', 'Latitude'] + ['Sea_(MW)_H{}'.format(str(hr).zfill(2)) for hr in range(24)]

	# --------------------------------------------------------------------------- 6) Append Height         
	PPlocs['Height [m]']=PPdb['master']['Stack height [m]']
	height = PPlocs[['WRF cell id','Height [m]']]
	height.reset_index(drop=True, inplace=True)
	height = height.drop_duplicates(subset='WRF cell id')
	height = height.rename(columns={'WRF cell id': 'cell ID'})
	height = height.set_index('cell ID')

	if With_height:     
		WRF_SH=pd.concat([WRF_SH,height], axis=1).fillna(0) 
		WRF_LH=pd.concat([WRF_LH,height], axis=1).fillna(0)
        
	PPlocs['PP Longitude']=PPdb['master']['long (°)']
	PPlocs['PP Latitude']=PPdb['master']['lat (°)']
	PP_long_lat = PPlocs[['WRF cell id','PP Longitude','PP Latitude']]
	PP_long_lat.reset_index(drop=True, inplace=True)
	PP_long_lat = PP_long_lat.drop_duplicates(subset='WRF cell id')
	PP_long_lat = PP_long_lat.rename(columns={'WRF cell id': 'cell ID'})
	PP_long_lat = PP_long_lat.set_index('cell ID')
	WRF_SH=pd.concat([PP_long_lat,WRF_SH], axis=1)
	WRF_SH=WRF_SH.drop(['Longitude', 'Latitude'], axis = 1)
	#WRF_SH=WRF_SH.reset_index(drop=True)

	WRF_LH=pd.concat([PP_long_lat,WRF_LH], axis=1)
	WRF_LH=WRF_LH.drop(['Longitude', 'Latitude'], axis = 1)
	#WRF_LH=WRF_LH.reset_index(drop=True)
	#WRF_LH=WRF_LH.style.hide_index()
    
	WRF_Sea=pd.concat([PP_long_lat,WRF_Sea], axis=1)
	WRF_Sea=WRF_Sea.drop(['Longitude', 'Latitude'], axis = 1)
    
	# --------------------------------------------------------------------------- 7) Write to disk
	if write_to_disk:
		#dir_name_SH = path.join(PATHS['To WRF'], 'SH Input - {}'.format(scenario))
		#dir_name_LH = path.join(PATHS['To WRF'], 'LH Input - {}'.format(scenario))
		#dir_name_Sea = path.join(PATHS['To WRF'], 'Sea Input - {}'.format(scenario))
		dir_name_SH = path.join(PATHS['To WRF'], '{}'.format(scenario), 'SH Input' )
		dir_name_LH = path.join(PATHS['To WRF'], '{}'.format(scenario), 'LH Input')
		dir_name_Sea = path.join(PATHS['To WRF'], '{}'.format(scenario), 'Sea Input')      

		if not path.isdir(path.join(PATHS['To WRF'], '{}'.format(scenario))):
			mkdir(path.join(PATHS['To WRF'], '{}'.format(scenario)))
                          
		if not path.isdir(dir_name_SH):
			mkdir(dir_name_SH)
		if not path.isdir(dir_name_LH):
			mkdir(dir_name_LH)
		if not path.isdir(dir_name_Sea):
			mkdir(dir_name_Sea)            

		WRF_SH.to_csv(path.join(dir_name_SH, '{} - {}_sensible_air.csv'.format(scenario, day)), index=False, header=True)
		WRF_LH.to_csv(path.join(dir_name_LH, '{} - {}_latent_air.csv'.format(scenario, day)), index=False, header=True)
		WRF_Sea.to_csv(path.join(dir_name_Sea, '{} - {}_seawater.csv'.format(scenario, day)), index=False, header=True)
		#print('Files written to: {}'.format(dir_name_SH))
		print('Files written to: {}'.format(path.join(PATHS['To WRF'], '{}'.format(scenario))))


	return WRF_SH, WRF_LH, WRF_Sea, total_MWh




def calc_TotalMWh_fromWRFinputs(WRF_SH, WRF_LH, WRF_Sea, WRF_cell_area=300*300):
	"""Function to check the WRF input files"""
	return pd.Series({
		'sensible, air': WRF_SH.loc[:, WRF_SH.columns[2:]].sum().sum()*WRF_cell_area*10**-6,
		'latent, air': WRF_LH.loc[:, WRF_LH.columns[2:]].sum().sum()*WRF_cell_area*10**-6,
		'seawater': WRF_Sea.loc[:, WRF_Sea.columns[2:]].sum().sum()*WRF_cell_area*10**-6,
	})


def toWRF_vector(heatstreams_24h, scalar):
	"""Converts a 24-h, 30-min res, MW heatstream data into three hourly & W/m^2 arrays as:
		1) sensible, air
		2) latent, air
		3) seawater

	The down-sampling is done by taking the mean, which has the property of preserving the energy.

	"""

	sens_air = np.zeros(24, dtype='f8')
	lath_air = np.zeros(24, dtype='f8')
	seawater = np.zeros(24, dtype='f8')

	for hr in range(24):
		# MW
		heat_hr = heatstreams_24h.iloc[2*hr:2*(hr+1)].mean(axis=0)

		sens_air[hr] = heat_hr.at['sensible, air']
		lath_air[hr] = heat_hr.at['latent, air']
		seawater[hr] = heat_hr.at['seawater']

	# to W/m^2 (unless scalar=1.0, if the original MW units was requested)
	sens_air = sens_air * scalar
	lath_air = lath_air * scalar
	seawater = seawater * scalar

	return sens_air, lath_air, seawater


def analyze_variation(GenFleet):
	SH_all = None
	LH_all = None
	Sea_all = None

	for day in range(1, 31):
		date = '2016 Apr {}'.format(str(day).zfill(2))

		# ......................................................................... a) Calc WRF heat maps
		# [MWh]
		WRF_SH, WRF_LH, WRF_Sea, total_MWh = prep_WRF_inputs(
			GenFleet=GenFleet, day=date, write_to_disk=False, PPcells_only=True, unit='MW')

		# ......................................................................... b) Wide to long
		# 1) Set data cols (24-h) to 0, 1, .. 23
		list_24h = list(range(24))
		WRF_SH.columns = ['Longitude', 'Latitude', ] + list_24h
		WRF_LH.columns = ['Longitude', 'Latitude', ] + list_24h
		WRF_Sea.columns = ['Longitude', 'Latitude', ] + list_24h

		# 2) Use index as id col
		WRF_SH['cell ID'] = WRF_SH.index
		WRF_LH['cell ID'] = WRF_LH.index
		WRF_Sea['cell ID'] = WRF_Sea.index

		# 3) melt
		date2 = 'Apr {}'.format(str(day).zfill(2))
		_args = {
			'id_vars': ['cell ID'],
			'var_name': 'hour',
			'value_vars': list_24h,
			'value_name': date2,
		}
		SH_long = pd.melt(WRF_SH, **_args)
		LH_long = pd.melt(WRF_LH, **_args)
		Sea_long = pd.melt(WRF_Sea, **_args)

		# 4) set 'cell ID' and 'hour' as multi-index
		SH_long.set_index(keys=['cell ID', 'hour'], inplace=True)
		LH_long.set_index(keys=['cell ID', 'hour'], inplace=True)
		Sea_long.set_index(keys=['cell ID', 'hour'], inplace=True)

		# ......................................................................... c) Append to DF
		if SH_all is None:
			SH_all = pd.DataFrame(index=SH_long.index)
			LH_all = pd.DataFrame(index=LH_long.index)
			Sea_all = pd.DataFrame(index=Sea_long.index)

		SH_all[date2] = SH_long
		LH_all[date2] = LH_long
		Sea_all[date2] = Sea_long

	# Finally, transpose so that axis 0: days
	SH_all = SH_all.transpose()
	LH_all = LH_all.transpose()
	Sea_all = Sea_all.transpose()

	return SH_all, LH_all, Sea_all



