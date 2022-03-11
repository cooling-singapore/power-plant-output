"""Top-level workspace module of the Singapore Power Plant Dispatch project.

Program Startup:
	- All startup sequence and overhead is implemented in gd_core.
	- gd_core also defines all of the global variables that can be borrowed by other modules (namely this one,
	and gd_analysis)
	- Importing gd_core calls gd_core.__init_sequence(), and reads all the inputs (config, PPdb,
	and fundamentals)
	- gd_core is imported by this module


Author:         Pang Teng Seng, D. Kayanan
Create date:    Oct. 14, 2019
Revised date:   Feb. 22, 2022
Version:        2.0
Release date:   Aug. 03, 2020
"""
# IMPORTS
# ----------------------------------------------------- Project modules
import gd_core as __core
import PowerPlants as pp
get_plantclass = pp.get_plantclass
import NEMS as mkt

# Import globals from gd_core into the work namespace
# Power plants
PPdb = __core.PPdb
GenFleet = pp.GenUnit.get_fleet()  # GenFleet is just introduced here, and is not used in any of the functions
GenCos   = __core.GenCos

# Input data
dFundamentals   = __core.dFundamentals
dGenParameters  = __core.dGenParameters
dParameters     = __core.dParameters

# The rest
dStats = __core.dStats
Scenarios   = __core.Scenarios
PPdispatchError = __core.PPdispatchError
# todo - will probably remove the PPdispatchError construct
if PPdispatchError is None:
	from gd_analysis import *

# -----------------------------------------------------  Other modules
# Python
import os
import logging
import datetime as dttm
import warnings


# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2, suppress=True)

# DK
from DK_Collections import fmtdict
import DataHandler as dh


def set_period(t_from=None, t_to=None):
	"""Sets the simulation period to [t_from, t_to] (discrete days as datetime.date, or as date strings in 'YYYY
	MMM DD'. The market will be solved from t_from at 00:00 until the end of t_to.

	Passing arguments that are not datetime.date, date strings in the expected format, or are outside t_common would
	raise exceptions.

	Both parameters are optional; leaving out t_from will set it to the beginning of the common period, and similarly
	for t_to.


	---------------------------------------- DEV NOTES -- from _subinit4 ------------------------------------------
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
		'periods per D':    N_periods,
		'periods duration': P_duration,
		'DxP_freq':         DxP_freq,
	})
	logging.info("market period duration: {}".format(time['periods duration']))


	# -------------------------------------------------------------------- 2.3) User-set variables
	time.update({
		'simulation period':    [None, None],
		'D_index':              None,
		'DxP_index':            None,
	})

	"""
	# ----------------------------------------------------------------------------------- 1) Param checks
	# 1.1 Default, set to common period ends
	if t_from is None:
		t_from = __core.time['common period'][0]
	if t_to is None:
		t_to = __core.time['common period'][1]

	# 1.2 If string, try to parse as datetime --> date
	if isinstance(t_from, str):
		try:
			t_from = dttm.datetime.strptime(t_from, "%Y %b %d").date()
		except:
			print("Failed to convert passed string argument into a date.", t_from)
			raise

	if isinstance(t_to, str):
		try:
			t_to = dttm.datetime.strptime(t_to, "%Y %b %d").date()
		except:
			print("Failed to convert passed string argument into a date.", t_to)
			raise

	# 1.3 By now, should be date
	if not isinstance(t_from, dttm.date):
		raise TypeError("Pls. pass a datetime.date.", t_from)

	if not isinstance(t_to, dttm.date):
		raise TypeError("Pls. pass a datetime.date.", t_to)

	# 1.4 See if within common period
	if not(__core.time['common period'][0] <= t_from <= __core.time['common period'][1]):
		t_common_str = [dt.strftime("%Y %b %d") for dt in __core.time['common period']]
		raise ValueError("Pls. pass a datetime.date within the common period [{}, {}].".format(*t_common_str), t_from)

	if not(__core.time['common period'][0] <= t_to <= __core.time['common period'][1]):
		t_common_str = [dt.strftime("%Y %b %d") for dt in __core.time['common period']]
		raise ValueError("Pls. pass a datetime.date within the common period [{}, {}].".format(*t_common_str), t_to)

	# ----------------------------------------------------------------------------------- 2) Set __core.time parameters
	__core.time.update({
		'simulation period':    [t_from, t_to],
		'D_index':              pd.period_range(t_from, t_to, freq='D'),
		'DxP_index':            pd.period_range(t_from,
		                                        pd.Timestamp(t_to) + pd.Timedelta('1D') - __core.time['periods duration'],
		                                        freq=__core.time['DxP_freq']),
	})


	# ----------------------------------------------------------------------------------- 3) Done - log and report
	t_simul_str = [dt.strftime("%Y %b %d") for dt in __core.time['simulation period']]
	N_days = len(__core.time['D_index'])
	logging.info("SIMULATION TIME: set to [{},  {}] ({} days).\n".format(*t_simul_str, N_days))
	print("Simulation time set to [{},  {}] ({} days).\n".format(*t_simul_str, N_days))
	return


def set_scenario(fundamentalKeys_toscenarios):
	"""Sets the scenario as {dFundamental key: 'scenario name'}. Pass a dictionary of {fundamental key: scenario
	name} (for fuel prices, the sub-dict is accessed). Can pass None as the scenario name, to set it to the
	default/base scenario (i.e. first column). The passed scenario would be checked if it exists."""
	for fundamental_key, scenario in fundamentalKeys_toscenarios.items():
		if fundamental_key in ('sys demand', 'forex'):
			_data = dFundamentals[fundamental_key].val
		else:
			_data = dFundamentals['fuel prices'][fundamental_key].val

		if isinstance(_data, pd.DataFrame):
			if scenario in _data:
				Scenarios[fundamental_key] = scenario
				print("Fundamental '{}' set to scenario '{}'".format(fundamental_key, scenario))

			elif scenario is None:
				if fundamental_key in Scenarios:
					del Scenarios[fundamental_key]
				print("Fundamental '{}' set to base scenario".format(fundamental_key))

			else:
				raise ValueError("The given scenario does not exist. Existing scenarios: {}".format(
					", ".join(_data.columns)))
		else:
			raise RuntimeError("There are no scenarios in the target fundamental data.")
	return


def get_forex_D(day):
	"""Wrapper of the market module's get_forex_D"""
	return mkt.get_forex_D(day,
	                       forex=dFundamentals['forex'],
	                       Scenarios=__core.Scenarios,
	                       mkt_currency=__core.config['market settings']['currency'])


def solve(simulation_name=None, runchecks=False, upon_exception='continue', get_tracking=False, daily_progress=False):
	"""Runs the market simulation. The API defines this as a three-step process:
		1) Pre-simulation*
		2) Simulation (i.e. market clearing)
		3) Post-simulation*

	*If these are unnecessary, then implement a function that does nothing.

	"""
	if __core.time['simulation period'] == [None, None]:
		raise RuntimeError('Simulation period has not been set. Pls. call set_period()')
	# ........................................................................... logging
	t_simul_str = (dt.strftime("%Y %b %d") for dt in __core.time['simulation period'])
	if len(Scenarios) == 0:
		Scenarios_str = "Scenarios: none"
	else:
		Scenarios_str = "Scenarios:\n{}".format(fmtdict(Scenarios))
	report_str = "[{}] Market simulation launched for [{} to {}]\n\n{}\n".format(
		__core.log_time(), *t_simul_str, Scenarios_str)
	print(report_str)
	logging.info(report_str)

	# ........................................................................... launch mkt routines
	Aux, Results = mkt.pre_simulation(simulation_name)

	# .simulate() and .post_simulation() modify Results in-place
	tracking = mkt.simulate(Aux, Results, runchecks=runchecks, upon_exception=upon_exception,
	                        daily_progress=daily_progress)
	mkt.post_simulation(Results)

	if get_tracking:
		return Results, tracking
	else:
		return Results


def get_rand_D():
	D0, DN = __core.time['common period']
	D = D0 + dttm.timedelta(days=np.random.randint((DN - D0).days + 1))

	assert D0 <= D <= DN
	return D




