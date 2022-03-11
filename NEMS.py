"""This module implements the market model of the National Electricity Market of Singapore. Market specific info are
implemented here, so that PowerPlants.py remains market agnostic.

Extensions to other market models, whilst using the same GenUnit class in PowerPlants.py, would need an API to define
the interaction between market agents and the market itself. Resources in the underlying market model would need to
be wrapped in order to conform to his API, and promote true modularity.

Author:         Pang Teng Seng, D. Kayanan
Create date:    Nov. 7, 2019
Revised date:   Feb. 22, 2022
Version:        2.0
Release date:   TBA

COMPONENTS:
	Pre-simulation prep
	Bid aggregation
	Clearing

line 360 - Added GSF
"""
# Project
from PowerPlants import GenUnit
from gd_core import time, config, Scenarios, dFundamentals

# Python
from itertools import chain
import logging
import time as timemod

# Python 3rd party
import numpy as np
import pandas as pd

# DK library
#from DK_Collections import isnotebook, clear_console
from DK_Collections import *


def get_forex_D(day, forex, Scenarios, mkt_currency='USD'):
	"""Returns the foreign exchange rate (local currency/USD) of the given day. Pass an argument suitable to
	Timestamp indexing in Pandas.

	PARAMETERS:
		day             The particular day to get the forex rate, as Pandas Timestamp coercible or Pandas Period
		forex           The forex data as dh.Metadata
		Scenarios       The Scenarios variable in PPdispatch
		mkt_currency    The market currency as a string code (e.g. 'USD', 'SGD')

	"""
	if mkt_currency == 'USD':
		return 1
	else:
		# TODO -- eliminate repeated func calls and isinstance()
		# First, check scenarios
		_forexval = forex.val

		if isinstance(_forexval, pd.DataFrame):
			if 'forex' in Scenarios:
				targetCol = Scenarios['forex']
			else:
				targetCol = _forexval.columns[0]
		else:
			targetCol = None
		return forex.tsloc(day, col=targetCol)


def pre_simulation(simulation_name):
	"""To be called right before simulating the market (looping over days and throughtout the day). Does the ff.
	preparatory steps:

		1) Initialize gen.Schedule as numpy 4D (5D for cogen) with the ff. cols:
				0   availability              as ... 1, 1, 1, ... 0, 0, 0, ... (1 is online)
				1   Po                        MW dispatch
				2   ne, achieved              [0, 1]
				3   ne, potential
				4   Qo, if cogen

		2) Sample the FL efficiency parameter; binds to GenUnits

		3) Simulating the planned maintenance (outage) schedule. Results are written to self.Schedule (col 0)

	Enhancements:
		1) Subset the Demand Series and return it as a numpy (to be indexed via DxP_idx (positionally) by simulate())

	PARAMETERS:
		time, GenFleet, dFundamentals, Scenarios -- all from the namespace of PPdispatch.py


	RETURN:
		Aux         Dictionary of auxiliary objects, containing:

		vDemand     ndarray of the relevant slice of the demand (in dFundamentals['sys demand']). This is then
						indexed by DxP_idx in simulate() (this is a performance enhancement).

		Results     Dictionary of results objects. See docstring of simulate() for contents.

	OUTAGES MODEL:
		The power plants are modelled as simple repairable systems with two states: UP / online / 1 and DOWN /
		offline / 0.
	"""
	logging.info("Start NEMS pre-simulation")
	# ------------------------------------------------------------------------- 0) Prep and checks
	if time['simulation period'] == [None, None]:
		raise RuntimeError('No simulation period set. Pls. set the simulation period first  with set_period().')

	n_DxP = int(time['DxP_index'].shape[0])

	# gen.Schedule column identifiers
	_Available, _Po, _ne, _ne_pot, _Qo = 0, 1, 2, 3, 4
	GenFleet = GenUnit.get_fleet()

	# ------------------------------------------------------------------------- 1) Init gen.Schedule
	#                                                                           All cols are np.nan, except Po
	for gen in GenFleet:
		if gen.is_['Cogen']:
			gen.Schedule = np.empty((n_DxP, 5))
			gen.Schedule[:, _Qo] = 0.0
		else:
			gen.Schedule = np.empty((n_DxP, 4))
		gen.Schedule[:, (_Available, _ne, _ne_pot)] = np.nan, np.nan, np.nan
		gen.Schedule[:, _Po] = 0.0

	# ------------------------------------------------------------------------- 2) Calculate stochastic components
	logging.info("Calculating stochastic components:")
	# ----------------------------------------------------------------------- 2.1) Potential FL electrical efficiency
	for gen in GenFleet:
		gen.rsamples['n_e_FL'] = gen.rv['n_e_FL'].rvs(n_DxP, random_state=gen.rseeds['n_e_FL'])
		gen.Schedule[:, _ne_pot] = gen.rsamples['n_e_FL']
	logging.info(" - Full load electrical efficiency potential sampled.")

	# ----------------------------------------------------------------------- 2.2) Solve generator outages
	#                                                          Results written to gen.Schedule, col _Available (col 0)
	# Solve the availability schedule of each gen
	for gen in GenFleet:
		# .......................................................... a) init RS_seq, start_ol
		RS_seq = []
		np.random.seed(gen.rseeds['start ol'])
		start_ol = np.random.rand() <= gen.reliability['Limiting Availability']

		# .......................................................... b) Sample T and D loop to extend RS_seq
		for itr in range(20):
			# i) Sample T and D
			T_rvs = np.round(gen.rv['T'].rvs(100, random_state=gen.rseeds['T']))   # online durations
			D_rvs = np.round(gen.rv['D'].rvs(100, random_state=gen.rseeds['D']))   # offline durations

			# ii) Set the sequence of T and D of the RS sequence
			if start_ol:
				zipped = zip(T_rvs, D_rvs)
			else:
				zipped = zip(D_rvs, T_rvs)

			# iii) Extend RS the sequence
			new_seq = np.fromiter(chain.from_iterable(zipped), dtype='int32')

			# iv) Find the idx of new_seq to extend RS_seq with
			# Do this by calculating the total duration (as cummulative sum of new_seq)
			new_cumsum = np.cumsum(new_seq)
			lacking_duration = n_DxP - sum(RS_seq)

			# Then, find until which index of new_seq to add to RS_seq
			for idx, total_dur in enumerate(new_cumsum):
				if total_dur > lacking_duration:
					break

			RS_seq.extend(new_seq[:idx + 1])

			# v) If RS_seq already covers the simulation duration, exit for itr loop.
			if sum(RS_seq) >= n_DxP:
				break
		else:
			raise RuntimeError('Availability was not solved after the max iterations')

		# .......................................................... c) Convert RS_seq := [..Ti, Di..] to
		#                                                               self.Schedule['Available'] :=
		#                                                                 numpy array (..1,1,1..0,0,0..)
		# i) init
		is_online = np.array([])
		now_online = start_ol

		# ii) Iterate over durations in RS_seq to convert into 1's or 0's
		for dur in RS_seq:
			if now_online:
				new_arr = np.ones(dur)
			else:
				new_arr = np.zeros(dur)

			is_online = np.append(is_online, new_arr)
			now_online = not now_online

		# Check if lengths match. This is in general longer than n_DxP
		assert len(is_online) == sum(RS_seq)

		# iii) Now, trim is_online to n_DxP and bind to GenUnit
		gen.Schedule[:, _Available] = is_online[:n_DxP]

	# End for gen in fleet loop of solve outages
	logging.info(" - Generator outages solved.")

	# ------------------------------------------------------------------------- 3) Subset the Demand (enhancement)
	demand_md = dFundamentals['sys demand']
	if isinstance(demand_md.val, pd.DataFrame):
		if 'sys demand' in Scenarios:
			targetCol = Scenarios['sys demand']
		else:
			targetCol = demand_md.val.columns[0]

		vDemand = demand_md.val.loc[(per.to_timestamp() for per in time['DxP_index']), targetCol].values

	else:
		vDemand = demand_md.val.loc[(per.to_timestamp() for per in time['DxP_index'])].values

	# ------------------------------------------------------------------------- 4) Init Results
	GenUnit.dispatch = np.zeros((n_DxP, GenFleet.shape[0]))

	Results = {
		'name': simulation_name,
		'dispatch [MW]': GenUnit.dispatch,
		'Total demand [MWh]': np.sum(vDemand) * time['periods duration'].seconds/3600
	}

	Aux = {
		'vDemand':  vDemand
	}

	return Aux, Results


def simulate(Aux, Results, runchecks=False, upon_exception='continue', daily_progress=False):
	"""Simulates the market.

	Parameters are as in PPdispatch counterparts, except for:
		Aux, Results        As returned by pre_simulation()

		mkt_currency        __config['market settings']['currency']

		upon_exception      (Optional; defaults to 'continue') Defines the behavior of the program upon encountering
							pre-defined exceptions during simulation.

		runchecks           (Optional; defaults to False) If True, then additional assertions are executed.

		daily_progress      (Optional; defaults to False) If True, then the simulation progress is printed daily. If
							False, then the progress is reported at the 25%, 50%, and 75% mark (w/ estimated
							remaining time).


	RETURN:
		Results = {
			Name                    Name assigned to simulation
			Prices                    Clearing price of the market, as Series {DxP_index : price}
			dispatch [MW]           Dataframe of Po of each generator (ordered in GenFleet) indexed by DxP_index
			demand [MW]             Demand input for the simulation
			Total demand [MWh]      Total demand served for the period
		}

		tracking        Tracks the simulation and records the status of each clearing period

	Note:
		DxP_idx         Is the simulation counter (range index starting from 0)
		DxP_index       Is the Pandas Period Index, for the entire simulation duration. This is stored in __time
	"""
	# ------------------------------------------------------------------------ Init
	# i) counters and time
	ts = timemod.time()
	N_simul = time['DxP_index'].shape[0]
	N_days = time['D_index'].shape[0]

	# ii) Local bindings
	vDemand = Aux['vDemand']
	forex = dFundamentals['forex']
	WtE_sched = dFundamentals['WtE Sched']
	mkt_currency = config['market settings']['currency']

	GenFleet = GenUnit.get_fleet()

	# iii) Tracking and Price arrays
	# Tracking col indices [Start Time | Delta time | Status]
	_StartT, _deltaT, _Stat = 0, 1, 2
	tracking = np.empty((N_simul, 3))

	Price = np.empty((N_simul,))
	tracking[:], Price[:] = np.nan, np.nan

	# iv) Quarterly tracking
	progress_marks = 10
	if not daily_progress:
		print("Mark Reached \t\t Est. Time Left")

	# ------------------------------------------------------------------------------ I. Loop through days --------------
	for D_idx, D_day in enumerate(time['D_index']):
		# todo -- possible enhancement
		# This part of the code wouldn't change with changes in params. Consider pre-determining fuel prices,
		# and store as an array (GenUnit attr)
		# ................................................................. 0) Update progress bar
		progress_perc = D_idx / N_days * 100

		if daily_progress and D_idx > 0:
			# Clear
			# if isJupyter :
				# clear_output(True)
			# else:
				# clear_console()
			# Print
			print("Simulating {}\n".format(D_day))
			print("Day: {} / {}".format(D_idx, N_days, progress_perc))
			print("{progbar:<100}\t({perc:0.1f}%)".format(
				progbar="." * round(progress_perc),
				perc=progress_perc
			))

		elif not daily_progress and progress_perc > progress_marks:
			t_elapased = timemod.time() - ts
			t_totalest = (100/progress_marks)*t_elapased
			t_leftest = t_totalest - t_elapased
			print("{}% \t\t\t {:0.1f}s".format(progress_marks, t_leftest))

			# Update to next mark
			if progress_marks == 10:
				progress_marks = 50
			elif progress_marks == 50:
				progress_marks = 75
			else:
				progress_marks = 100

		# ................................................................. 1) get forex
		forex_D = get_forex_D(D_day, forex=forex,
		                      Scenarios=Scenarios,
		                      mkt_currency=mkt_currency)

		# ................................................................. 2) get the D fuel price for all gens
		for gen in GenFleet:
			if not gen.is_['WtE']:
				gen.get_fuelprice_D(D_day, forex_D=forex_D, Scenarios=Scenarios)

		# ------------------------------------------------------------------------- II. Loop periods of the day --------
		for P_idx in range(time['periods per D']):
			# ............................................................. 1) Calculate indices, update simul counter,
			#                                                                  track time
			DxP_idx = D_idx * time['periods per D'] + P_idx
			tracking[DxP_idx, _StartT] = timemod.time()-ts

			# Update simul counter. Recall -- this alters gen.effcurves()
			GenUnit._GenUnit__simul = DxP_idx

			# ............................................................. 2) Calculate SUPPLY
			# ...................................................... a) Get ol fleet
			# And reset the dynamic attrs of offline units
			# todo olFleet = [x for x in GenFleet if x>0 else d=0]
			olFleet = []
			for gen in GenFleet:
				if gen.Schedule[DxP_idx, 0]:
					olFleet.append(gen)
				else:
					gen.Po_P, gen.TotalCosts_P, gen.SRMarginalCosts_P, gen.bids_P = None, None, None, None

			# ...................................................... b) Calculate bids of ol gens
			# Todo Adjust the bidding behavior here
			# Note x = [(gen.x(), gen.y(), gen.y()) for gen in olFleet] form did not improve speed

			for gen in olFleet:
				if gen.is_['WtE']:
					gen.calc_bids(main_mode='WtE', WtE_sched=WtE_sched, D_day=D_day)
				elif gen.is_['GSF']:
					gen.calc_bids(main_mode='GSF')                   
				else:
					# i) Set bid quantities
					#gen.set_bidquantities(mode=2, Nbids=10, runchecks=runchecks)
					gen.set_bidquantities(mode=2, Nbids=10, runchecks=runchecks)

					# ii) Calc TC (bypassed) and SRMC
					gen.calc_SRMC(runchecks=runchecks)

					# iii) Calc bids
					gen.calc_bids(sub_mode='shift all')

			# ...................................................... c) Aggregate the bids --> Supply
			_SupplyCurve = _formSupplyCurve(olFleet, runchecks=runchecks)
			#print("Demand: ",vDemand[DxP_idx], " Supply: ", _SupplyCurve[len(_SupplyCurve)-1,4]) 
			# ............................................................. 3) Clear market (S+D intersection)
			# Writes acceptance values to _SupplyCurve in-place
			try:
				Price[DxP_idx] = _SDintersection(_SupplyCurve, Demand_P=vDemand[DxP_idx], runchecks=runchecks)
			except InadequateSupply:
				if upon_exception == 'continue':
					tracking[DxP_idx, _Stat] = 'InadequateSupply'
					continue
				else:
					raise

			# ............................................................. 4) Report clearing results
			# Transfers the acc in the _SupplyCurve to gen.bids_P
			_writeAcceptance(olFleet, _SupplyCurve, Price[DxP_idx], runchecks=runchecks)
			# Writes to gen.Schedule and Dispatch
			_calcSchedule(DxP_idx, olFleet, Results)

			# ............................................................. Exit inner loop
			tracking[DxP_idx, _Stat] = True



	# ------------------------------------------------------------------------------------------ EXIT
	# ...................................................................... a) Numpy to Pandas + compile Results
	# Tracking to Pandas
	tracking[1:, _deltaT] = tracking[1:, _StartT]-tracking[:-1, _StartT]
	tracking = pd.DataFrame({
		'Start Time': tracking[:, _StartT],
		'Delta Time': tracking[:, _deltaT],
		'Status':     tracking[:, _Stat],
	}, index=time['DxP_index'])

	# Dispatch to Pandas
	_cols = pd.Index("Gen {id}: {name}".format(id=gen.id['rowID'], name=gen.id['Unit Name']) for gen in GenFleet)
	_df = pd.DataFrame(
		columns=_cols,
		data = {col: GenFleet.at[idx].Schedule[:, 1] for idx, col in enumerate(_cols)},
		index=time['DxP_index'],
		dtype='f8',
	)
	GenUnit.dispatch = _df
	Results['dispatch [MW]'] = _df

	# Update Results
	Results.update({
		'Prices': pd.Series(data=Price, index=time['DxP_index'], dtype='f8', name=mkt_currency),
		'demand [MW]': pd.Series(data=vDemand, index=time['DxP_index']),
	})

	# ...................................................................... b) Misc
	# Turn OFF simulation mode in GenUnit
	GenUnit._GenUnit__simul = -1

	# if daily_progress:
		# if isJupyter:
			# clear_output(True)
		# else:
			# clear_console()

	print("Simulation complete. Elapsed time: {:0.1f}s".format(timemod.time()-ts))

	# Todo log (examine tracking)
	return tracking


def _formSupplyCurve(olFleet, runchecks=False):
	"""Builds the supply curve from the bids of the online generators.

	DEV NOTES:
		How to sort a numpy array according to the nth col:
		arr = arr[arr[:,n].argsort()]

		Columns:
			0       Bid id
			1       qty
			2       price
			3       acceptance
			4       Supply (accumulation of qty)
	"""
	# 1) Init - Concatenate bids_P of ol gens
	#for gen in olFleet:
		#print(gen.bids_P)
	_SupplyCurve = np.concatenate(tuple(gen.bids_P for gen in olFleet), axis=0)

	# 2) Add the accumulated supply column [bid id | qty | price | acc | Supply]
	Supply_MW = np.zeros((_SupplyCurve.shape[0], 1))
	_SupplyCurve = np.concatenate((_SupplyCurve, Supply_MW), axis=1)

	# 3) Sort bids in inc price
	_SupplyCurve = _SupplyCurve[_SupplyCurve[:, 2].argsort()]
	# Asserts that price is inc (allows no change)
	assert not runchecks or all((_SupplyCurve[1:, 2] - _SupplyCurve[:-1, 2]) >= 0)

	# 4) Calc the accumulated Supply
	_SupplyCurve[:, 4] = np.cumsum(_SupplyCurve[:, 1], axis=0)

	return _SupplyCurve


def _SDintersection(_SupplyCurve, Demand_P, runchecks=False):
	"""Locates the intersection of the Supply and Demand curves, and writes the bid acceptance to _SupplyCurve (
	in-place). Returns the clearing price.

	LIMITATION:     The marginal bid (i.e. the last bid accepted, whose 0 < acceptance <= 1) may be [0, Pmin] bid of
	a GenUnit. This is physically not allowed, but the current implementation does not prevent this. This can be
	solve by altering the bidding behavior, s.t. the [0, Pmin] bids are artifically lower as to significantly dec.
	the likelihood of this (it does seem that this is how it's done).
	"""
	# _SupplyCurve column identifiers
	_qty, _Pr, _acc, _Supply = 1, 2, 3, 4
	# ............................................................... 0) Locate fully-accepted bids and marginal bid
	Lf_underdemand = _SupplyCurve[:, _Supply] < Demand_P
	#print(Lf_underdemand)    
	mbid_idx = np.sum(Lf_underdemand)
   
	# ............................................................... 1) Bids < Demand --> Fully accepted
	_SupplyCurve[Lf_underdemand, _acc] = 1.0

	# ............................................................... 2) Partially accept the marginal bid
	#print("mbid_idx:",  mbid_idx, " _SupplyCurve (len):", len(_SupplyCurve)," Demand_P:" ,Demand_P, " Pre_Supply: ",_SupplyCurve[mbid_idx, _Supply], " Pre_bid",_SupplyCurve[mbid_idx, _qty]  )
	if mbid_idx == len(_SupplyCurve): # added
		#print("mbid_idx:",  mbid_idx, " Demand_P:" ,Demand_P, " Pre_Supply: ",_SupplyCurve[mbid_idx-1, _Supply], " Pre_bid", _SupplyCurve[mbid_idx-1, _qty])        
		mbid_idx=mbid_idx-1           #added

	mbid_qty = Demand_P - (_SupplyCurve[mbid_idx, _Supply] - _SupplyCurve[mbid_idx, _qty])
	_SupplyCurve[mbid_idx, _acc] = mbid_qty / _SupplyCurve[mbid_idx, _qty]

	# Assert power balance via qty * acc
	assert not runchecks or abs(np.sum(_SupplyCurve[:, _qty] * _SupplyCurve[:, _acc]) - Demand_P) < 10 ** -6

	# ............................................................... 3) Get the clearing price
	#if mbid_idx == len(_SupplyCurve): #added
		#mbid_idx=mbid_idx-1           #added
	Price_P = _SupplyCurve[mbid_idx, _Pr]
	return Price_P


def _writeAcceptance(olFleet, _SupplyCurve, Price_P, runchecks=False):
	"""Writes the clearing results from _SupplyCurve to the GenUnits in olFleet."""
	# Get the bid id and simplify to the gen id. Used to index _SupplyCurve for each ol gen
	bids_by_gen = np.around(_SupplyCurve[:, 0] / 100, decimals=0)

	for gen in olFleet:
		# Subset the Supply Curve
		_fromSupply = _SupplyCurve[bids_by_gen == gen.id['rowID']]

		# Copy acceptance
		gen.bids_P[:, 3] = _fromSupply[:, 3]

		# Assert that _fromSupply is in the same order as in bids_P (compare bid ID col)
		assert not runchecks or all((_fromSupply[:, 0] - gen.bids_P[:, 0]) == 0)
		# Assert that all bids under the clearing price are fully accepted
		assert not runchecks or all(gen.bids_P[gen.bids_P[:, 2] < Price_P, 3] == 1)
	return


def _calcSchedule(DxP_idx, olFleet, Results):
	"""Updates the results of the cleared period (DxP_idx) to gen.Schedule and to Results['dispatch MW']"""
	# bids_P column identifiers
	_qty, _Pr, _acc = 1, 2, 3

	# gen.Schedule column identifiers
	_Available, _Po, _ne, _ne_pot, _Qo = 0, 1, 2, 3, 4

	for gen in olFleet:
		# ---------------------------------------------------------------------- 1) Po, net_HHVeff, Qo(cogen) to gen.Schedule
		# ............................................... a) Calc Po
		Po = np.sum(gen.bids_P[:, _acc] * gen.bids_P[:, _qty])
		gen.Schedule[DxP_idx, _Po] = Po

		# ............................................... b) Calc achieved efficiency
		if Po >= gen.Pmin:
			gen.Schedule[DxP_idx, _ne] = gen.effcurve(Po_MW=Po, normalized=False)[0]
		elif Po == 0:
			# Not dispatched, so left as NaN
			pass
		else:
			# todo Note: bandaid soln to P<Pmin problem of marginal units. Assume net_HHVeff at Pmin
			gen.Schedule[DxP_idx, _ne] = gen.effcurve(Po_MW=gen.Pmin, normalized=False)[0]


		# ............................................... c) Calc Cogen
		if gen.is_['Cogen']:
			# Note: HPR is assumed constant
			gen.Schedule[DxP_idx, _Qo] = Po * gen.cogentech['HPR [-]']

		# ---------------------------------------------------------------------- 2) Update Dispatch
		Results['dispatch [MW]'][DxP_idx, gen.id['rowID']] = Po
	return


def post_simulation(Results):
	"""Performs the ff. post-simulation steps:
		Calculates the ff. keys of Results:
			fuel input [MW]
			Total fuel consumption [MWh]
			Schedules
			Stats per gen
			Total output by plant [ktoe]


	RESULTS:
		Results is modified in-place. It adds the ff. keys:

		fuel input [MW]       Fuel input to the GenUnit, as a DataFrame with the same indices as Results['dispatch [MW]']

		Total fuel consumption [MWh]     Total fuel input to all GenUnits.
	"""
	# gen.Schedule column identifiers
	_Available, _Po, _ne, _ne_pot, _Qo = 0, 1, 2, 3, 4
	GenFleet = GenUnit.get_fleet()

	tol = 10**-6

	# ------------------------------------------------------------------------------- 0) Save all gen.Schedule
	# This is done purely for the reason that it would be available if simulation is
	# bypassed and the results are read
	Results['Schedules'] = tuple(gen.Schedule for gen in GenFleet)

	# ------------------------------------------------------------------------------- 1) Fuel input and total fuel conso
	# This is done by assuming a linear (proportional) relationship between the fuel input and dispatch, as defined by
	# the electrical efficiency, gen.Schedule['net_HHVeff']

	# _all_ne follows the same index and cols as Results['dispatch [MW]']
	_all_ne = pd.DataFrame({idx: GenFleet[idx].Schedule[:, _ne] for idx in range(Results['dispatch [MW]'].shape[1])})
	_all_ne.index, _all_ne.columns = Results['dispatch [MW]'].index, Results['dispatch [MW]'].columns

	Results['fuel input [MW]'] = Results['dispatch [MW]'] / _all_ne
	Results['fuel input [MW]'].fillna(0.0, inplace=True)

	Results['Total fuel consumption [MWh]'] = Results['fuel input [MW]'].sum().sum() \
	                                          * time['periods duration'].seconds/3600

	# Todo - c_ is not defined here
	# Results['Total output by plant [ktoe]'] = Results['dispatch [MW]'].sum()* c_['period hr'] * c_['MWh to ktoe']

	# ------------------------------------------------------------------------------- 2) Calc gen dispatch stats
	# Simulation duration in hr
	period_h = time['periods duration'].seconds / 3600
	Duration_h = time['DxP_index'].shape[0] * period_h

	stats_df = pd.DataFrame(
		index=Results['dispatch [MW]'].columns,
		columns=['AF', 'CF', 'UF', 'LF', 'Ave Load [MW]', 'Max Load [MW]', 'Capacity [MW]', 'Total Load [MWh]'],
		dtype='f8', )

	for gen in GenFleet:
		genID = "Gen {id}: {name}".format(id=gen.id['rowID'], name=gen.id['Unit Name'])

		# 0.1) Ave load
		TotalLoad_MWh = np.sum(gen.Schedule[:, _Po]) * period_h
		AveLoad_MW = TotalLoad_MWh / Duration_h

		# 0.2) Max load
		MaxLoad_MW = np.max(gen.Schedule[:, _Po])

		# 1.0) Availability Factor
		AF = np.mean(gen.Schedule[:, _Available])

		# 1.1) Capacity Factor
		CF = AveLoad_MW / gen.GenCap

		# 1.2) Utilization Factor
		UF = MaxLoad_MW / gen.GenCap

		# 1.3) Load Factor
		if MaxLoad_MW > 0:
			LF = AveLoad_MW / MaxLoad_MW
			#assert LF >= CF
		else:
			LF = np.nan


		#assert AveLoad_MW-tol <= MaxLoad_MW <= gen.GenCap+tol
		#assert AF >= CF-tol and UF >= CF-tol

		# ............................................................ Assign
		stats_df.loc[genID, stats_df.columns] = [AF, CF, UF, LF, AveLoad_MW, MaxLoad_MW, gen.GenCap, TotalLoad_MWh]

	Results['Stats per gen'] = stats_df

	# ------------------------------------------------------------------------------- EXIT
	return


class InadequateSupply(Exception):
	"""Raised when the demand is too much for the online supply."""
	pass

#isJupyter = isnotebook()
#if isJupyter:
#	from IPython.display import clear_output
