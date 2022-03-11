"""Dispatch, statistics and heat analysis module

Author:         Pang Teng Seng, D. Kayanan
Create date:    Jan. 29, 2020
Revised date:   Feb. 22, 2022
Version:        2.0
Release date:   TBA
"""
# Project dependencies
import DataHandler as dh
import PowerPlants as pp
from gd_core import time, PPdb, config, dFundamentals
mkt_currency = config['market settings']['currency']

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path


# Python
from calendar import month_abbr
import warnings

#from DK_Collections import basic_plot_polishing, plot_exit, fuzzyLookUp
from DK_Collections import *

# ------------------------------------------------- RUDIMENTS -------------------------------------------------------- #
def group_dispatch_by(dispatch, ids_per_class):
	"""Auxiliary function that groups and sums the dispatch data in Results (returned by solve()) according to the
	power plant class grouping, as implied by ids_per_class.

	PARAMETERS:
		dispatch        Results['dispatch [MW]'], a DF of dispatch profiles (results of market clearing),

		ids_per_class   {(gen tech key, fuel key): set(GenUnit ids of this class)}

	DEV NOTES -- more on dispatch:
		Dispatch is indexed by __time['DxP_index'] and with every plant in GenFleet the table (ordered as they appear in
		GenFleet, which is the same order as in PPdb. This means that the ids are the positional column index of
		dispatch).
	"""
	return pd.DataFrame({
		"{} {}".format(*ppcls).strip(): dispatch.iloc[:, list(ids)].sum(axis=1)
		for ppcls, ids in ids_per_class.items()
	})


def calc_fuelmix(Results, ppclasses='by fuel', timeby='month', mixby='output', units='MWh',
                 cogen_alloc=None, percent_byrow=True):
	"""
	(Note: This was originally developed with the misconception that fuel mix could also refers to the input side.
	Fuel mix is always referring to the electricity output (does not take into account efficiencies and other
	products, such as cogen heat)

	Calculate the fuel mix, as specified by:
		- power plant classes
		- time periods
		- input or output
		- as MWh/ktoe/percentage

	The resulting table is returned, as well as recorded in the results dictionary (returned value for the
	key).

	PARAMETERS:
		Results             The results dictionary

		ppclasses           Tuple of 2-tuple power plant classes to group the fuel mix results by,
								e.g. ppclasses=(('CCGT', 'PNG'), ('CCGT', 'LNG'))

							There are predefined power plant classes, which can be accessed by passing the ff:
								'default', 'by fuel', 'Cogens', 'CCGT', 'Oil'

		timeby              The time periods to group the fuel mix by. Valid arguments:
								'all'       Summed for the entire simulation period, with no groupings
								'week'      by weeks
								'month'     by months
								'year'      by years
								None        Then no time aggregation is performed

								You may combine weeks, months and years in one string (e.g. 'month, year') to have
								multi-level groupings (just be careful with 'month week', because weeks shared by two
								months would result into two rows). The result would always be from
								weeks-months-years, regardless of the order in the argument.

		mixby               One of 'input', 'input w/ cogen alloc', or 'output' (default).

								output                      fuel mix (contributions by produced electricity)
								input                       fuel consumption
								input w/ cogen alloc        fuel consumption, with cogen fuel allocated to electricity

		units               (Optional; defaults to 'MWh'). Target units of the fuel mix. Suggested args: '%', 'ktoe'

		cogen_alloc         (Optional; required only if mixby='input w/ cogen alloc') Cogeneration fuel allocation to
							electricity, as pd.Series({gen id: weight})

		percent_byrow       (Optional; defaults to True). This parameter is relevant only if units='%'. If True,
							the resulting rows each sum to 100. Otherwise, the entire table sums to 100.

	RETURN:
	 	The fuel mix as a DataFrame(index= Power plant classes, columns = time grouping). This is also writen in
		Results[str1, str2, str3], where:

			str1    'input mix [-]' or 'output mix [-]', as per parameter mixby, and where - is 'MWh' or '%'
			str2    If ppclasses is predefined, then use this. Else, this would be 'specific'
			str3    Parameter timeby
	"""
	GenFleet = pp.GenUnit.get_fleet()
	# ------------------------------------------------------------ 1) Pre-defined classes
	if isinstance(ppclasses, str):
		res_str2 = ppclasses
		try:
			ppclasses = predefined_classes[ppclasses]
		except KeyError:
			raise ValueError('Passed plant classes is not predefined.')
	else:
		res_str2 = 'specfic'


	# ------------------------------------------------------------ 2) Resolve PP grouping (finalizes ids_per_class)
	#                                                  For fuel mix,  PP grouping must be - mutually exclusive AND
	#                                                                                     - collectively exhaustive
	ids_per_class = {ppcls: pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True) for ppcls in ppclasses}

	# Test 1: Groupings are disjoint
	assert all(ids_per_class[cls1].isdisjoint(ids_per_class[cls2]) for idx1, cls1 in enumerate(ids_per_class.keys())
	           for idx2, cls2 in enumerate(ids_per_class.keys()) if idx2>idx1), \
		"The passed power plant grouping is not mutually exclusive."
		# Double for is a self set cross product of ids_per_class.keys(). idx2>idx1 ensures unique comparisons,
		# (visualize the coords of an upper triangular matrix, excluding the diagonals)


	# Test 2: Must be collectively exhaustive (otherwise, add 'Others')
	allPPs = set(GenFleet.index)
	grouping_allPPs = set()
	for PPcls, grp in ids_per_class.items():
		grouping_allPPs.update(grp)

	others = allPPs - grouping_allPPs

	if len(others) > 0:
		warnings.warn("The passed power plant grouping is not collectively exhaustive. Adding the remaining plants as "
		              "the 'Others' category.")
		ids_per_class[('','Others')] = others

	# ------------------------------------------------------------ 3) Get the relevant PP sched (dispatch or input fuel)
	# Note that PPsched_MW is defined here.
	if mixby == 'input':
		PPsched_MW = Results['fuel input [MW]']

	elif mixby == 'input w/ cogen alloc':
		PPsched_MW = Results['fuel input [MW]'].copy()
		for genid, wt_elec in cogen_alloc.iteritems():
			PPsched_MW.iloc[:, genid] *= wt_elec
	else:
		PPsched_MW = Results['dispatch [MW]']
	# ------------------------------------------------------------ 4) Groupby and SUM (by PP classes and time) [MW]
	# Both groupings would sum the members, so grpd_byPPtime is still in MW
	# .................................................. a) by PP class
	grpd_byPPs = group_dispatch_by(PPsched_MW, ids_per_class)

	# .................................................. b) by time
	# This is technically an integration over time, but the time duration is not factored in until step 5. This is
	# because if the mix is expressed as percent, then this step is unnecesary.

	if timeby == 'all':
		grpd_byPPtime = grpd_byPPs.sum()
		# Convert to 1-row DataFrame for compatibility with step 5
		grpd_byPPtime = pd.DataFrame(grpd_byPPtime.to_dict(), index=('all',))
	elif timeby is None:
		# No time aggregation
		grpd_byPPtime = grpd_byPPs
	else:
		time_grps = []
		for timeperiod in ('year', 'month', 'week', 'day'):
			if timeperiod in timeby:
				time_grps.append(getattr(grpd_byPPs.index, timeperiod))

		grpd_byPPtime = grpd_byPPs.groupby(time_grps).sum()
		grpd_byPPtime.index.name = timeby



	# ------------------------------------------------------------ 5) Scaling to energy or %     [MWh], [%], [others]
	if units == '%':
		# .................................................... a) as percentage
		if percent_byrow:
			period_totals = grpd_byPPtime.sum(axis=1)

			for idx, total in period_totals.iteritems():
				grpd_byPPtime.loc[idx, grpd_byPPtime.columns] = grpd_byPPtime.loc[idx, grpd_byPPtime.columns]/total*100

			assert (abs(grpd_byPPtime.sum(axis=1) - 100) < 10 ** -6).all()

		else:
			grpd_byPPtime = grpd_byPPtime / grpd_byPPtime.sum().sum() * 100
			assert abs(grpd_byPPtime.sum().sum() - 100) < 10 ** -6
	else:
		# .................................................... b.0) First as native MWh (completes time integration)
		grpd_byPPtime = grpd_byPPtime * c_['period hr']
		if mixby == 'output':
			assert abs(grpd_byPPtime.sum().sum() - Results['Total demand [MWh]']) < 10 ** -3              
		elif mixby == 'input':
			assert abs(grpd_byPPtime.sum().sum() - Results['Total fuel consumption [MWh]']) < 10 ** -3

		# .................................................... b.1) Convert if necessary
		if units != 'MWh':
			# Get pint for conversion
			Q_ = dh.Metadata.get_UHaux('Q_')
			conv = Q_('1 MWh').to(units).magnitude

			grpd_byPPtime = grpd_byPPtime * conv


	# ------------------------------------------------------------ 6) Write to Results
	# If single row, then collapse to a Series
	if grpd_byPPtime.shape[0] == 1:
		grpd_byPPtime = grpd_byPPtime.iloc[0, :]


	# 		str1    'input mix [{units}]' or 'output mix [{units}]', as per parameter mixby and units
	# 		str2    If ppclasses is predefined, then use this. Else, this would be 'specific'
	# 		str3    timeby
	Results["{} mix [{}]".format(mixby, units), res_str2, timeby] = grpd_byPPtime

	# ------------------------------------------------------------ Exit
	return grpd_byPPtime


def calc_dispatchstats(Results, ppclasses='default'):
	"""Calculates the ff. stats per power plant class:
		- Availability Factor
		- Capacity Factor
		- Ave Load [MW]
		- Total Capacity [MW]
		- Total Load [ktoe]
	"""
	stats_df = Results['Stats per gen']
	GenFleet = pp.GenUnit.get_fleet()
	tol = 10**-6
	# ------------------------------------------------------------------------ 1) Get pp classes
	if isinstance(ppclasses, str):
		res_str2 = ppclasses
		try:
			ppclasses = predefined_classes[ppclasses]
		except KeyError:
			raise ValueError('Passed plant classes is not predefined.')
	else:
		res_str2 = 'specfic'

	ids_per_class = {ppcls: pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True) for ppcls in
	                 ppclasses}

	# ------------------------------------------------------------------------ 2) Calculate stats per pp class
	cls_stats = pd.DataFrame(index=["{} {}".format(*ppcls).strip() for ppcls in ids_per_class],
	                         columns=['AF', 'CF', 'Ave Load [MW]', 'Total Capacity [MW]', 'Total Load [ktoe]'],
	                         dtype='f8', )

	for ppcls, ids in ids_per_class.items():
		ppcls_str = "{} {}".format(*ppcls).strip()
		subset_df = stats_df.iloc[list(ids), :]

		# 1) AF
		AF = (subset_df['AF'] * subset_df['Capacity [MW]']).sum() / subset_df['Capacity [MW]'].sum()

		# 2) CF
		CF = (subset_df['CF'] * subset_df['Capacity [MW]']).sum() / subset_df['Capacity [MW]'].sum()

		# 3) Ave Load
		AveLoad_MW = subset_df['Ave Load [MW]'].sum()

		# 3) Capacity
		Cap_MW = subset_df['Capacity [MW]'].sum()

		# 4) Total Load
		TotalLoad_ktoe = subset_df['Total Load [MWh]'].sum() * c_['MWh to ktoe']

		# ............................................................ Assign
		assert AF >= CF-tol
		cls_stats.loc[ppcls_str, cls_stats.columns] = [AF, CF, AveLoad_MW, Cap_MW, TotalLoad_ktoe]


	# ------------------------------------------------------------------------ 1)
	Results['Stats', res_str2] = cls_stats
	return cls_stats


# ------------------------------------------------- PLOTTING -------------------------------------------------------- #
def plot_OnlineCapacity(Results, ppclasses='default', totalcap_offset=0.003, show_plot=True, save_as=False, **kwargs):
	"""Plots the online capacity per power plant class. You can plot the total online capacity by specifying
	ppclasses = 'total'. """
	GenFleet = pp.GenUnit.get_fleet()
	#  ----------------------------- Defaults plot features
	def_kws = {
		'title'   : 'Online Capacity',
		'ylabel'  : 'MW',
		'xlabel'  : 'Simulation Period',
		'stacked' : True,
		'xtick_kw': {
			'fontsize': 11,
		},
	}
	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	# ---------------------------------------------------------------------------- 0) Bypass if Total Cap is requested
	# If total online capacity is requested, then bypass this function and call the original implementation.
	if ppclasses == 'total':
		_plot_TotalOnlineCapacity(Results)
		return

	# ---------------------------------------------------------------------------- 1) Calculate capacities
	# .................................................................. a) Total
	TotalCap = PPdb['master']['Registered Capacity [MW]'].sum()

	# .................................................................. b) Calculate online capacities per GenUnit
	# 2D arr of GenCap * Availability from gen.Schedule (stored in Results)
	OlCaps = np.column_stack(tuple(gen.GenCap * Results['Schedules'][gen.id['rowID']][:, 0] for gen in GenFleet))

	# .................................................................. c) Aggregate by ppclass
	if isinstance(ppclasses, str):
		try:
			ppclasses_tup = predefined_classes[ppclasses]
		except KeyError:
			raise ValueError('Passed plant classes is not predefined.')
		store = True
	else:
		ppclasses_tup = ppclasses
		store = False

	ids_per_class = {ppcls: pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True)
	                 for ppcls in ppclasses_tup}

	plot_df = pd.DataFrame({
		"{} {}".format(*ppcls).strip(): OlCaps[:, list(ids)].sum(axis=1)
		for ppcls, ids in ids_per_class.items()
	})

	# ---------------------------------------------------------------------------- 2) Plot
	ax = plot_df.plot.area(figsize=kwargs['figsize'], colormap=kwargs['colormap'], stacked=kwargs['stacked'])

	# Mark total capacity
	if plot_df.max().max() > 0.9 * TotalCap:
		ax.axhline(y=TotalCap, ls='--', color='#95A5A6')
		ax.text(x=plot_df.index[int(0.03 * plot_df.shape[0])], y=min(1 + totalcap_offset) * TotalCap,
		        s="Total: {:0.0f} MW".format(TotalCap))

	# x ticks -- label first week of the month with the month
	locs = []
	labels = []
	for mm in range(1, 13):
		try:
			arr = time['DxP_index'].month == mm
			locs.append(np.where(arr == True)[0].min())
			labels.append(month_abbr[mm])
		except ValueError:
			continue

	plt.xticks(locs, labels, **kwargs['xtick_kw'])

	# legend
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1], title='Power Plant Class', bbox_to_anchor=(1.01, 0.7))

	basic_plot_polishing(ax, **kwargs)
	#plt.show()
	plot_exit(ax, save_as, show_plot)

	# ---------------------------------------------------------------------------- 3) Save and exit
	if store:
		plot_df.index = time['DxP_index']
		Results['Ol Capacities', ppclasses] = plot_df
	return


def _plot_TotalOnlineCapacity(Results, totalcap_offset=0.003, **kwargs):
	"""Plots the online capacity throughout the simulation period. kwargs are for plot polishing (kwargs of
	DK_Collections.basic_plot_polishing)

	PARAMETERS:
		totalcap_offset         Offset for the total capacity text above the marked total capacity line.

	"""
	GenFleet = pp.GenUnit.get_fleet()
	#  ----------------------------- Defaults plot features
	def_kws = {
		'title'    : 'Online Capacity',
		'ylabel'   : 'MW',
		'xlabel'   : 'Simulation Period',
	}
	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	# ---------------------------------------------------------------------------- 1) Calculate capacities
	TotalCap = PPdb['master']['Registered Capacity [MW]'].sum()

	# Calculate the online capacity
	# Pandas perfomance note: use vectorize methods of Series and DataFrame
	# df of online capacity (cols: gen id; rows: simul idx)
	#_df = pd.DataFrame({idx: gen.GenCap * gen.Schedule[:, 0] for idx, gen in enumerate(GenFleet)})
	_df = pd.DataFrame({gen.id['rowID']: gen.GenCap * Results['Schedules'][gen.id['rowID']][:, 0] for gen in GenFleet})
	_OnlineCap = pd.Series(data=_df.sum(axis=1))
	_OnlineCap.index = time['DxP_index']

	# ---------------------------------------------------------------------------- 2) Plot capacities
	plt.figure(figsize=kwargs['figsize'])
	ax = _OnlineCap.plot()

	# Mark total capacity
	ax.axhline(y=TotalCap, ls='--', color='#95A5A6')
	ax.text(x=_OnlineCap.index[int(0.03 * _OnlineCap.shape[0])], y=(1+totalcap_offset)*TotalCap,
	        s="Total: {:0.0f} MW".format(TotalCap))

	# ---------------------------------------------------------------------------- 3) Polish
	basic_plot_polishing(ax, **kwargs)

	# ---------------------------------------------------------------------------- 4) Exit
	plt.show()

	return


def plot_dispatch(Results, ppclasses='default', timeslice=None, plot_demand=True, get_df=False, savefig=False,
                  **kwargs):
	"""Plots the resulting power plant dispatch, by ppclasses.

	PARAMETERS:
		Results             The results dictionary returned by solve()

		ppclasses           Tuple of 2-tuple power plant classes,
								e.g. ppclasses=(('CCGT', 'PNG'), ('CCGT', 'LNG'))

							There are predefined power plant classes, which can be accessed by passing the ff:
								'default', 'by fuel', 'Cogens', 'CCGT', 'Oil'

		timeslice           (Optional; defaults to None). To subset time, pass timeslice = [t1, t2], where t1 and t2
							are date strings as 'YYYY-MM-DD' or 'YYYY-MM' (as in Pandas time index slice). You can
							also pass a single month (e.g. 'Apr') if the simulation period is within the same year.

		plot_demand         (Optional; defaults to True) Plots the demand as well, to see whether or not you've
							included all the plants.

		get_df              (Optional; defaults to False) If True, the grouped dispatch data is returned.

		kwargs              Plot polishing kwargs, as in the kwargs of DK_Collections.basic_plot()
	"""
	GenFleet = pp.GenUnit.get_fleet()
	# ------------------------------------------------------------ 0a) Pre-defined classes
	if isinstance(ppclasses, str):
		try:
			ppclasses = predefined_classes[ppclasses]
		except KeyError:
			raise ValueError('Passed plant classes is not predefined.')

	# ------------------------------------------------------------ 0b) Parse timeslice
	if isinstance(timeslice, str):
		monthnum = str(mmm_tonum[timeslice]).zfill(2)

		_year = time['DxP_index'][0].year

		if _year != time['DxP_index'][-1].year:
			raise ValueError(
				'Cannot pass a month abbreviation if the simulation period is not within the same year.')
		yyyy_mm = "{}-{}".format(_year, monthnum)
		timeslice = [yyyy_mm, yyyy_mm]

	# ------------------------------------------------------------ 0c) Default kwargs
	def_kws = {
		'title_kw' : {'fontsize': 15},
		'ylabel'   : 'MW',
		'ylims': (0, 1.05*Results['demand [MW]'].max()),
		'xlabel'   : 'Simulation Period',
		'bbox_to_anchor': (1.01, 0.7),
		'legend_prop': {'size': 9},
	}
	# Default title incorporates simulation period
	if timeslice:
		_time_title = timeslice
	else:
		_time_title = time['simulation period']

	if _time_title[0] == _time_title[1]:
		def_kws['title'] = 'Dispatch of {}'.format(_time_title[0])
	else:
		def_kws['title'] = 'Dispatch of {} to {}'.format(*_time_title)

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})


	# ------------------------------------------------------------ 1) Group into power plant classes and calc dispatch
	ids_per_class = {ppcls: pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True)
	                 for ppcls in ppclasses}
	grpd_dispatch = group_dispatch_by(Results['dispatch [MW]'], ids_per_class)

	if timeslice:
		grpd_dispatch = grpd_dispatch[timeslice[0]:timeslice[1]]

	# ------------------------------------------------------------ 2a) Plot dispatch
	grpd_dispatch.plot.area(figsize=kwargs['figsize'], colormap=kwargs['colormap'], stacked=kwargs['stacked'])

	# ------------------------------------------------------------ 2b) Plot demand
	if plot_demand:
		if timeslice:
			demand_toplot = Results['demand [MW]'][timeslice[0]:timeslice[1]]
		else:
			demand_toplot = Results['demand [MW]']
		demand_toplot.plot(linestyle='dashed', color='k', label='Demand')

	# ------------------------------------------------------------ 3) Polishing
	ax = plt.gca()
	ax = basic_plot_polishing(ax, **kwargs)

	# legend
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1], title='Power Plant Class',
	          bbox_to_anchor=kwargs['bbox_to_anchor'], prop=kwargs['legend_prop'])

	if savefig:
		plt.savefig(savefig)
	else:
		plt.show()
	if get_df:
		return grpd_dispatch


def plot_fuelmix(Results, ppclasses='default', timeby='week', plotgraph=True, **kwargs):
    """Plots the output fuel mix sharing (as stacked plot, summing to 100) for the whole simulation period.

    PARAMETERS:
        Results             The results dictionary returned by solve()

        ppclasses           Tuple of 2-tuple power plant classes,
                                e.g. ppclasses=(('CCGT', 'PNG'), ('CCGT', 'LNG'))

                            There are predefined power plant classes, which can be accessed by passing the ff:
                                'default', 'by fuel', 'Cogens', 'CCGT', 'Oil'

        timeby              (Optional; defaults to 'month week', such that the simulation period is aggregated
                            weekly and also indexed by month). As in the argument to calc_fuelmix.

        kwargs              Plot polishing kwargs, as in the kwargs of DK_Collections.basic_plot()

        plotgraph           (Optiona; defaults to True) If True, then plt.show() is called. Otherwise,
                            the axes handle is returned.

    DEV NOTES:
        Uses calc_fuelmix()


    """
    # ------------------------------------------------------------ 0) Default kwargs
    def_kws = {
        'title'     : 'Output fuel mix',
        'ylabel'    : '%',
        'ylims'     : (0, 110),
        'xtick_kw'  : {
            'fontsize': 13,
        },
        'legend_prop': {'size': 11},
        'bbox_to_anchor': (1.05, 0.7),

    }
    if time['DxP_index'][0].year == time['DxP_index'][-1].year:
        def_kws['xlabel'] = str(time['DxP_index'][0].year)

    kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
    kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

    # ------------------------------------------------------------ 1) Calculate the fuel mix by output % and by week
    try:
        plot_df = Results['output mix [%]', ppclasses, 'week']
    except KeyError:
        plot_df = calc_fuelmix(Results, ppclasses=ppclasses, timeby=timeby, mixby='output', units='%')

    # ------------------------------------------------------------ 2) Plot
    ax = plot_df.plot.area(figsize=kwargs['figsize'], colormap=kwargs['colormap'], stacked=kwargs['stacked'])

    # ------------------------------------------------------------ 3) Polish
    # x ticks -- label first week of the month with the month
    locs = []
    labels = []
    wk_st, wk_end = plot_df.index[0], plot_df.index[-1]
    for mm, first_wk in first_wk_ofmonth.items():
        if wk_st <= first_wk <= wk_end:
            locs.append(first_wk)
            labels.append(month_abbr[mm])
    plt.xticks(locs, labels, **kwargs['xtick_kw'])

    # legend -- move to the right and present in reverse order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Power Plant Class', bbox_to_anchor=kwargs['bbox_to_anchor'],
              prop=kwargs['legend_prop'])

    # The rest
    ax = basic_plot_polishing(ax, **kwargs)

    if plotgraph:
        plt.show()
    else:
        return ax


def plot_Prices(Results,  timeslice=None, **kwargs):
	"""Plots the clearing prices"""
	# ------------------------------------------------------------ 0b) Parse timeslice
	if isinstance(timeslice, str):
		monthnum = str(mmm_tonum[timeslice]).zfill(2)

		_year = time['DxP_index'][0].year

		if _year != time['DxP_index'][-1].year:
			raise ValueError(
				'Cannot pass a month abbreviation if the simulation period is not within the same year.')
		yyyy_mm = "{}-{}".format(_year, monthnum)
		timeslice = [yyyy_mm, yyyy_mm]


	# ------------------------------------------------------------------- default kwargs
	def_kws = {
		'title'    : 'Clearing Prices',
		'ylabel'   : '{}/MWh'.format(mkt_currency),
		'xlabel'   : 'Simulation Period',
		'ylims': (0, Results['Prices'].max()+10),
		'figsize'  : (12, 8),
		'grid'     : True,
	}
	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})
	# --------------------------------------------------------------------- Plot
	if timeslice:
		plot_df = Results['Prices'][timeslice[0]:timeslice[1]]
	else:
		plot_df = Results['Prices']
	ax = plot_df.plot(figsize=kwargs['figsize'])
	basic_plot_polishing(ax, **kwargs)
	plt.show()
	return


def plot_relFuelPrices(reference_D=0, save_as=False, show_plot=True, **kwargs):
	"""Ad hoc function that plots the relative movement of fuel prices. Parameter reference_D refers to the index of
	the reference day (days before this are removed from the calculation; this can be handy if there is an ISO week
	53), which is also used to normalize the price data.
	"""
	# ------------------------------------------------------------------- 1) Init
	# a) Get the time series fuel prices
	# cols = [key for key, fuel_md in dFundamentals['fuel prices'].items() if fuel_md.dstruct == 'time series']
	cols = []
	for _, majorfuel in predefined_classes['by fuel']:
		if majorfuel == 'Waste': continue
		fuelkey = synonyms.get(majorfuel, majorfuel)

		for matchedkey, _score in fuzzyLookUp(fuelkey, dFundamentals['fuel prices'], matcheslim=3):
			fuel_md = dFundamentals['fuel prices'][matchedkey]
			#if fuel_md.dstruct == 'time series':
			cols.append(matchedkey)


	# b) Initialize the df to query the daily fuel price
	allfuelpr_D = pd.DataFrame(index=time['D_index'][reference_D:], columns=cols, dtype='f8')


	# ------------------------------------------------------------------- 2) Query fuel prices [loc currency]
	for idx, _per in enumerate(allfuelpr_D.index):
		for fuelkey in allfuelpr_D:
			allfuelpr_D.at[_per, fuelkey] = dFundamentals['fuel prices'][fuelkey].tsloc(_per) * dFundamentals[
				'forex'].tsloc(_per)

	# ------------------------------------------------------------------- 3) Normalize [-]
	reference_D = allfuelpr_D.index[0]
	base_prices = {fuelkey: allfuelpr_D.at[reference_D, fuelkey] for fuelkey in allfuelpr_D}

	for fuelkey, basePr in base_prices.items():
		allfuelpr_D[fuelkey] = allfuelpr_D[fuelkey] / basePr

	# ------------------------------------------------------------------- 4) Downscale to weekly res
	normfuelpr_W = allfuelpr_D.groupby(allfuelpr_D.index.week).mean()

	# ------------------------------------------------------------------- 5) Plot
	def_kws = {
		'title'  : 'Relative Fuel Prices',
		'ylabel' : '[-]',
		'ylabel_kw': {'fontsize'  : 14,},
		'xlabel' : 'Simulation Period',
		'xtick_kw': {'fontsize': 13,},
	}
	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	ax = normfuelpr_W.plot(figsize=kwargs['figsize'], colormap=kwargs['colormap'],)

	# x ticks -- label first week of the month with the month
	locs = []
	labels = []
	wk_st, wk_end = normfuelpr_W.index[0], normfuelpr_W.index[-1]
	for mm, first_wk in first_wk_ofmonth.items():
		if wk_st <= first_wk <= wk_end:
			locs.append(first_wk)
			labels.append(month_abbr[mm])
	plt.xticks(locs, labels, **kwargs['xtick_kw'])

	# legend -- move to the right and present in reverse order
	handles, labels = ax.get_legend_handles_labels()
	#ax.legend(handles[::-1], labels[::-1], title='Fuel', bbox_to_anchor=(1.2, 0.7))
	ax.legend(handles[::-1], labels[::-1], title='Fuels', prop={'size': 9.8})

	basic_plot_polishing(ax, **kwargs)
	return plot_exit(ax, save_as, show_plot)


def plot_ElecAndHeatByPlant(Results, orderby='class', show_plot=True, save_as=False, **kwargs):
	"""Bar plot of:
		- elec production
		- total heat (dark red)
		- cogeneration heat (overlayed over total heat and in pink; remaining dark red is thus waste heat)

	First implementation orders plant as in Fleet / PPdb
	"""
	# ------------------------------------------------------------------ i) kwargs
	def_kws = {
		'title'    : 'Power plant gross production and total heat, 2016',
		'title_kw' : {'fontsize': 14, 'fontweight': 'bold'},
		'ylabel'   : 'ktoe',
		'ylabel_kw': {'fontsize': 12, 'fontweight': 'bold'},
		'yticks_kw': {'fontsize': 11.5},

		'grid'     : False,
		'figsize'  : (10, 6),
		'legend'   : True,
	}

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	# ------------------------------------------------------------------ ii) Get locs
	GenFleet = pp.GenUnit.get_fleet()
	# Note: locs_all is in the order of GenFleet; contains plot locs
	#       locs_cogen is in the order of Results['Total cogen heat by plant [ktoe]']; also contains plot locs

	if orderby == 'class':
		ppclasses = predefined_classes['default']

		ids_per_class = {
			' '.join(ppcls).strip(): list(pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True))
			for ppcls in ppclasses}


		locs_start = {}
		locs_all = [None] * len(GenFleet)
		start = 0

		for ppcls, ids in ids_per_class.items():
			locs_start[ppcls] = start

			for offset, genid in enumerate(ids):
				locs_all[genid] = start + offset

			start += 2 + len(ids)

		locs_cogen = [locs_all[genid] for genid in Results['Total cogen heat by plant [ktoe]'].index]
	else:
		locs_all = range(len(GenFleet))
		locs_cogen = [gen.id['rowID'] for gen in GenFleet if gen.is_['Cogen']]

	if 'xlims' not in kwargs:
		kwargs['xlims'] = (min(locs_all)-1, max(locs_all)+1)

	# ------------------------------------------------------------------  patch
	if 'Total output by plant [ktoe]' not in Results:
		Results['Total output by plant [ktoe]'] = Results['dispatch [MW]'].sum() * c_['period hr'] * c_['MWh to ktoe']

	# ------------------------------------------------------------------ iii) Plot
	plt.figure(figsize=kwargs['figsize'])

	# a) Electricity
	plt.bar(x=locs_all, height=Results['Total output by plant [ktoe]'].values, align='edge', width=-0.4,
	        label='Electricity')

	# b) Total heat
	plt.bar(x=locs_all, height=Results['Total PP heat by plant [ktoe]'].values, align='edge', width=0.4, label='Waste Heat',
	        color='#A93226')

	# c) Cogen heat
	plt.bar(x=locs_cogen, height=Results['Total cogen heat by plant [ktoe]'].values, align='edge', width=0.4,
	        label='Cogen Heat', color='#D98880')

	# ------------------------------------------------------------------ iii) Polishing
	ax = plt.gca()

	# x-ticks
	max_len = 20
	labels = []
	for name in Results['Total output by plant [ktoe]'].index:
		if len(name) > max_len:
			name = name[:max_len - 2] + '..'
		labels.append(name)

	plt.xticks(locs_all, labels, rotation=90)

	# other polishing
	basic_plot_polishing(ax, **kwargs)
	ax.yaxis.grid(True)

	return plot_exit(ax, save_as, show_plot)


def plot_wasteheatstr_byplant(heat_summary, Results, orderby='class', show_plot=True, save_as=False, **kwargs):
	"""Similar to the plot_ElecAndHeatByPlant() bar chart, this plots the waste heat streams

	PARAMETERS:
		heat_summary        As returned by ToWRF.summarize_heatstreams()

	"""
	# ------------------------------------------------------------------ i) kwargs
	def_kws = {
		'title'    : '',
		'title_kw' : {'fontsize': 14, 'fontweight': 'bold'},
		'ylabel'   : 'ktoe',
		'ylabel_kw': {'fontsize': 12, 'fontweight': 'bold'},
		'yticks_kw': {'fontsize': 11.5},

		'grid'     : {'axis': 'y'},
		'figsize'  : (12, 8),
		'legend'   : True,

		'color_SH': '#E67E22',
		'color_LH': '#2471A3',
		'color_Sea': '#16A085',
	}

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	# ------------------------------------------------------------------ ii) Get locs
	GenFleet = pp.GenUnit.get_fleet()
	# Note: locs_all is in the order of GenFleet; contains plot locs
	#       locs_cogen is in the order of Results['Total cogen heat by plant [ktoe]']; also contains plot locs

	if orderby == 'class':
		ppclasses = predefined_classes['default']

		ids_per_class = {
			' '.join(ppcls).strip(): list(pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True))
			for ppcls in ppclasses}


		locs_start = {}
		locs_all = [None] * len(GenFleet)
		start = 0

		for ppcls, ids in ids_per_class.items():
			locs_start[ppcls] = start

			for offset, genid in enumerate(ids):
				locs_all[genid] = start + offset

			start += 1 + len(ids)

		locs_cogen = [locs_all[genid] for genid in Results['Total cogen heat by plant [ktoe]'].index]
	else:
		locs_all = range(len(GenFleet))
		locs_cogen = [gen.id['rowID'] for gen in GenFleet if gen.is_['Cogen']]



	# Multiply locs by three so we can put one bar per x-value
	all_locs_0 = [5*_x for _x in locs_all]
	all_locs_1 = [5*_x+1 for _x in locs_all]
	all_locs_2 = [5*_x+2 for _x in locs_all]

	if 'xlims' not in kwargs:
		kwargs['xlims'] = (min(all_locs_0)-1, max(all_locs_2)+1)


	# ------------------------------------------------------------------ iii) Preprocess
	# cols: streams, index: GenFleet sequence
	streams_byplant = heat_summary.iloc[:, 1:].transpose()



	# ------------------------------------------------------------------ iv) Plot
	plt.figure(figsize=kwargs['figsize'])

	# a) Sensible heat
	plt.bar(x=all_locs_0, height=streams_byplant['sensible, air'].values, label='Sensible, air',
	        color=kwargs['color_SH'], width=kwargs['width'], )

	# b) Latent heat
	plt.bar(x=all_locs_1, height=streams_byplant['latent, air'].values, label='Latent, air',
	        color=kwargs['color_LH'], width=kwargs['width'], )

	# c) Seawater
	plt.bar(x=all_locs_2, height=streams_byplant['seawater'].values, label='Seawater',
	        color=kwargs['color_Sea'], width=kwargs['width'], )

	# ------------------------------------------------------------------ v) Polishing
	ax = plt.gca()

	# x-ticks
	max_len = 20
	labels = []
	for name in Results['dispatch [MW]'].columns:
		if len(name) > max_len:
			name = name[:max_len - 2] + '..'
		labels.append(name)

	plt.xticks(all_locs_0, labels, rotation=90)


	basic_plot_polishing(ax, **kwargs)
	return plot_exit(ax, save_as, show_plot)








# ----------------------------------------------- Final Analyses ----------------------------------------------------- #
def calc_allheat(Results):
	"""Adds the ff. keys to Results:

	Input-Output
		Total fuel consumption [ktoe]
		Total demand [ktoe]

	Elec & fuel by plant
		Total output by plant [ktoe]
		Total fuel conso by plant [ktoe]

	Cogeneration
		Cogen heat [MW]
		Total cogen heat by plant [ktoe]
		Cogen heat per mo [ktoe]
		Total cogen heat [ktoe]

	Total PP heat
		Total PP heat [MW]
		Total PP heat [ktoe]
		Total PP heat by plant [ktoe]
	"""
	GenFleet = pp.GenUnit.get_fleet()
	cogens = pp.get_plantclass(('cogen', ''), GenFleet, PPdb, synonyms=synonyms, as_ids=True)

	# ----------------------------------------------------------------------------- a) Initial calc
	# Cogen Qo schedule [MW]
	cogen_sched_MW = pd.DataFrame({idx: Results['Schedules'][idx][:, 4] for idx in cogens},
	                                          index=time['DxP_index'])

	# Cogen heat per month [ktoe]
	cogen_msched_ktoe = cogen_sched_MW.sum(axis=1).groupby(cogen_sched_MW.index.month).sum() * c_['period hr'] * c_[
		'MWh to ktoe']

	# ----------------------------------------------------------------------------- b) Update Results
	Results.update({
		# Input and output
		'Total fuel consumption [ktoe]': Results['Total fuel consumption [MWh]'] * c_['MWh to ktoe'],

		'Total demand [ktoe]' : Results['Total demand [MWh]'] * c_['MWh to ktoe'],

		# Elec & fuel by plant
		'Total output by plant[ktoe]': Results['dispatch [MW]'].sum()* c_['period hr'] * c_['MWh to ktoe'],
		'Total fuel conso by plant [ktoe]': Results['fuel input [MW]'].sum() * c_['period hr'] * c_['MWh to ktoe'],

		# Cogen heat
		'Cogen heat [MW]': cogen_sched_MW,

		'Total cogen heat by plant [ktoe]': cogen_sched_MW.sum() * c_['period hr'] * c_['MWh to ktoe'],

		'Cogen heat per mo [ktoe]': cogen_msched_ktoe,

		'Total cogen heat [ktoe]': cogen_msched_ktoe.sum(),

		# Total PP heat
		'Total PP heat [MW]': (Results['fuel input [MW]']-Results['dispatch [MW]']).sum(axis=1),
	})
	Results.update({
		'Total PP heat [ktoe]': Results['Total fuel consumption [ktoe]'] - Results['Total demand [ktoe]'],
		'Total PP waste heat [ktoe]': Results['Total fuel consumption [ktoe]'] - Results['Total demand [ktoe]']
		                              - Results['Total cogen heat [ktoe]'],
		# Waste+Cogen
		'Total PP heat by plant [ktoe]': (Results['fuel input [MW]']-Results['dispatch [MW]']).sum(axis=0) *
		                                 c_['period hr'] * c_['MWh to ktoe'],
	})

	return


def plot_params(path, param, ppclasses='default', get_df=False, **kwargs):
	"""Reads the calibration's output parameters file and plots the parameter given by param.

	ARGUMENTS:
		path        Absolute path of the calibrated parameters file
		param       Parameter name
		ppclasses   As in calc_fuelmix()


	"""
	GenFleet = pp.GenUnit.get_fleet()
	if isinstance(ppclasses, str):
		ppclasses = predefined_classes[ppclasses]

	# ................................................................... a) Prep pp classes
	ids_per_class = {ppcls: pp.get_plantclass(ppcls, GenFleet, PPdb, synonyms=synonyms, as_ids=True) for ppcls in
	                 ppclasses}

	# ................................................................... b) Read params file
	params_df = pd.read_csv(path, index_col=0)

	# Param bounds Note: not used right now
	bounds = {}
	# i) net_HHVeff
	bounds['net_HHVeff [%]'] = {
		('CCGT', 'PNG')      : np.array([45, 63]),
		('CCGT', 'LNG')      : np.array([45, 63]),
		('Cogen CCGT', 'PNG'): np.array([34, 40]),
		('Cogen CCGT', 'LNG'): np.array([34, 40]),
		('Cogen ST', 'Coal') : np.array([22, 40]),
		('', 'Oil')          : np.array([30, 44]),
		('WtE', '')          : np.array([17, 24]),
	}
	# ii) Alim
	bounds['Alim [-]'] = np.array([0.844, 0.97])

	# iii) AF [yr-1]
	bounds['AF [yr-1]'] = np.array([1, 24])

	# ................................................................... c) Plot prep
	def_kws = {
		'title'  : param,
	}
	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})


	plt.figure(figsize=kwargs['figsize'])
	ylevel = 0
	cmap = plt.cm.get_cmap(kwargs['colormap'])
	N = len(ids_per_class)
	labels = []

	# ................................................................... d) Plotting
	if param != 'net_HHVeff [%]':
		arr_bounds = bounds.get(param, np.array([0,0]))

	for ppcls, ids in ids_per_class.items():
		# lb, ub first
		if param == 'net_HHVeff [%]':
			arr_bounds = bounds['net_HHVeff [%]'][ppcls]

		plt.plot(arr_bounds, np.zeros_like(arr_bounds) + ylevel, 'kd')

		# Params second
		arr_param = params_df.loc[ids, param].values
		label = "{} {}".format(*ppcls).strip()
		labels.append(label)

		plt.plot(arr_param, np.zeros_like(arr_param) + ylevel, 'o', label=label,
		         color=cmap(ylevel / N))

		ylevel += 1


	ax = plt.gca()

	# legend
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1], title='Power Plant Class', bbox_to_anchor=(1.3, 0.7))

	# yticks
	plt.yticks(list(range(N)), labels)

	basic_plot_polishing(ax, **kwargs)
	plt.show()

	if get_df:
		return params_df


def plot_outfmix_errors(err_out, **kwargs):
	"""Plots the output fuel mix errors table (a DataFrame with index=1 to 12, columns=major fuels)"""
	def_kws = {
		'title'       : 'Output fuel mix error',
		'xlabel'      : '2016',
		'ylabel'      : 'ktoe',
		'xlims'       : (1, 12),
		'fuels colors': {
			'Natural Gas'       : '#117A65',
			'Petroleum Products': '#7D3C98',
			'Coal'              : '#5D6D7E',
			'Others'            : '#D4AC0D',
		},
		'xticks_kw'   : {'fontsize': 12},
		'yticks_kw'   : {'fontsize': 11},

	}

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	plt.figure(figsize=kwargs['figsize'])
	for colname, Ser in err_out.iteritems():
		plt.plot(Ser, label=colname, color=kwargs['fuels colors'][colname])

	ax = plt.gca()

	# x-axis
	locs = range(1, 13)
	labels = (month_abbr[idx] for idx in locs)
	plt.xticks(locs, labels, **kwargs['xticks_kw'])

	# legend
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, title='Fuels', prop={'size': 9.8})

	# Basic plot polishing
	basic_plot_polishing(ax, **kwargs)

	plt.show()


def plot_outfmix_errors_II(err_out, save_as=False, show_plot=True, **kwargs):
	width = 6
	base_locs = np.arange(0, 12 * width, width)

	def_kws = {
		'title'       : 'Output fuel mix error',
		'xlabel'      : '2016',
		'ylabel'      : 'ktoe',
		'xlims'       : (0, base_locs.max() + width),
		'ylims'       : (-15, 15),

		'fuels colors': {
			'Natural Gas'       : '#117A65',
			'Petroleum Products': '#7D3C98',
			'Coal'              : '#5D6D7E',
			'Others'            : '#D4AC0D',
		},
		'xticks_kw'   : {'fontsize': 12},
		'yticks_kw'   : {'fontsize': 11},

	}

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	plt.figure(figsize=kwargs['figsize'])

	for idx, fuel in enumerate(err_out):
		plt.bar(x=base_locs + idx, height=err_out[fuel].values, align='edge', width=1,
		        label=fuel, color=kwargs['fuels colors'][fuel])
	ax = plt.gca()

	# x-axis
	plt.xticks(base_locs + 2, month_abbr[1:], **kwargs['xticks_kw'])

	# legend
	handles, labels = ax.get_legend_handles_labels()
	def_legend_kw = {
		'title': 'Fuels',
		'prop': {'size': 9.8},
	}
	passed_legend_kw = kwargs.get('legend_kw', {})
	passed_legend_kw.update({key: val for key, val in def_legend_kw.items() if key not in passed_legend_kw})

	ax.legend(handles, labels, **passed_legend_kw)
	#ax.legend(handles, labels, title='Fuels', prop={'size': 9.8})

	# Basic plot polishing
	basic_plot_polishing(ax, **kwargs)

	ax = plot_exit(ax, save_as, show_plot)
	return ax


def plot_fuelmixcomp(actual, res, fuel='Natural Gas', actual_label=None, res_label=None, prev_ax=None, save_as=False,
                     show_plot=True, **kwargs):
	def_kwgs = {
		'figsize'     : (12, 8),
		'ylims'       : (0, actual[fuel].max() * 1.1),

		'fuels colors': {
			'Natural Gas'       : '#117A65',
			'Petroleum Products': '#7D3C98',
			'Coal'              : '#5D6D7E',
			'Others'            : '#D4AC0D',
		},
		'ylabel'      : 'Electricity Produced [ktoe]',
		'ylabel_kw'   : {'fontweight': 'bold', 'fontsize': 12.5},

		'xlabel'      : '2016',
		'xlabel_kw'   : {'fontweight': 'bold', 'fontsize': 13.5},

		'xticks_kw'   : {'fontsize': 12},
		'yticks_kw'   : {'fontsize': 12},

	}
	kwargs.update({key: val for key, val in def_kwgs.items() if key not in kwargs})

	# --------------------------------------------------------------------------- PLOT
	if prev_ax is None:
		plt.figure(figsize=kwargs['figsize'])

	plt.plot(actual[fuel], marker="o", color=kwargs['fuels colors'][fuel], label=actual_label)
	plt.plot(res[fuel], marker="x", color=kwargs['fuels colors'][fuel], linestyle='--', label=res_label)

	# --------------------------------------------------------------------------- Polish
	ax = plt.gca()

	# x-axis
	locs = range(1, 13)
	labels = (month_abbr[idx] for idx in locs)
	plt.xticks(locs, labels, **kwargs['xticks_kw'])

	# legend
	if actual_label:
		ax.legend(**kwargs.get('legend_kw', {}))

	# Basic plot polishing
	basic_plot_polishing(ax, **kwargs)

	return plot_exit(ax, save_as, show_plot)


def plot_ScenarioComp(df_comp, save_as=False, show_plot=True, **kwargs):
	"""Bar plot of df_comp, which is assumed to the first two columns as:
		Scenario | Base |
		(other cols are ignored)

	Scenario is plotted first, and then Base is plotted over it.

	e.g.

	plot_ScenarioComp(df_Fuel_bydefault, title='Added Fuel Consumption',
                  xlabels=['CCGT\nPNG', 'CCGT\nLNG', 'Cogen\nCCGT PNG', 'Cogen \nCGT LNG',
       'Cogen\nST Coal', 'Oil', 'WtE'],
                  scenario_color='#1F618D',
                  base_color='#85C1E9',
                  bar_width=0.4,
                  figsize=(8,6),
                  legend_prop= {'size': 12},
                 )
	"""
	# ------------------------------------------------------------------ i) kwargs
	def_kws = {
		'title_kw'      : {'fontsize': 14, 'fontweight': 'bold'},
		'ylabel'        : 'ktoe',
		'ylabel_kw'     : {'fontsize': 13, 'fontweight': 'bold'},
		'yticks_kw'     : {'fontsize': 12},

		'grid'          : False,
		'figsize'       : (8, 6),
		'legend'        : True,
		'scenario_color': '#1F618D',
		'base_color'    : '#85C1E9',
		'xtick_kw'      : {'fontsize': 11, 'rotation': 0, },
		'bar_width'     : 0.4,
		'legend_prop'   : {'size': 12},
		'step'          : 1,
		'xlabels'       : df_comp.index,
	}

	kwargs.update({key: val for key, val in def_kws.items() if key not in kwargs})
	kwargs.update({key: val for key, val in plotdef_kws.items() if key not in kwargs})

	# ------------------------------------------------------------------ ii) plot
	locs = range(0, df_comp.shape[0] * kwargs['step'], kwargs['step'])
	Ser_scenario = df_comp.iloc[:, 0]
	Ser_base = df_comp.iloc[:, 1]

	plt.figure(figsize=kwargs['figsize'])
	plt.bar(x=locs, height=Ser_scenario.values, align='center', width=kwargs['bar_width'],
	        color=kwargs['scenario_color'],
	        label=Ser_scenario.name)
	plt.bar(x=locs, height=Ser_base.values, align='center', width=kwargs['bar_width'], color=kwargs['base_color'],
	        label=Ser_base.name)

	# ------------------------------------------------------------------ iii) polish
	ax = plt.gca()

	# xticks
	plt.xticks(locs, kwargs['xlabels'], **kwargs['xtick_kw'])

	basic_plot_polishing(ax, **kwargs)
	ax.yaxis.grid(True)

	# legend
	ax.legend(prop=kwargs['legend_prop'])

	return plot_exit(ax, save_as, show_plot)


def getKeys(Results):
	print("\n".join("\t{}".format(key) for key in Results.keys()))
	return










# --------------------------------------------- SIMPLE CONSTANTS ----------------------------------------------------- #
# Predefined power plant classes, as arguments to the ff. analysis functions:
#   - plot_dispatch()
#   - calc_fuelmix()
predefined_classes = {
	'default': (('CCGT', 'PNG'), ('CCGT', 'LNG'), ('Cogen CCGT', 'PNG'), ('Cogen CCGT', 'LNG'),
	            ('Cogen ST', 'Coal'), ('', 'Oil'), ('WtE', ''),),

	# remember that '' means all
	'by fuel': (('', 'PNG'), ('', 'LNG'), ('', 'Waste'), ('', 'Coal'), ('', 'Oil')),

	'by fuel type': (('', 'PNG'), ('', 'LNG'), ('', 'Waste'), ('', 'Coal'), ('', 'Oil'), ('', 'Biomass')),    # newly added

	'by fuel type 1': (('', 'Natural Gas'), ('', 'Waste'), ('', 'Coal'), ('', 'Oil'), ('', 'Biomass')),           # newly added
    
	'by fuel type 2': (('', 'Natural Gas'), ('', 'Oil'), ('', 'Coal'),),  
    
	'by turbine': (('CCGT', ''), ('Cogen', ''), ('OCGT', ''), ('ST', '')),

	'Cogens' : (('Cogen CCGT', ''), ('Cogen ST', ''),),

	'CCGT'   : (('CCGT', ''),),

	'Oil'    : (('ST', 'Oil'), ('OCGT', 'Oil')),
}

# Synonyms for get pp.get_plantclass()
synonyms = {
		'Oil'     : 'Crude',
		'Cogen ST': 'Cogen Extraction ST',
		'Natural Gas': 'nat gas',     
	}


# Calendar months to number
mmm_tonum = {mm: num  for num, mm in enumerate(month_abbr) if num !=0}

# Constants
Q_ = dh.Metadata.get_UHaux('Q_')
c_ = {
	'MWh to ktoe':  Q_('1 MWh').to('ktoe').magnitude,
	'period hr':    time['periods duration'].seconds/3600,
}

# Shared default plotting kwargs
plotdef_kws = {
	# Titles and labels
	'title_kw' : {'fontsize': 15},

	'ylabel_kw': {
		'fontweight': 'bold',
		'fontsize'  : 14,
	},

	'xlabel_kw': {
		'fontsize': 14,
	},

	# legend
	'legend_prop': {'size': 9},

	# Stacked area plots
	'colormap' : 'RdYlGn_r',
	'stacked': True,

	# General figure
	'figsize'   : (10, 6),
	'grid'      : True,
}

# todo - does this hold for leap years as well?
first_wk_ofmonth =  {
	1: 1,
	2: 5,
	3: 9,
	4: 13,
	5: 17,
	6: 22,
	7: 26,
	8: 31,
	9: 35,
	10: 39,
	11: 44,
	12: 48
}


