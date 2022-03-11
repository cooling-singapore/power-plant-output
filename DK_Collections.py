"""Generic utility scripts"""
# From Others.py in DK_Collections
#from Others import *
from os import path
path_dir = path.dirname(path.abspath(__file__))

# CONDITIONAL IMPORT
_deps = {mod: False for mod in ('fuzzywuzzy', 'scipy family')}

try:
	from fuzzywuzzy import process
	_deps['fuzzywuzzy'] = True
except ImportError:
	pass

try:
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	_deps['scipy family'] = True
except ImportError:
	pass


class Bunch(object):
	"""Converts a dictionary into an object whose attributes are the dict keys mapped to the values.

	e.g.
		x = Bunch({
		    'a': 1,
		    'b': 20,
		})
		x.a
		>>> 1

	An dictionary-like update method is also defined to allow you to update the attributes with another dictionary.

	"""
	def __init__(self, adict):
		self.__dict__.update(adict)
		return

	def update(self, adict):
		self.__dict__.update(adict)
		return


def fmtdict(dictio: dict, keyval_line=None):
	"""Alternative string representation of a dict, formatted with a new line for every key:val pair,
	and with proper spacing. You can customize how every line is printed by passing a formatted string to
	keyval_line, which accepts two parameters: key, val (e.g. "{key}: {val}") """
	max_keylen = max(len(key) for key in dictio.keys())

	if keyval_line is None:
		keyval_line = "\t{{key:<{maxl}}}\t{{val}}".format(maxl=max_keylen)
		#           = tab | space-padded key up to length of longest key | tab | val
	return "\n".join(keyval_line.format(key=key, val=val) for key, val in dictio.items())


if _deps['fuzzywuzzy']:
	def fuzzyLookUp(key: str, table, get='val', matcheslim=1, tol=85, NoMatch='KeyError', **kwargs):
		"""Implements a fuzzy look up of a table for key: value pairs that match a hit, using fuzzywuzzy.process.

		REQUIRED PARAMETERS:
			key         The query string.
			table       The table of key-value pairs. Supports a dictionary and a Pandas Series.

		OPTIONAL:
			get         This parameter is relevant if and only if matcheslim=1. Supports two modes:
							get='val'           (default) returns the value of the hit
							get='key'           returns the hit's key
							get='key, val'      returns the tuple of the hit's key, value

			matcheslim      The maximum number of matches to return. If matcheslim > 1, then a list of (key, score) is
							returned, ranked by relevance (as returned by fuzzywuzzy.process.extract()). If
							matcheslim=None, then all matches are returned.

			tol         The match threshhold. The matches returned by fuzzywuzzy.process.extract() are filtered,
						and only those with scores >= tol are considered.

			NoMatch     Defines the behavior if no matches are obtained.
							if NoMatch == 'KeyError', then raise KeyError
							else, return NoMatch as a default value

			kwargs      Additional parameters for fuzzywuzzy.process.extract()

		RETURNS:
			if matcheslim=1, then the ff. are returned based on get

				get='val'           value mapped to key
				get='key'           matching key
				get='key, val'      key, val

			Otherwise, returns a tuple of (matching key, score) (highest scores first)
		"""

		# ......................................................... a) Get the table keys
		if isinstance(table, dict):
			choices = table.keys()
		elif 'pd' in globals() and isinstance(table, pd.Series):
			choices = table.index
		else:
			RuntimeError('Could not extract the keys of the table.')

		# ......................................................... b) Extract the fuzzy matches (w/ filters)
		if matcheslim is None:
			matcheslim = len(choices)


		# process.extract() returns a list of (matching key, score), with 0 <= score <= 100
		# As many matches as the limit param is returned (even if score ~10)
		# Thus, filter this with the set tolerance (default 85 seems to be good enough for the cases I've seen)
		extracted = tuple(itm for itm in process.extract(key, choices, limit=matcheslim, **kwargs) if itm[1] >= tol)

		# ......................................................... c) Return based on get
		#                                           c.1) No matches
		if len(extracted) == 0:
			if NoMatch == 'KeyError':
				raise KeyError(key)
			elif get == 'key, val':
				return None, None
			else:
				return None

		#                                           c.2) Allow multiple matches
		if matcheslim != 1:
			return extracted

		#                                           c.3) One match only
		key = extracted[0][0]
		val = table[key]

		# return eval(get) works but not elegant
		if get == 'val':
			return val
		elif get == 'key':
			return key
		elif get == 'key, val':
			return key, val
		else:
			raise ValueError('Parameter get has an unexpected value.')

	# Alias -- think of a better name
	fuzGet = fuzzyLookUp


if _deps['scipy family']:
	def basicplot(data, plotter='plot', newfig=True, showplot=True, **kwargs):
		"""Basic plotting feature. Can be used stand-alone, in which case you can ignore the 'newfig' and 'plotfig'
		parameters. Otherwise, you may suppress these by passing False and using the returned axes object.

		PARAMETERS:
			data            Data to plot. Can support Series, DataFrame, numpy array.
			plotter         Pass the pyplot plotting function to use as a string.
			newfig          If True (default), calls plt.figure() before the plotting function is called.
			plotfig         If True (default), calls plt.show() after the plotting function is called.

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

		if newfig:
			plt.figure(figsize=kwargs.get('figsize', (8, 6)))

		# ............................................................................ Plot per type
		if isinstance(data, (pd.Series, np.ndarray)):
			if plotter == 'plot':
				if hasattr(data, 'name'):
					plt.plot(data, label=data.name)
				else:
					plt.plot(data)
			elif plotter == 'step':
				plt.step(data.index, data, where=kwargs.get('where', 'pre'))

		elif isinstance(data, pd.DataFrame):
			for colname, Ser in data.iteritems():
				if plotter == 'plot':
					plt.plot(Ser, label=colname)
				else:
					raise NotImplementedError

			kwargs['legend'] = True
		else:
			raise NotImplementedError('Basic plotting is not supported for the underlying data type.')

		# ............................................................................ polishing
		#ax = plt.gca()
		ax = basic_plot_polishing(plt.gca(), **kwargs)
		# ............................................................................ EXIT
		if showplot:
			plt.show()
			return
		else:
			return ax

	def basic_plot_polishing(ax, **kwargs):
		"""Template for polishing a matplotlib plot."""
		# Title
		ax.set_title(kwargs.get('title'), **kwargs.get('title_kw', {}))

		# ............................................... X- and Y-axes
		# Axes Labels
		ax.set_xlabel(kwargs.get('xlabel'), **kwargs.get('xlabel_kw', {}))
		ax.set_ylabel(kwargs.get('ylabel'), **kwargs.get('ylabel_kw', {}))
		# Limits
		ax.set_xlim(kwargs.get('xlims'))
		ax.set_ylim(kwargs.get('ylims'))
		# Ticks
		plt.xticks(**kwargs.get('xticks_kw', {}))
		plt.yticks(**kwargs.get('yticks_kw', {}))

		# ............................................... Grid, legend
		#if kwargs.get('grid', True):
			#ax.grid(True)
		if kwargs.get('grid'):
			if kwargs['grid'] is True:
				ax.grid()
			else:
				ax.grid(**kwargs['grid'])

		# todo recommend to interpret legend as the kw params
		#if kwargs.get('legend'):
		#	ax.legend(**kwargs.get('legend_kw', {}))

		if kwargs.get('legend'):
			# backwards compatibility and allows default call
			if kwargs['legend'] is True:
				ax.legend()
			else:
				ax.legend(**kwargs.get('legend'))

		return ax

	def plot_exit(ax=None, save_as=False, show_plot=True, dpi=300):
		"""Standard exit sequence of plots with two independent options:

		if save_as='*.png', then figure is saved
		if show_plot, then plt.show() is called. Else, ax is returned


		"""
		if save_as:
			plt.savefig(path.join(path_dir, 'New figs', save_as))

		if show_plot:
			plt.show()
			return
		else:
			return ax
