"""
Collection of algorithms (geared towards abstract math and numerical problems)

Picked up from MA and now continued from Cooling Singapore. Oct 2019
"""

# This control flag is used to skip sections if their prerequisite modules are not available, so that DK_Numerical
# can be imported more flexibly with regards to the environment.

_control = {section: True for section in ('NetworkX', 'Pandas', 'numpy')}

try:
	import networkx as nx
except ModuleNotFoundError:
	_control['NetworkX'] = False

try:
	import pandas as pd
except ModuleNotFoundError:
	_control['Pandas'] = False

try:
	import numpy as np
except ModuleNotFoundError:
	_control['numpy'] = False

import warnings


# -------------------------------------------- NetworkX ------------------------------------------------ #
if _control['NetworkX']:
	def get_treeroot(tree, guess=None):
		"""
		Finds the root node of a NetworkX DiGraph tree.

		Args:
			tree: NetworkX DiGraph tree
			guess: Best bet for which node is the root. The closer this node is to the root, the faster the search.

		Returns:
			Root node of tree.
		"""

		# Step 1: Param checks
		if not isinstance(tree, nx.DiGraph):
			raise TypeError("Must pass a NetworkX Digraph to parameter 'tree.'")

		if not nx.is_tree(tree):
			raise ValueError("Passed graph is not a tree.")

		if guess is None:
			guess = list(tree.nodes)[0]
		else:
			if guess not in tree:
				warnings.warn("The guessed root node is not in the tree.")
				guess = list(tree.nodes)[0]


		# Step 2: Follow predecessors until you find the root
		root = guess
		counter = 1
		n_nodes = len(tree)

		# This is a redundant loop, but Iterator Protocol + NetworkX guarantees that this is unnecessary.
		while counter <= n_nodes:
			try:
				root = next(tree.predecessors(root))
				counter += 1
			except StopIteration: # raised for the root node (no predecessors)
				break


		return root


	def compare_trees(T1, T2):
		"""
		Returns true if the trees are IDENTICAL (isomorphism is necessary but insufficient), and False otherwise.

		Two trees are identical iff:
			1) same set of nodes
			2) same set of branches
			3) same branch params
			(for now, assumed no node params)

		DEVELOPER'S CORNER:
			I learned that dictionaries can be tested for equality via ==. There's also talk on StackOverflow about this
			equality test holding for nested dicts (and other nesting combinations with other data structures).

		"""
		if not isinstance(T1, nx.DiGraph) or not isinstance(T2, nx.DiGraph):
			raise TypeError("This function only accepts NetworkX DiGraphs.")

		if not nx.faster_could_be_isomorphic(T1, T2):
			return False

		# Check 1: Compare nodes
		if set(T1) != set(T2):
			return False

		# Check 2: Compare branches
		br1 = set(br for br in T1.edges())
		br2 = set(br for br in T2.edges())

		if br1 != br2:
			return False

		# Check 3: Compare branch parameters (final test)
		return all([T1.edges[up, down] == T2.edges[up, down] for up, down in br1])


# -------------------------------------------- Pandas ------------------------------------------------ #
if _control['Pandas']:
	def empty_df_with_dtypes(col_dtype: dict):
		"""Initializes a pandas DataFrame with NO ROWS, but with col labels AND datatypes. The DataFrame constructor only
		lets you specify one datatype for all the columns, so this function fills that gap.

		Args:
			col_dtype: as dictionary of 'column label':'numpy dtype' pairs

		Note: This solution is adapted from the one proposed in stackoverflow,
		https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types/48374031#48374031


		Returns:
			Empty DataFrame, with columns + dtype specified.

		"""
		if not isinstance(col_dtype, dict):
			raise TypeError("Pls. pass a dictionary of 'column label':'numpy dtype' pairs.")

		if len(col_dtype) == 0:
			raise RuntimeError("Requested table has no columns.")

		df = pd.DataFrame()

		for c, d in col_dtype.items():
			df[c] = pd.Series(dtype=d)

		return df


	def is_xymonotonic(x, y, slope='pos', getdf=False, as_assertion=False):
		"""Checks if the given relationship <x, y> is monotonic in the specified direction ('pos' or 'neg' slope). A
		monotonic line is one whose slope is consistently either non-neg or non-pos. Pass vectors (iterables) to x
		and y. x has to be strictly increasing, whereas y can have repeating subsequent points (flat slope).

		Returns a tuple (bool, list):
			Bool is True if monotonic in the specified direction.

			List is empty if bool is True. Otherwise, contains the indices (0-indexed range is assigned to the data)

			if getdf is True, then a third item is returned as a pandas DataFrame of vectors x and y, with column
			names as 'x' and 'y'.

		To assert monotonicity:
			assert is_xymonotonic(x, y, slope='pos')[0]

		To assert monotinicity internally
			is_xymonotonic(x, y, slope='pos', as_assertion=True)

		"""

		# ----------------------------------------------------------------------- Checks
		length = len(x)
		if length != len(y):
			raise ValueError("Vectors x and y have to be of the same length.")
		if length < 3:
			raise ValueError("Vectors x and y must have at least 3 points.")



		# ----------------------------------------------------------------------- Convert to Series
		# x2, y2 is the same data but shifted one index forward wrt x1 and y1  -- old implementation
		x = pd.Series(data=(pt for pt in x))
		# x2 = pd.Series(data=x1.values, index=pd.RangeIndex(-1, length-1))

		y = pd.Series(data=(pt for pt in y))
		# y2 = pd.Series(data=y1.values, index=pd.RangeIndex(-1, length-1))

		# ----------------------------------------------------------------------- Calc dx and dy
		# dx = x2-x1
		dx = x.diff()
		dx = dx.loc[dx.notna()]

		# dy = y2-y1
		dy = y.diff()
		dy = dy.loc[dy.notna()]

		# Check that all dx > 0
		Lf_invalid_dx = dx <= 0
		if Lf_invalid_dx.any():
			# Collect indeces of points (+next point) where dx <= 0
			invalid_idx = Lf_invalid_dx.loc[Lf_invalid_dx].index
			invalid_idx_withadj = pd.Index(set(invalid_idx).union(invalid_idx+1))

			print("x vector can only be increasing. Pls. view the slice: \n{}".format(x.loc[invalid_idx_withadj]))
			raise ValueError(invalid_idx_withadj)

		# ----------------------------------------------------------------------- Check if monotonic
		sign = {'pos': 1}.get(slope, -1)
		# sign * dy must be >= 0 for all entries to fulfill the specified monotonicity
		Lf_failed = sign * dy < 0

		returned_value = (not any(Lf_failed), list(Lf_failed.loc[Lf_failed].index))

		if as_assertion:
			assert returned_value[0], returned_value[1]

		if getdf:
			returned_value += (pd.DataFrame({'x': x, 'y': y}), )
		return returned_value


	def sortxy(x, y, ascending=True):
		"""Given vectors x and y (as iterables that are ordered and 1:1), sortxy() sorts these vectors in ascending
		or descending order (via param 'ascending'), and returns them in a DataFrame with a range index. No check is
		done if x has repeating values."""
		# todo consider keeping the original index, if ever necessary

		# ----------------------------------------------------------------------- Checks
		length = len(x)
		if length != len(y):
			raise ValueError("Vectors x and y have to be of the same length.")

		# ----------------------------------------------------------------------- Convert to Series
		# Range index here stores the mapping x --> y
		Serx = pd.Series(data=(pt for pt in x))
		Sery = pd.Series(data=(pt for pt in y))

		# ----------------------------------------------------------------------- Sort and return df
		Serx_sorted = Serx.sort_values(ascending=ascending)

		return pd.DataFrame({
			'x': Serx_sorted.values,
			'y': Sery.loc[Serx_sorted.index].values,
		})


# -------------------------------------------- numpy ------------------------------------------------ #
if _control['numpy']:
	def clip(f, lb=None, ub=None):
		"""Return a clipped version of callable f(). This is the counterpart of numpy.clip(),
		which only works on explicit arrays, for callables.

		lb and ub follow the same requirements as in their counterparts in numpy.clip().
		"""
		if ub < lb:
			raise ValueError("ub < lb")
		return lambda x: np.clip(f(x), lb, ub)


	def apply_f_every_n(arr, n, func=np.sum, res_dtype='f8'):
		"""Apply function func() to every n elements of array. The length of array arr must be a multiple of n,
		but this is not checked (last operation would fall short of elements w/o raising an exception).


		sample usage:
			# Downscale the resolution by half, while getting the mean of the data points
			apply_f_every_n(ser.values, 2, np.mean)

			# You can use this via DataFrame.apply(). Here, we are changing a 24-h, half-hourly dataset into an hourly one.
			df_24h_hh.apply(apply_f_every_n, axis=0, args=(2, np.mean))
		"""
		start_idxs = np.arange(0, arr.shape[0], n)

		return np.fromiter((func(arr[idx:idx + n]) for idx in start_idxs), dtype=res_dtype)


# -------------------------------------------- Uncategorized Python ------------------------------------------------ #
def get_dictvals(mydict):
	"""This function recursively gets all non-dict items in a hierarchical dict data structure. The order of the items have no intended meaning.

	e.g.
		myheir = {
			0: '0.0',
			1: '0.1',
			2: {0: '1.0', 1: '1.1'},
			3: {0: '1.2', 1: {0: '2.0', 1: '2.1', 2: {0: '3.0'}, 3:'2.2'}},
			4: {0: '1.3', 1: '1.4'},
			5: '0.2',
		}

	get_dictvals(myheir)
	>> ['0.0', '0.1', '1.0', '1.1', '1.2', '2.0', '2.1', '3.0', '2.2', '1.3', '1.4', '0.2']

	"""
	values = []

	for val in mydict.values():
		if isinstance(val, dict):
			values.extend(get_dictvals(val))
		else:
			values.append(val)
	return values



# -------------------------------------------- Tuple Arithmetic ------------------------------------------------ #
# FOR DEVELOPMENT:
#    1) Sequences of tuples should be contained in sets.
#

def totuple(item):
	if isinstance(item, (str, int)):
		return tuple([item])
	else:
		return tuple(item)


def tupsum(*addends, strAsOne=True):
	"""
	Performs a tuple addition on an indefinite number of tuples. Arguments must be tuple-coercibles (they must be
	iterables or iterators), or scalars (i.e. tuple([arg]) succeeds).

	Returns:
		A tuple of the tuple sum of the addends.

	DEVELOPER'S CORNER:
		Tuples support concatenation, ie (1) + (0) = (1,0), which is really what tupple addition is in the context of
		the Cartesian product.

		tuple() returns an empty tuple, which guarantees that 'sum' will support concatenation.

		If an addend is not a tuple, tuple() coerces it (same elements).
		If an addend cannot be coerced directly, then it is expected to be a scalar (usu. numeric or string),
		and thus containing it via tuple([a]) should work.
	"""
	sum = tuple()

	for a in addends:
		if type(a) is str and strAsOne:
			a = tuple([a])
		try:
			sum = sum + tuple(a)
		except TypeError:
			sum = sum + tuple([a])

	return sum


def tupadd(a, b):
	"""
	Performs a tuple addition of a and b.

	Args: a and b can either be 1) scalars (i.e. single coordinate) or 2) elementary tuples

	Any argument that is a tuple (assumed elementary; i.e. all its elements are scalars) is unpacked as can be seen
	below.

	As far as the author knows, this function cannot be extended to an arbitrary number of addends (by taking a
	var-positional parameter), because the statement a = *a is not allowed. To sum an indefinite number of addends,
	one way to achieve this is by looping every addend pair, much like how tuppi works.

	Returns:
		The tuple sum of a and b (GUARANTEED elementary tuple if a and b are elementary tuples or scalars).
	"""
	if type(a) == tuple and type(b) == tuple:
		return tuple([*a, *b])
	elif type(a) == tuple:
		return tuple([*a, b])
	elif type(b) == tuple:
		return tuple([a, *b])
	else:
		return tuple([a, b])


def tuppi(*factors):
	"""
	Performs a Cartesian product of all of its factors.

	Args:
		*factors: An iterable of tuple-coercibles. If you want a 1-dim tuple, pass it as (a,) and not a.

	Returns:
		The Cartesian product as a tuple.

	DEVELOPER'S CORNER:
		If the factors are all elementary tuples or scalars, the product is GUARANTEED to be an elementary tuple.
		Even if after every iteration of the for factor .. loop, product is contained in a tuple, this container is
		"opened up" by the same generator expression come next iteration. The container is opened and new (and more)
		elements are repacked in it.

	"""
	# Require that factors is an iterable of tuple-coercibles
	tup_factors = [tuple(i) for i in factors]

	# init product to 1st factor * 1
	product = tup_factors[0]

	for factor in tup_factors[1:]:
		product = tuple(tupadd(a, b) for a in product for b in factor)
		#         RHS is precisely the def'n of the Cartesian Product, product x factor

	return product


def tupsumproduct(tree):
	"""
	This function performs a sum of products.

	Args:
		tree: dictionary of {iterable of parents : iterable of children}

	Returns:
		The tuple sum of products as a tuple, i.e., sum( keys x values )


	DEVELOPER'S CORNER:
		This generator expression will iterate through all its for clauses, before stringing them together.
		As tupadd() is guaranteed to return elementary tuples (so long as arguments are elementary tuples or scalars),
		the generator produces a sequence of elementary tuples, which are then finally packed in the tuple() constructor.
		I could have used a list(), but using tuple() is safe as tuple(any_tuple) == any_tuple. The external
		container type can be any sequence. I just chose tuples for both this function and tuppi so that the
		containers are immutable, if you want to use them directlty.

	"""
	prm_tuples = {totuple(key): totuple(val) for key, val in tree.items()}
	return tuple(tupadd(parent, child) for key, val in prm_tuples.items() for parent in key for child in val)
	#                                  SUM (of heterogenous key:val mappings)     PRODUCTS (key x val)




