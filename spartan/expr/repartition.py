'''
Implementation of the repartition expression.
'''
from spartan import util
from spartan.array import extent
from .builtins import tocsr, tocsc
from .shuffle import shuffle
import numpy as np
import scipy.sparse as sp

def _repartition_nnz_mapper(array, ex, ul_order = None):
  tile = array.fetch(ex)
  order = ul_order[ex.ul]
  new_ex = extent.create((order, 0), (order + 1, 1), (len(ul_order), 1))
  yield (new_ex, np.asarray((tile.nnz)).reshape((1, 1)))

def _repartition_partition_mapper(array, ex, splits = None, ul_order = None, axis = None):
  # Since the return matrix of mapper must be a fixed shape, we have to 
  # decide the maximum partitions per extent for `array`.
  max_partitions = 32

  tile = array.fetch(ex)
  util.log_warn(type(tile))
  split = splits[ex.ul]

  # The first partition should always start from ex.ul[1].
  partition = [ex.lr[1] for i in range(max_partitions)]
  partition[0] = ex.ul[1] 

  previous_nnz = 0
  for i in range(len(split) - 1):
    nnz = split[i] + previous_nnz
    split_point = np.searchsorted(tile.indptr, nnz)
    if tile.indptr[split_point] - nnz > tile.indptr[split_point + 1] - nnz:
      split_point += 1
    previous_nnz = tile.indptr[split_point]
    # The first partition should always start from ex.ul[1].
    if axis == 0:
      partition[i + 1] = split_point + ex.ul[0]
    else:
      partition[i + 1] = split_point + ex.ul[1]

  order = ul_order[ex.ul]
  new_ex = extent.create((order, 0), (order + 1, max_partitions), 
                         (len(ul_order), max_partitions))

  yield (new_ex, np.asarray(partition).reshape((1, max_partitions)))

def _repartition_shuffle_mapper(array, ex, sub_exs_for_tiles = None, 
                               ul_order = None, axis = None):
  order = ul_order[ex.ul]
  sub_exs = sub_exs_for_tiles[order]
  util.log_warn(sub_exs)
  new_tile = None

  for sub_ex in sub_exs:
    tile = array.fetch(sub_ex)
    if new_tile == None:
      new_tile = tile
    else:
      if axis == 0:
        new_tile = sp.vstack([new_tile, tile])
      elif axis == 1:
        new_tile = sp.hstack([new_tile, tile])

  new_ex = extent.create(sub_exs[0].ul, sub_exs[-1].lr, array.shape)
  util.log_warn('nnz = %d' % new_tile.nnz)
  yield (new_ex, new_tile)

def repartition(matrix, axis = 0): 
  ''' Repartition the 'matrix'
  
  This API is used to repartition a sparse matrix to make each extent
  have roughly the same number of non-zero elements. The new extents
  may have different shapes.

  Args:
    matrix: The distarray to be repartitioned. 
            The distarray should be a sparse matrix.
    axis: Scipy only support two-dimensions sparse matrices. Therefore, 
          0 is row and 1 is column.
  Return:
    matrix: A new distarray with new partitions. Whether the distarray is
            csc or csr depends on 'axis'.
  '''

  if axis == 0:
    matrix = tocsr(matrix)
  else:
    matrix = tocsc(matrix)

  matrix = matrix.force()

  all_ul = []
  ul_order = {}
  ul_order_lookup = {}
  all_lr = {}
  for ex in matrix.tiles.iterkeys():
    all_ul.append(ex.ul)
    all_lr[ex.ul] = ex.lr

  # The original extents are stored in a dictionary.
  # But, we need a ordered extent list to do repartition.
  all_ul.sort()
  for i in range(len(all_ul)):
    ul_order[all_ul[i]] = i
    ul_order_lookup[i] = all_ul[i]

  # Get the non-zero elements information from all extents.
  nnz_info = shuffle(matrix, fn = _repartition_nnz_mapper,
                     kw={'ul_order':ul_order}).glom()

  total_elements = 0
  for row in nnz_info:
    total_elements += row[0]
  elements_per_extent = util.divup(total_elements, len(all_ul))

  # Estimate the number of non-zero elements for each new sub-partition.
  previous_nnz = 0
  splits = {}
  for i in range(len(nnz_info)):
    ul = ul_order_lookup[i]
    nnz = nnz_info[i][0]
    splits[ul] = []

    if previous_nnz + nnz >= elements_per_extent: 
      splits[ul].append(elements_per_extent - previous_nnz)
      nnz -= (elements_per_extent - previous_nnz)
      previous_nnz = 0
    elif previous_nnz != 0:
      splits[ul].append(nnz)
      previous_nnz += nnz
      continue

    while nnz >= elements_per_extent:
      splits[ul].append(elements_per_extent)
      nnz -= elements_per_extent
    if nnz != 0:
      splits[ul].append(nnz)
    previous_nnz = nnz

  # Get the final sub-partition from all extents.
  partition_info = shuffle(matrix, fn = _repartition_partition_mapper,
                           kw={'splits':splits, 'ul_order':ul_order,
                               'axis':axis}).glom()

  previous_nnz = 0
  sub_exs_for_tiles = [[] for i in range(len(all_ul))]
  extent_idx = 0
  for order in range(len(all_ul)):
    ul = ul_order_lookup[order]
    lr = all_lr[ul]
    partition = partition_info[order]
    for i in range(len(splits[ul])):
      if axis == 0:
        sub_ex_ul = (partition[i], ul[1])
        sub_ex_lr = (partition[i + 1], ul[1])
      else:
        sub_ex_ul = (ul[0], partition[i])
        sub_ex_lr = (lr[0], partition[i + 1])

      sub_ex = extent.create(sub_ex_ul, sub_ex_lr, matrix.shape)
      sub_exs_for_tiles[extent_idx].append(sub_ex)

      nnz = previous_nnz + splits[ul][i]
      if nnz < elements_per_extent:
        assert i == (len(splits[ul]) - 1), (ul, i, order)
        previous_nnz = nnz
      else:
        assert nnz == elements_per_extent, (nnz, elements_per_extent, previous_nnz)
        extent_idx += 1
        previous_nnz = 0

  # Repartition
  matrix = shuffle(matrix, fn = _repartition_shuffle_mapper,
                   kw={'sub_exs_for_tiles':sub_exs_for_tiles, 'ul_order':ul_order,
                       'axis':axis}).force()

  return matrix
