#!/usr/bin/env python

from . import prims, distarray, extent, tile
from .tile import Tile
from spartan import util
from spartan.util import join_tuple, Assert
import numpy as np

def largest_value(vals):
  return sorted(vals, key=lambda v: np.prod(v.shape))[-1]

def eval_Value(ctx, prim):
  return prim.value

def eval_Map(ctx, prim):
  inputs = [evaluate(ctx, v) for v in prim.inputs]
  largest = largest_value(inputs)
  map_fn = prim.map_fn
  
  def mapper(ex, tile):
    slc = ex.to_slice()
    local_values = [input[slc] for input in inputs]
    result = map_fn(*local_values)
    assert isinstance(result, np.ndarray), result
    return [(ex, Tile(ex, result))]
  
  return largest.map_to_array(mapper)


def eval_Reduce(ctx, prim):
  input_array = evaluate(ctx, prim.input)
  dtype = prim.dtype_fn(input_array)
  axis = prim.axis
  shape = extent.shape_for_reduction(input_array.shape, prim.axis)
  tile_accum = tile.TileAccum(prim.combiner_fn)
  output_array = distarray.create(ctx, shape, dtype, accum=tile_accum)
  local_reducer = prim.local_reducer_fn
  
  def mapper(ex, tile):
    reduced = local_reducer(ex, tile)
    dst_extent = extent.index_for_reduction(ex, axis)
    output_array.update(dst_extent, reduced)
  
  input_array.foreach(mapper)
  
  return output_array
  
def slice_mapper(ex, tile, region, matching_extents):
  if ex in matching_extents:
    intersection = matching_extents[ex]
    local_slc = extent.offset_slice(ex, intersection)
    output_ex = extent.offset_from(region, intersection)
    return [(output_ex, Tile(intersection, tile[local_slc]))]

def eval_Slice(ctx, prim):
  src = evaluate(ctx, prim.input)
  idx = evaluate(ctx, prim.idx)
  
  slice_region = extent.from_slice(idx, src.shape)
  matching_extents = dict(distarray.extents_for_region(src, slice_region))
  
  util.log('Taking slice: %s from %s', idx, src.shape)
  util.log('Matching: %s', matching_extents)
  
  return src.map_to_array(lambda k, v: slice_mapper(k, v, slice_region, matching_extents))


def int_index_mapper(ex, tile, src, idx, dst):
  '''Map over the index array, fetching rows from the data array.'''
  idx_slc = ex.to_slice()[0]
  idx_vals = idx[idx_slc]
  
  util.log('Dest shape: %s, idx: %s, %s', tile.shape, ex, idx_vals)
  for dst_idx, src_idx in enumerate(idx_vals):
    tile[dst_idx] = src[int(src_idx)]
  return [(ex, tile)]


def bool_index_mapper(ex, tile, src, idx):
  slc = ex.to_slice()
  local_val = src[slc]
  local_idx = idx[slc]
  return [(ex, local_val[local_idx])]


def eval_Index(ctx, prim):
  dst = ctx.create_table()
  src = evaluate(ctx, prim.input)
  idx = evaluate(ctx, prim.idx)
  
  if idx.dtype == np.bool:
    dst = src.map_to_array(bool_index_mapper)
    # scan over output, compute running count of the size 
    # of the first dimension
    row_counts = src.map_to_table(lambda k, v: v.shape[0])
    for _, v in row_counts:
      pass
    raise NotImplementedError
  else:
    # create destination of the appropriate size
    dst = distarray.create(ctx, 
                           join_tuple([idx.shape[0]], src.shape[1:]),
                           dtype = src.dtype)
    
    # map over it, replacing existing items.
    return dst.map_inplace(lambda k, v: int_index_mapper(k, v, src, idx, dst))


def _evaluate(ctx, prim):
  return globals()['eval_' + prim.typename()](ctx, prim)    
    

def evaluate(ctx, prim):
  Assert.is_instance(prim, prims.Primitive) 
  util.log('Evaluating: %s', prim)
  if prim.cached_value is None:
    prim.cached_value = _evaluate(ctx, prim)
  
  return prim.cached_value