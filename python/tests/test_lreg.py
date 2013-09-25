from spartan import util
from spartan.array import distarray, expr
from spartan.util import Assert
import numpy as np
import test_common

DIM = 1000
distarray.TILE_SIZE = DIM ** 2 / 4
  
def _dot(ex, x, w):
  return (ex[0].add_dim(), np.dot(x[ex], w))
  
def test_linear_regression(ctx):
  N_EXAMPLES = 2 * 1000 * 1000 * ctx.num_workers()
  N_DIM = 10
  distarray.TILE_SIZE = N_EXAMPLES / (4 * ctx.num_workers()) 
  x = expr.lazify(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(10000, 10)).evaluate())
  y = expr.lazify(expr.rand(N_EXAMPLES, 1, tile_hint=(10000, 1)).evaluate())
  w = np.random.rand(N_DIM, 1)
  
  for i in range(10):
    yp = expr.map_extents(x, lambda tiles, ex: _dot(ex, x, w))
    Assert.all_eq(yp.shape, y.shape)
    diff = x * (yp - y)
    grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
    w = w - grad * 1e-6
    util.log('Loop: %d', i)
    util.log('Weights: %s', w)
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)