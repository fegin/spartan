import sys
import time
import types
import signal
import os
import numpy as np
import scipy.sparse as sp

import spartan
from spartan import expr, util, config
from spartan.config import FLAGS
from spartan.examples.ssvd import svd as ssvd
from spartan.examples.svd import svds as svd

def sig_handler(sig, frame):
  import threading
  import traceback

  for thread_id, stack in sys._current_frames().items():
    print '-' * 100
    traceback.print_stack(stack)

def algorithm_main(master):
  numpy_matrix = np.load('/home/fegin/workspace/dataset/feature-matrix-noNAs-int.npy')
  matrix = expr.from_numpy(numpy_matrix)
  matrix = expr.astype(matrix, np.float64).force()
  matrixT = expr.transpose(matrix).force()
  
  print matrix.shape

  print ('Start to do SSVD')
  begin = time.time()
  U1, S1, T1 = ssvd(matrixT)
  print ('Finished SSVD in %f' % (time.time() - begin))

  print ('Start to do SVD')
  begin = time.time()
  U2, S2, T2 = svd(matrix)
  print ('Finished SVD in %f' % (time.time() - begin))

def main():
  signal.signal(signal.SIGQUIT, sig_handler)
  spartan.config.parse(sys.argv)
  master = spartan.initialize()
  algorithm_main(master)
  spartan.shutdown()

if __name__ == '__main__':
  main()
