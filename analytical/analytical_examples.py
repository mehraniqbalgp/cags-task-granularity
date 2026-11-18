"""Reproduce the survey's hypothetical examples; this is not a benchmark.
Python 3 standard library only. All time variables below are in seconds.
"""
from fractions import Fraction as F
from math import ceil

W, P, h = F(1), 8, F(2, 10**6)
grains = [F(1,10**7), F(1,10**6), F(10,10**6), F(100,10**6),
          F(1,1000), F(125,1000), F(250,1000)]
print('Analytical Table III: W=1 s, P=8, h=2 microseconds')
print('grain_seconds,tasks,elapsed_seconds,speedup,aggregate_worker_cpu_seconds')
for g in grains:
    n = W / g
    assert n.denominator == 1
    t = ceil(n/P) * (g+h)
    print(f'{float(g):.7f},{int(n)},{float(t):.6f},{float(W/t):.2f},{float(n*(g+h)):.6f}')
eta = F(9,10)
print('\n90% target: minimum useful time / overhead =', eta/(1-eta))
print('Scheduler illustration (lower bounds, not completion times):')
for g in [F(10,10**6), F(100,10**6)]:
    n = W/g
    print('g=',float(g),'worker bound=',float(n*(g+h)/64),
          'scheduler bound=',float(n/10**6))
print('Imbalance illustration: coarse 0.700 s; refined',
      float(ceil(F(10,4))*(F(100,1000)+F(1,1000))), 's')
