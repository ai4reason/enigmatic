#!/usr/bin/env python3

from pyprove import expres, log

experiment = {
   "bid": "test/bushy100",
   "cores": 4,
}

log.start("CNFize Benchmark(s)", experiment)

expres.benchmarks.cnfize(**experiment)

