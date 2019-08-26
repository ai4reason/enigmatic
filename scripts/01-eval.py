#!/usr/bin/env python3

from pyprove import expres, log

experiment = {
   "bid": "mizar40/10k",
   #"pids": ["mzr01","aim01","tptp01"],
   "pids": open("eval").read().strip().split("\n"),
   "limit": "T5",
   "cores": 60,
   "eargs": "--training-examples=3 -s --free-numbers"
}

log.start("Evaluation:", experiment)

experiment["results"] = expres.benchmarks.eval(**experiment)

#expres.dump.processed(**experiment)
expres.dump.solved(**experiment)

