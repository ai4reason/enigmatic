#!/usr/bin/python

import sys
import json
from enigmatic.learn import LightGBM

if len(sys.argv) != 5:
   print "usage: %s train.in model.out train.stats {json params}" % sys.argv[0]
   sys.exit()

f_in = sys.argv[1]
f_mod = sys.argv[2]
f_stats = sys.argv[3]
params = json.loads(sys.argv[4])

booster = LightGBM(**params)
booster.train(f_in, f_mod, f_stats=f_stats)

