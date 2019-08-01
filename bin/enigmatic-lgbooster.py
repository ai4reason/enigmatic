#!/usr/bin/python

import sys
import json
from enigmatic.learn import LightGBM

if len(sys.argv) != 4:
   print "usage: %s train.in model.out {json params}"
   sys.exit()

f_in = sys.argv[1]
f_mod = sys.argv[2]
params = json.loads(sys.argv[3])

booster = LightGBM(**params)
booster.train(f_in, f_mod)

