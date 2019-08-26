#!/usr/bin/env python3

from pyprove import log, expres
from enigmatic import models, learn

settings = {
   "bid"     : "mizar40/10k",
   "pids"    : ["mzr02"],
   "ref"     : "mzr02",
   "limit"   : "T5",
   "cores"   : 68,
   "version" : "VHSLCXPh",
   "eargs"   : "--training-examples=3 -s --free-numbers",
   "hashing" : 2**15,
   #"ramdisk" : "/dev/shm/enigma",
   "learner" : learn.LightGBM(
      max_depth=16,
      num_round=150,
      num_leaves=600,
      learning_rate=0.2,
   )
}
  
log.start("Starting Enigma experiments:", settings)

model = models.name(**settings)
for n in range(3):
   models.loop(model, settings, nick="loop%02d"%n)
   expres.dump.solved(**settings)
   
expres.dump.processed(**settings)
expres.dump.solved(**settings)

