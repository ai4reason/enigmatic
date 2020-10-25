import os
import json
from multiprocessing import Process, Manager
import logging

from . import trains, protos, enigmap
from pyprove import expres, log

DEFAULT_NAME = "Enigma"
DEFAULT_DIR = os.getenv("ENIGMA_ROOT", DEFAULT_NAME)

logger = logging.getLogger(__name__)

def name(bid, limit, dataname, features, learner, **others):
   return "%s-%s/%s/%s/%s" % (bid.replace("/","-"), limit, dataname, features, learner.desc())

def path(**others):
   return os.path.join(DEFAULT_DIR, name(**others))

def pathfile(f_file, **others):
   return os.path.join(path(**others), f_file)

def filename(learner, **others):
   model = name(learner=learner, **others)
   f_file = "model.%s" % learner.ext()
   f_mod = pathfile(f_file, learner=learner, **others)
   return f_mod

def build(learner, debug=[], **others):
   f_in = os.path.join(trains.path(**others), "train.in") 
   model = name(learner=learner, **others)
   logger.info("+ building model %s" % model)
   f_mod = filename(learner=learner, **others)
   os.system('mkdir -p "%s"' % os.path.dirname(f_mod))
   enigmap.build(learner=learner, debug=debug, **others)
   if os.path.isfile(f_mod) and not "force" in debug:
      logger.debug("- skipped building model %s" % f_mod)
      return
   f_log = pathfile("train.log", learner=learner, **others)
   #learner.build(f_in, f_mod, f_log)
   p = Process(target=learner.build, args=(f_in,f_mod,f_log))
   p.start()
   p.join()

def accuracy(learner, f_in, f_mod):
   manager = Manager()
   ret = manager.dict()
   p = Process(target=learner.accuracy, args=(f_in,f_mod,ret))
   p.start()
   p.join()
   return ret["acc"]

