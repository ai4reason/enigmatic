import os, shutil
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

def filename(learner, part=None, **others):
   model = name(learner=learner, **others)
   f_file = "model.%s" % learner.ext()
   if part is not None:
      f_file = os.path.join("part%03d"%part, f_file)
   f_mod = pathfile(f_file, learner=learner, **others)
   return f_mod

def batchbuilds(f_in, f_mod, learner, options, **others):
   n = 0
   f_part = None
   f_log = None
   def nextbatch():
      nonlocal n, f_part, f_log
      n += 1
      f_part = trains.filename(part=n, **others)
      f_log = filename(learner=learner, part=n, **others) + ".log"
   nextbatch()
   while trains.exist(f_part) or os.path.isfile(f_part):
      #learner.params["learning_rate"] = learner.params["learning_rate"]*0.1
      logger.info("- next batch build with %s" % f_part)
      os.system('mkdir -p "%s"' % os.path.dirname(f_log))
      f_piece = filename(learner=learner, part=n-1, **others)
      shutil.copy(f_mod, f_piece)
      p = Process(target=learner.build, args=(f_part,f_mod,f_log,options,f_mod))
      p.start()
      p.join()
      nextbatch()

def build(learner, f_in=None, debug=[], options=[], **others):
   f_in = f_in if f_in else trains.filename(part=0, **others)
   model = name(learner=learner, **others)
   logger.info("+ building model %s" % model)
   logger.info("- building with %s" % f_in)
   f_mod = filename(learner=learner, **others)
   os.system('mkdir -p "%s"' % os.path.dirname(f_mod))
   enigmap.build(learner=learner, debug=debug, **others)
   #learner.params["num_feature"] = enigmap.load(learner=learner, **others)["count"]
   new = protos.build(model, learner=learner, debug=debug, **others)
   if os.path.isfile(f_mod) and not "force" in debug:
      logger.debug("- skipped building model %s" % f_mod)
      return new

   f_log = filename(learner=learner, part=0, **others) + ".log"
   os.system('mkdir -p "%s"' % os.path.dirname(f_log))
   p = Process(target=learner.build, args=(f_in,f_mod,f_log,options,None))
   p.start()
   p.join()
   batchbuilds(f_in, f_mod, learner=learner, debug=debug, options=options, **others)
   return new

def loop(pids, results, nick, **others):
   others["dataname"] += "/" + nick
   trains.build(pids=pids, **others)
   newp = build(pids=pids, **others)
   newr = expres.benchmarks.eval(pids=newp, **others)
   pids.extend(newp)
   results.update(newr)

def accuracy(learner, f_in, f_mod):
   manager = Manager()
   ret = manager.dict()
   p = Process(target=learner.accuracy, args=(f_in,f_mod,ret))
   p.start()
   p.join()
   return ret["acc"]

