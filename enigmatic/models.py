import os, shutil
import json
from multiprocessing import Process, Manager
import logging

from . import trains, protos, enigmap
from pyprove import expres, log

DEFAULT_NAME = "Enigma"
DEFAULT_DIR = os.getenv("ENIGMA_ROOT", DEFAULT_NAME)

logger = logging.getLogger(__name__)

def name(learner, **others):
   others = dict(others, learner=learner)
   return os.path.join(trains.name(**others), learner.desc())

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

def build(learner, f_in=None, split=False, debug=[], options=[], **others):
   others = dict(others, learner=learner, split=split, debug=debug, options=options)
   f_in = f_in if f_in else trains.filename(part=0, **others)
   model = name(**others)
   logger.info("+ building model %s" % model)
   logger.info("- building with %s" % f_in)
   f_mod = filename(**others)
   os.system('mkdir -p "%s"' % os.path.dirname(f_mod))
   enigmap.build(**others)
   #learner.params["num_feature"] = enigmap.load(learner=learner, **others)["count"]
   new = protos.build(model, **others)
   if os.path.isfile(f_mod) and not "force" in debug:
      logger.debug("- skipped building model %s" % f_mod)
      return new

   f_log = filename(part=0, **others) + ".log"
   os.system('mkdir -p "%s"' % os.path.dirname(f_log))
   p = Process(target=learner.build, args=(f_in,f_mod,f_log,options,None))
   p.start()
   p.join()

   if "acc" in debug:
      accuracy(learner, f_in, f_mod)
      if split:
         f_test = trains.filename(f_name="test.in", part=0, **others)
         if os.path.isfile(f_test) or trains.exist(f_test):
            accuracy(learner, f_test, f_mod)

   batchbuilds(f_in, f_mod, **others)
   return new

def loop(pids, results, nick, refs, **others):
   others["dataname"] += "/" + nick
   trains.build(pids=pids, refs=refs, **others)
   newp = build(pids=pids, refs=refs, **others)
   if refs and len(refs) > 1:
      # loop with coop strategies only when two or more refs are provided
      newp = [p for p in newp if "coop" in p]
   newr = expres.benchmarks.eval(pids=newp, refs=refs, **others)
   pids.extend(newp)
   results.update(newr)

def accuracy(learner, f_in, f_mod):
   manager = Manager()
   ret = manager.dict()
   p = Process(target=learner.accuracy, args=(f_in,f_mod,ret))
   p.start()
   p.join()
   return ret["acc"]

