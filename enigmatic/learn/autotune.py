import os
import time
import json
from multiprocessing import Process
import sherpa

from .lgbooster import LightGBM
from enigmatic import models, protos
from pyprove import log, expres, eprover

PARAMETERS = [
   sherpa.Continuous(name="learning_rate", range=[0.1,0.3]),
   sherpa.Discrete(name="max_depth", range=[10,100]),
   sherpa.Discrete(name="num_leaves", range=[10,10000])
]

DEFAULTS = {
   "learning_rate": 0.15,
   "max_depth": 50,
   "num_leaves": 40
}

class AutoTuneLgb(LightGBM):

   def __init__(self, tunetime, crossval):
      self.params = {"tunetime": tunetime}
      self.stats = None
      self.tunetime = tunetime
      self.crossval = crossval

   def name(self):
      return "AutoTuneLgb"
   
   def desc(self):
      return "AutoTuneLgb-t%s" % self.tunetime

   def observe(self, param, model, ref, **crossval):
      print("observing")
      param["num_leaves"] = int(param["num_leaves"])
      param["max_depth"] = int(param["max_depth"])
      print(param)
      learner = LightGBM(**param)
      learner.nobar()
      model0 = "%s/tuning/%s" % (model, learner.desc())
      f_in = models.path(model, "train.in") # shared by all models
      f_mod = models.path(model0, "model.%s" % learner.ext())
      f_stats = models.path(model0, "train.stats")
      f_log = models.path(model0, "train.log")
      d_model0 = models.path(model0)
      os.system("mkdir -p %s" % d_model0)
      os.system("cp %s/enigma.map %s" % (models.path(model), d_model0))

      print("prepared",model0,f_in,f_mod,f_log,f_stats)
      if not os.path.isfile(f_mod):
         p = Process(target=learner.build, args=(model0,f_in,f_mod,f_log,f_stats))
         p.start()
         p.join()

      print("finished")
      pid = protos.coop(ref, model0, noinit=True, efun=learner.efun())
      result = expres.benchmarks.eval(pids=[pid], **crossval)

      sols = [r for r in result if eprover.result.solved(result[r])]
      o_solved = len(sols)
      o_proc = sum([result[s]["PROCESSED"] for s in sols]) 
      o_gen = sum([result[s]["GENERATED"] for s in sols]) 
      obj = (len(result)-o_solved) + o_proc/1000000.0
      #obj = o_proc + (len(result)-o_solved)*1000000
      context = json.load(open(f_stats))
      context = {
         "time": context["model.train.seconds"], 
         "solved": o_solved, 
         "proc": o_proc / o_solved,
         "gen": o_gen / o_solved,
      }

      return (obj, context, pid, result, learner, model0)

   def build(self, model, f_in=None, f_mod=None, f_log=True, f_stats=True):
      f_in = models.path(model, "train.in") if not f_in else f_in
      f_mod = models.path(model, "model.%s" % self.ext()) if not f_mod else f_mod
      if f_log is True: f_log = models.path(model, "train.log")
      f_stats = False # TODO

      algorithm = sherpa.algorithms.GPyOpt()
      study = sherpa.Study(
         parameters=PARAMETERS,
         algorithm=algorithm,
         lower_is_better=True
      )
      best_obj = None
      results = {}
      pids = []

      ##log.disable()
      start = time.time()
      for trial in study:
         print(trial)
         (obj, context, pid, result, learner, model0) = self.observe(trial.parameters, model, **self.crossval)
         print("observed.")
         if (not best_obj) or obj < best_obj:
            best_obj = obj
            best_learner = learner
            best_model = model0
         results.update(result)
         pids.append(pid)
         #redirect.finish(*redir)
         ##log.enable()
         log.text("| %s = %s" % (learner.desc(), obj))
         ##log.disable()
         #redir = redirect.start(f_log)
         study.add_observation(trial=trial, objective=obj, context=context)
         study.finalize(trial)
         if time.time() - start > self.tunetime:
            break
         print("done.")
      
      #redirect.finish(*redir)
      ##log.enable()
      os.system("cp %s/model.%s %s" % (models.path(best_model), best_learner.ext(), models.path(model)))
      expres.dump.solved(pids=pids, results=results, **self.crossval)

