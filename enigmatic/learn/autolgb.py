import re, os, shutil
import logging, json
import lightgbm as lgb
from .learner import Learner
from .lgbooster import LightGBM
from pyprove import log
from .. import trains, lgbtune

logger = logging.getLogger(__name__)

DEFAULTS = {
   "iters": 30,
   "timeout": None,
   "init_params": None,
   "phases": "l:b:m",
}

class AutoLgb(LightGBM):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)
      Learner.__init__(self)

   def efun(self):
      return "EnigmaticLgb"

   def ext(self):
      return "lgb"

   def name(self):
      return "AutoLgb"

   def desc(self):
      def add(param, short):
         return ("-%s%s" % (short, self.params[param])) if self.params[param] else ""
      d = "autolgb" + add("iters", "i") + add("timeout", "t") + add("phases", "")
      return d

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s)" % (self.name(), args)

   def train(self, f_in, f_mod=None, init_model=None, handlers=None):
      raise NotImplementedError

   def build(self, f_in, f_mod, f_log, options=[], init_model=None, f_test=None):
      logger.info("- building model %s" % f_mod)
      logger.debug(log.data("- learning parameters:", self.params))
      f_test = f_test if f_test else f_in
      d_tmp = os.path.join(os.path.dirname(f_mod), "optuna-tmp")
      usebar = "headless" not in options
      (_, acc, f_m, dur, params, pos, neg) = lgbtune.train(f_in, f_test, d_tmp, usebar=usebar, **self.params)
      logger.debug("- best model after tuning is %s" % f_m)
      shutil.copy(f_m, f_mod)
      shutil.copy(f_m+".log", f_log)
      shutil.copy(os.path.join(d_tmp, "optuna.log"), "%s-optuna.log" % f_log)
      os.system('rm -fr "%s"' % d_tmp)
      self.readlog(f_log)
      self.stats["model.train.time"] = dur
      self.stats["train.size"] = trains.size(f_in)
      self.stats["train.format"] = trains.format(f_in)
      self.stats["train.counts"] = (pos+neg, pos, neg)
      self.stats["model.size"] = os.path.getsize(f_mod)
      #self.stats["test.acc"] = acc
      with open("%s-stats.json"%f_log,"w") as f: 
         json.dump(self.stats, f, indent=3, sort_keys=True)
      with open("%s-params.json"%f_log,"w") as f: 
         json.dump(params, f, indent=3, sort_keys=True)
      with open("%s-tuning.json"%f_log,"w") as f: 
         json.dump(self.params, f, indent=3, sort_keys=True)

