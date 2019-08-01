import subprocess
import json
from pyprove import log
from .learner import Learner

DEFAULTS = {
   'max_depth': 9, 
   'learning_rate': 0.3, 
   'objective': 'binary', 
   'num_round': 200,
   'num_leaves': 300
}

BOOSTER_BIN = "enigmatic-lgbooster.py"

class LightGBMExt(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)

   def efun(self):
      return "EnigmaLgb"

   def ext(self):
      return "lgb"

   def name(self):
      return "LightGBM"

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s)" % (self.name(), args)

   def train(self, f_in, f_mod, f_log=None, f_stats=None):
      out = file(f_log, "a")
      subprocess.call([BOOSTER_BIN, f_in, f_mod, f_stats, json.dumps(self.params)],
         stdout=out, stderr=subprocess.STDOUT)
      out.close()

