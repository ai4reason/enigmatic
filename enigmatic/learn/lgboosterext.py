import subprocess
import json
import os.path
from pyprove import log
from .lgbooster import LightGBM

DEFAULTS = {
   'max_depth': 9, 
   'learning_rate': 0.3, 
   'objective': 'binary', 
   'num_round': 200,
   'num_leaves': 300
}

BOOSTER_BIN = "enigmatic-lgbooster.py"

class LightGBMExt(LightGBM):

   def train(self, f_in, f_mod, f_log=None, f_stats=None):
      out = open(f_log, "w", buffering=1)
      subprocess.call([BOOSTER_BIN, f_in, f_mod, f_stats, json.dumps(self.params)],
         stdout=out, stderr=subprocess.STDOUT)
      out.close()

   def rounds(self):
      return self.params["num_round"]

   def current(self, f_log):
      cur = 0
      if not os.path.isfile(f_log):
         return 0
      with open(f_log) as file:
         for line in file:
            if line[0] == '[':
               str = line.lstrip("[").split("]")[0]
               if str.isdigit():
                  cur = max(cur, int(str))
      return cur


