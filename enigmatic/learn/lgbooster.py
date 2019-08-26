import lightgbm as lgb
import json
from pyprove import log, redirect
from .learner import Learner
from pyprove.bar import ProgressBar # FillingSquaresBar as Bar

DEFAULTS = {
   'max_depth': 9, 
   'learning_rate': 0.3, 
   'objective': 'binary', 
   'num_round': 200,
   'num_leaves': 300
}

class LightGBM(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)

   def efun(self):
      return "EnigmaLgb"

   def ext(self):
      return "lgb"

   def name(self):
      return "LightGBM"

   def desc(self):
      return "lgb-d%(max_depth)s-l%(num_leaves)s-e%(learning_rate)s" % self.params

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s)" % (self.name(), args)

   def train(self, f_in, f_mod, f_log=None, f_stats=None):
      stats = {}

      dtrain = lgb.Dataset(f_in)
      dtrain.construct()
      labels = dtrain.get_label()
      pos = float(len([x for x in labels if x == 1]))
      neg = float(len([x for x in labels if x == 0]))

      stats["train.pos.count"] = int(pos)
      stats["train.neg.count"] = int(neg)
      
      self.params["scale_pos_weight"] = (neg/pos)
      bar = ProgressBar("[3/3]", max=self.params["num_round"])
      bar.start()
      redir = redirect.start(f_log, bar)
      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain], callbacks=[lambda _: bar.next()])
      bar.finish() 
      redirect.finish(*redir)
      print()
      bst.save_model(f_mod)
      bst.free_dataset()
      bst.free_network()

      return stats

