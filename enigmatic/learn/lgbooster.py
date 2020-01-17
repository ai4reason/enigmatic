import re
import lightgbm as lgb
from .learner import Learner

DEFAULTS = {
   'max_depth': 9, 
   'learning_rate': 0.2, 
   'objective': 'binary', 
   'num_round': 150,
   'num_leaves': 300
}

class LightGBM(Learner):

   def __init__(self, **args):
      Learner.__init__(self)
      self.params = dict(DEFAULTS)
      self.params.update(args)
      self.num_round = self.params["num_round"]

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

   def readlog(self, f_log):
      losses = re.findall(r'\[(\d*)\].*logloss: (\d*.\d*)', open(f_log).read())
      if not losses:
         self.stats["model.loss"] = "error"
         return
      losses = {int(x): float(y) for (x,y) in losses}
      last = max(losses)
      best = min(losses, key=lambda x: losses[x])
      self.stats["model.loss.last"] = "%f [%s]" % (losses[last], last)
      self.stats["model.loss.best"] = "%f [%s]" % (losses[best], best)

   def train(self, f_in, f_mod, iter_done=lambda x: x):
      dtrain = lgb.Dataset(f_in)
      dtrain.construct()
      labels = dtrain.get_label()
      pos = float(len([x for x in labels if x == 1]))
      neg = float(len([x for x in labels if x == 0]))
      self.stats["train.count"] = len(labels)
      self.stats["train.count.pos"] = int(pos)
      self.stats["train.count.neg"] = int(neg)
      self.params["scale_pos_weight"] = (neg/pos)

      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain], callbacks=[lambda _: iter_done()])
      bst.save_model(f_mod)
      bst.free_dataset()
      bst.free_network()
      return bst

