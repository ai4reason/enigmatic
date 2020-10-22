import re
import xgboost as xgb
from .learner import Learner
from sklearn.datasets import load_svmlight_file
from pyprove import log

DEFAULTS = {
   'max_depth': 9, 
   'eta': 0.2, 
   'objective': 'binary:logistic', 
   'num_round': 150
}

class XGBoost(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)
      Learner.__init__(self, self.params["num_round"])
      self.num_round = self.params["num_round"]
      del self.params["num_round"]

   def efun(self):
      return "EnigmaXgb"

   def ext(self):
      return "xgb"

   def name(self):
      return "XGBoost"
   
   def desc(self):
      return ("xgb-t%d-"%self.num_round) + ("d%(max_depth)s-e%(eta)s"%self.params)

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s, num_round=%s)" % (self.name(), args, self.num_round)

   def readlog(self, f_log):
      losses = re.findall(r'\[(\d*)\].*error:(\d*.\d*)', open(f_log).read())
      if not losses:
         self.stats["model.loss"] = "error"
         return
      losses = {int(x): float(y) for (x,y) in losses}
      last = max(losses)
      best = min(losses, key=lambda x: losses[x])
      self.stats["model.loss.last"] = "%f [iter %s]" % (losses[last], last)
      self.stats["model.loss.best"] = "%f [iter %s]" % (losses[best], best)

   def train(self, f_in, f_mod, iter_done=lambda x: x):
      dtrain = xgb.DMatrix(f_in)
      labels = dtrain.get_label()
      pos = float(len([x for x in labels if x == 1]))
      neg = float(len([x for x in labels if x == 0]))
      self.stats["train.count"] = log.humanint(len(labels))
      self.stats["train.count.pos"] = log.humanint(pos)
      self.stats["train.count.neg"] = log.humanint(neg)
      self.params["scale_pos_weight"] = (neg/pos)
      
      bst = xgb.train(self.params, dtrain, self.num_round, evals=[(dtrain, "training")])
      bst.save_model(f_mod)
      return bst

   def predict(self, f_in, f_mod):
      bst = xgb.Booster(model_file=f_mod)
      (xs, ys) = load_svmlight_file(f_in, zero_based=True)
      preds = bst.predict(xgb.DMatrix(xs), validate_features=False)
      return zip(preds, ys)


