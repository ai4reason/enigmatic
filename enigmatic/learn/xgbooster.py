import xgboost as xgb
from .learner import Learner

DEFAULTS = {
   'max_depth': 9, 
   'eta': 0.3, 
   'objective': 'binary:logistic', 
   'num_round': 200
}

class XGBoost(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)
      self.num_round = self.params["num_round"]
      del self.params["num_round"]

   def efun(self):
      return "EnigmaXgb"

   def ext(self):
      return "xgb"

   def name(self):
      return "XGBoost"

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s, num_round=%s)" % (self.name(), args, self.num_round)

   def train(self, f_in, f_mod, f_log=None, f_stats=None):
      dtrain = xgb.DMatrix(f_in)
      labels = dtrain.get_label()
      pos = float(len([x for x in labels if x == 1]))
      neg = float(len([x for x in labels if x == 0]))

      self.params["scale_pos_weight"] = (neg/pos)
      bst = xgb.train(self.params, dtrain, self.num_round, evals=[(dtrain, "training")])
      bst.save_model(f_mod)
      return bst

