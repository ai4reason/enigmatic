import re, os
import logging
import xgboost as xgb
from .learner import Learner
from pyprove import log
from .. import trains 

logger = logging.getLogger(__name__)

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
      return "EnigmaticXgb"

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
      if not os.path.isfile(f_log):
         return
      losses = re.findall(r'\[(\d*)\].*error:(\d*.\d*)', open(f_log).read())
      if not losses:
         self.stats["model.loss"] = "error"
         return
      losses = {int(x): float(y) for (x,y) in losses}
      last = max(losses)
      best = min(losses, key=lambda x: losses[x])
      self.stats["model.last.loss"] = [losses[last], last]
      self.stats["model.best.loss"] = [losses[best], best]

   def train(self, f_in, f_mod, init_model=None, handlers=None):
      (atstart, atiter, atfinish) = handlers if handlers else (None,None,None)
      (xs, ys) = trains.load(f_in)
      dtrain = xgb.DMatrix(xs, label=ys)
      pos = sum(ys)
      neg = len(ys) - pos
      self.stats["train.count"] = len(ys)
      self.stats["train.pos.count"] = int(pos)
      self.stats["train.neg.count"] = int(neg)
      self.params["scale_pos_weight"] = (neg/pos)
      
      callbacks = [lambda _: atiter()] if atiter else None
      if atstart: atstart()
      bst = xgb.train(self.params, dtrain, self.num_round, evals=[(dtrain, "training")], xgb_model=init_model, callbacks=callbacks)
      if atfinish: atfinish()
      bst.save_model(f_mod)
      return bst

   def predict(self, f_in, f_mod):
      bst = xgb.Booster(model_file=f_mod)
      logger.debug("- loading training data %s" % f_in)
      (xs, ys) = trains.load(f_in)
      logger.debug("- predicting with xgb model %s" % f_mod)
      preds = bst.predict(xgb.DMatrix(xs), validate_features=False)
      return zip(preds, ys)


