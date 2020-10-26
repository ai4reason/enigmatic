import re
import logging
import lightgbm as lgb
from .learner import Learner
from pyprove import log
from .. import trains 

logger = logging.getLogger(__name__)

DEFAULTS = {
   'max_depth': 9, 
   'learning_rate': 0.2, 
   'objective': 'binary', 
   'num_round': 150,
   'num_leaves': 300
}

class LightGBM(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)
      Learner.__init__(self, self.params["num_round"])

   def efun(self):
      return "EnigmaLgb"

   def ext(self):
      return "lgb"

   def name(self):
      return "LightGBM"

   def desc(self):
      return "lgb-t%(num_round)s-d%(max_depth)s-l%(num_leaves)s-e%(learning_rate).2f" % self.params

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
      self.stats["model.last.loss"] = [losses[last], last]
      self.stats["model.best.loss"] = [losses[best], best]

   def train(self, f_in, f_mod, atstart=None, atiter=None, atfinish=None):
      logger.debug("- loading training data %s" % f_in)
      (xs, ys) = trains.load(f_in)
      dtrain = lgb.Dataset(xs, label=ys)
      dtrain.construct()
      pos = sum(ys)
      neg = len(ys) - pos
      self.stats["train.count"] = len(ys)
      self.stats["train.pos.count"] = int(pos)
      self.stats["train.neg.count"] = int(neg)
      self.params["scale_pos_weight"] = (neg/pos)

      callbacks = [lambda _: atiter()] if atiter else None
      logger.debug("- building lgb model %s" % f_mod)
      if atstart: atstart()
      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain], callbacks=callbacks)
      if atfinish: atfinish()
      logger.debug("- saving model %s" % f_mod)
      bst.save_model(f_mod)
      bst.free_dataset()
      bst.free_network()
      return bst

   def predict(self, f_in, f_mod):
      bst = lgb.Booster(model_file=f_mod)
      logger.debug("- loading training data %s" % f_in)
      (xs, ys) = trains.load(f_in)
      logger.debug("- predicting with lgb model %s" % f_mod)
      preds = bst.predict(xs, predict_disable_shape_check=True)
      return zip(preds, ys)

