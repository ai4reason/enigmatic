import re, os
import logging
import lightgbm as lgb
from .learner import Learner
from pyprove import log
from .. import trains 

logger = logging.getLogger(__name__)

DEFAULTS = {
   'learning_rate': 0.15, 
   'objective': 'binary', 
   'num_round': 150,
   'max_depth': 9, 
   'num_leaves': 300,
   # default values from the docs:
   'min_data': 20,
   'max_bin': 255,
   'feature_fraction': 1.0,
   'bagging_fraction': 1.0,
   'bagging_freq': 0,
   'lambda_l1': 0.0,
   'lambda_l2': 0.0,
}

# non-default values of these will influence model name
RELEVANT = {
   'min_data': 'min',
   'max_bin': 'max',
   'feature_fraction': 'ff',
   'bagging_fraction': 'bf',
   'bagging_freq': 'sf',
   'lambda_l1': '1l',
   'lambda_l2': '2l',
}

class LightGBM(Learner):

   def __init__(self, **args):
      self.params = dict(DEFAULTS)
      self.params.update(args)
      Learner.__init__(self, self.params["num_round"])

   def efun(self):
      return "EnigmaticLgb"

   def ext(self):
      return "lgb"

   def name(self):
      return "LightGBM"

   def desc(self):
      d = "lgb-t%(num_round)s-d%(max_depth)s-l%(num_leaves)s-e%(learning_rate)s" % self.params
      for p in RELEVANT:
         if self.params[p] != DEFAULTS[p]:
            d += "-%s%s" % (RELEVANT[p], self.params[p])
      #if self.params["min_data"] != DEFAULTS["min_data"]:
      #   d += "-min%(min_data)d" % self.params
      #if self.params["max_bin"] != DEFAULTS["max_bin"]:
      #   d += "-max%(max_bin)d" % self.params
      return d

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s)" % (self.name(), args)

   def readlog(self, f_log):
      if not os.path.isfile(f_log):
         return
      losses = re.findall(r'\[(\d*)\].*logloss: (\d*.\d*)', open(f_log).read())
      if not losses:
         self.stats["model.loss"] = "error"
         return
      losses = {int(x): float(y) for (x,y) in losses}
      last = max(losses)
      best = min(losses, key=lambda x: losses[x])
      self.stats["model.last.loss"] = [losses[last], last]
      self.stats["model.best.loss"] = [losses[best], best]

   def train(self, f_in, f_mod=None, init_model=None, handlers=None):
      (atstart, atiter, atfinish) = handlers if handlers else (None,None,None)
      (xs, ys) = trains.load(f_in)
      dtrain = lgb.Dataset(xs, label=ys, free_raw_data=(init_model is None))
      #dtrain.construct()
      pos = sum(ys)
      neg = len(ys) - pos
      self.stats["train.counts"] = (len(ys), int(pos), int(neg))
      self.params["scale_pos_weight"] = (neg/pos)
      #self.params["is_unbalance"] = True

      callbacks = [lambda _: atiter(), lgb.log_evaluation(1)] if atiter else None
      if atstart: atstart()
      #eta = self.params["learning_rate"]
      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain], init_model=init_model, callbacks=callbacks) #, learning_rates=lambda iter: 0.1*(0.95**iter))
      if atfinish: atfinish()
      if f_mod:
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

