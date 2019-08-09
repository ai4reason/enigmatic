import lightgbm as lgb
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

from pyprove import log
from .learner import Learner

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

   def __repr__(self):
      args = ["%s=%s"%(x,self.params[x]) for x in self.params]
      args = ", ".join(args)
      return "%s(%s)" % (self.name(), args)

   def validate(self, bst, xs, ys):
      ipos = [i for (i,y) in enumerate(ys) if y == 1]
      ineg = [i for (i,y) in enumerate(ys) if y == 0]
      dpos = xs[ipos]
      dneg = xs[ineg]
      ppos = bst.predict(dpos)
      pneg = bst.predict(dneg)
      pos = len([x for x in ppos if x>=0.5])
      neg = len([x for x in pneg if x<0.5])
      return (pos, neg)

   def train(self, f_in, f_mod, f_log=None, f_stats=None, test_size=0):

      def posneg(data):
         labels = data.get_label()
         pos = len([x for x in labels if x == 1])
         neg = len([x for x in labels if x == 0])
         return (pos, neg)

      stats = {}

      (xall, yall) = load_svmlight_file(f_in)
      if test_size:
         (xtrain, xtest, ytrain, ytest) = train_test_split(xall, yall, test_size=test_size, random_state=43)
      else:
         (xtrain, ytrain) = (xall, yall)

      dtrain = lgb.Dataset(xtrain, ytrain)
      (pos, neg) = posneg(dtrain)
      self.params["scale_pos_weight"] = (float(neg)/float(pos))
      stats["train.pos.count"] = pos
      stats["train.neg.count"] = neg

      if test_size:
         dtest = lgb.Dataset(xtest, ytest, reference=dtrain)
         (pos, neg) = posneg(dtest)
         stats["test.pos.count"] = pos
         stats["test.neg.count"] = neg

      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain,dtest] if test_size else [dtrain])
      bst.save_model(f_mod)

      (pos, neg) = self.validate(bst, xtrain, ytrain)
      stats["train.pos.acc"] = pos / float(stats["train.pos.count"])
      stats["train.neg.acc"] = neg / float(stats["train.neg.count"])

      if test_size:
         (pos, neg) = self.validate(bst, xtest, ytest)
         stats["test.pos.acc"] = pos / float(stats["test.pos.count"])
         stats["test.neg.acc"] = neg / float(stats["test.neg.count"])

      if f_stats:
         json.dump(stats, file(f_stats,"w"))

