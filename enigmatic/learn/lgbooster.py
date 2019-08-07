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

   def validate(self, bst, data):
      labels = data.get_label()
      ipos = [i for (i,y) in enumerate(labels) if y == 1]
      ineg = [i for (i,y) in enumerate(labels) if y == 0]
      dpos = data.subset(ipos)
      dneg = data.subset(ineg)
      ppos = bst.predict(dpos)
      pneg = bst.predict(dneg)
      pos = len([x for x in ppos if x>0.5])
      neg = len([x for x in pneg if x<0.5])
      return (pos, neg)

   def train(self, f_in, f_mod, f_log=None, f_stats=None):

      def posneg(data):
         labels = data.get_label()
         pos = len([x for x in labels if x == 1])
         neg = len([x for x in labels if x == 0])
         return (pos, neg)

      stats = {}

      (xall, yall) = load_svmlight_file(f_in)
      (xtrain, xtest, ytrain, ytest) = train_test_split(xall, yall, test_size=0.1, random_state=43)

      print type(xtrain), type(ytest)


      dtrain = lgb.Dataset(xtrain, ytrain)
      dtest = lgb.Dataset(xtest, ytest, reference=dtrain)
       
      (pos, neg) = posneg(dtrain)
      self.params["scale_pos_weight"] = (float(neg)/float(pos))
      stats["train.pos.count"] = pos
      stats["train.neg.count"] = neg

      (pos, neg) = posneg(dtest)
      stats["test.pos.count"] = pos
      stats["test.neg.count"] = neg

      bst = lgb.train(self.params, dtrain, valid_sets=[dtrain,dtest])
      bst.save_model(f_mod)

      print self.validate(bst, dtrain)


      if f_stats:
         json.dump(stats, file(f_stats,"w"))

