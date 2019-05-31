import sys
import numpy
import xgboost as xgb
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


def train(f_in, f_out, log=None, xgb_params=None):
   if log:
      log.write("\nTraining XGBoost model (%s):\n\n" % f_in)
      oldout = sys.stdout
      sys.stdout = log

   dtrain = xgb.DMatrix(f_in)

   labels = dtrain.get_label()
   pos = float(len([x for x in labels if x == 1]))
   neg = float(len([x for x in labels if x == 0]))

   param = {
      'max_depth': 9, 
      'eta': 0.3, 
      'objective': 'binary:logistic', 
      'scale_pos_weight': (neg/pos),
   }
   if xgb_params:
      param.update(xgb_params)
   if "num_round" in param:
      num_round = param["num_round"]
      del param["num_round"]
   else:
      num_round = 200

   bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, "training")])
   bst.save_model(f_out)

   if log:
      sys.stdout = oldout

   return bst

def train_old(f_in, f_out, log=None):
   if log:
      log.write("\nTraining XGBoost model (%s):\n\n" % f_in)
      oldout = sys.stdout
      sys.stdout = log
   
   X, y = load_svmlight_file(f_in)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   #X_all, X_none, y_all, y_none = train_test_split(X, y, test_size=0.0, random_state=42)

   c_pos = len([y for y in y_test if y == 1])
   c_neg = len([y for y in y_test if y == 0])
   cols = X_train.shape[1]
   X_pos = lil_matrix((c_pos, cols), dtype=numpy.float64)
   X_neg = lil_matrix((c_neg, cols), dtype=numpy.float64)

   i = 0
   j = 0
   for (row, label) in zip(X_test, y_test):
      if label == 1:
         X_pos[i] = row
         i += 1
      elif label == 0:
         X_neg[j] = row
         j += 1

   y_pos = numpy.empty(c_pos)
   y_pos.fill(1)
   y_neg = numpy.empty(c_neg)
   y_neg.fill(0)

   dtest = xgb.DMatrix(X_test, label=y_test)
   dpos = xgb.DMatrix(X_pos, label=y_pos)
   dneg = xgb.DMatrix(X_neg, label=y_neg)

   dtrain = xgb.DMatrix(X_train, label=y_train)
   #dtrain = xgb.DMatrix(X_all, label=y_all)
   param = {'max_depth':9, 'eta':0.3, 'objective':'binary:logistic'}
   num_round = 6
   #num_round = 200

   bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, "training"), (dtest,"testing"), (dpos, "pos"), (dneg, "neg")])
   bst.save_model(f_out)

   if log:
      sys.stdout = oldout

   return bst

