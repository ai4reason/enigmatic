#!/usr/bin/env python3

import os, sys, io, logging
import optuna
import lightgbm as lgb
from pyprove import redirect, human
from pyprove.bar import ProgressBar
from enigmatic import trains
from enigmatic.learn import lgbooster

logger = logging.getLogger(__name__)

POS_ACC_WEIGHT = 2.0

def accuracy(bst, xs, ys):
   def getacc(pairs):
      if not pairs: return 0
      return sum([1 for (x,y) in pairs if int(x>0.5)==y]) / len(pairs)
   preds = bst.predict(xs)
   preds = list(zip(preds, ys))
   acc = getacc(preds)
   posacc = getacc([(x,y) for (x,y) in preds if y==1])
   negacc = getacc([(x,y) for (x,y) in preds if y==0])
   return (acc, posacc, negacc)

def check(trial, params, dtrain, testd,  d_tmp):
   f_mod = os.path.join(d_tmp, "model%04d.lgb" % trial.number)
   f_log = f_mod + ".log"
   ProgressBar.file = None
   bar = ProgressBar("[trial %d]"%trial.number, max=params["num_round"])
   redir = redirect.start(f_log, bar)

   try:
      if bar: bar.start()
      bst = lgb.train(
         params,
         dtrain, 
         valid_sets=[dtrain],
         callbacks=[lgb.log_evaluation(1)]+([lambda _: bar.next()] if bar else [])
      )
      bst.save_model(f_mod)
      if bar:
         bar.finish()
         bar.file.flush()

      (xs0, ys0) = testd
      acc = accuracy(bst, xs0, ys0)
      score = POS_ACC_WEIGHT*acc[1] + acc[2]
      trial.set_user_attr(key="model", value=f_mod)
      trial.set_user_attr(key="acc", value=acc)
      trial.set_user_attr(key="score", value=score)
      bst.free_dataset()
      bst.free_network()
   except Exception as e:
      redirect.finish(*redir)
      raise e

   redirect.finish(*redir)
   return score

def check_leaves(trial, params, **args):
   num_leaves = trial.suggest_int('num_leaves', 512, 32768, step=512)
   #num_leaves = trial.suggest_int('num_leaves', 256, 4096, step=8)
   params = dict(params, num_leaves=num_leaves)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- leaves trial %d: %s [num_leaves=%s]" % (trial.number, acc, params["num_leaves"]))
   #print("- leaves trial %d: test accuracy: %.2f (%.2f / %.2f) [num_leaves=%s]" % (
   #   (trial.number,)+trial.user_attrs["acc"]+(params["num_leaves"],)))
   return score

def check_bagging(trial, params, **args):
   bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
   bagging_fraction = min(trial.suggest_float("bagging_fraction", 0.4, 1.0+1e-12), 1.0)
   params = dict(params, bagging_freq=bagging_freq, bagging_fraction=bagging_fraction)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- bagging trial %d: %s [freq=%s, frac=%s]" % (trial.number, acc, params["bagging_freq"], params["bagging_fraction"]))
   #print("- bagging trial %d: test accuracy: %.2f (%.2f / %.2f) [freq=%s, frac=%s]" % (
   #   (trial.number,)+trial.user_attrs["acc"]+(params["bagging_freq"], params["bagging_fraction"])))
   return score

def check_min_data(trial, params, **args):
   min_data = trial.suggest_int("min_data", 5, 100)
   params = dict(params, min_data=min_data)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- min_data trial %d: %s [min_data=%s]" % (trial.number, acc, params["min_data"]))
   #print("- min_data trial %d: test accuracy: %.2f (%.2f / %.2f) [min_data=%s]" % (
   #   (trial.number,)+trial.user_attrs["acc"]+(params["min_data"],)))
   return score

def check_regular(trial, params, **args):
   lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0)
   lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0)
   params = dict(params, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- regular trial %d: %s [l1=%s, l2=%s]" % (trial.number, acc, params["lambda_l1"], params["lambda_l2"]))
   #print("- lambdas trial %d: test accuracy: %.2f (%.2f / %.2f) [lambda_l1=%s, lambda_l2=%s]" % (
   #   (trial.number,)+trial.user_attrs["acc"]+(params["lambda_l1"], params["lambda_l2"])))
   return score



def tune(check_fun, nick, iters, timeout, d_tmp, sampler=None, **args):
   d_tmp = os.path.join(d_tmp, nick)
   os.system('mkdir -p "%s"' % d_tmp)
   study = optuna.create_study(direction='maximize', sampler=sampler)
   objective = lambda trial: check_fun(trial, d_tmp=d_tmp, **args)
   study.optimize(objective, n_trials=iters, timeout=timeout)
   return study.best_trial

def tune_leaves(**args):
   return tune(check_leaves, "leaves", **args)

def tune_bagging(**args):
   return tune(check_bagging, "bagging", **args)

def tune_min_data(**args):
   #name = "min_data"
   #values = [5, 10, 25, 50, 100]
   #sampler = optuna.samplers.GridSampler({name: values})
   sampler = None
   return tune(check_min_data, "min_data", sampler=sampler, **args)

def tune_regular(**args):
   return tune(check_regular, "regular", **args)

PHASES = {
   "l": tune_leaves,
   "b": tune_bagging,
   "r": tune_regular,
   "m": tune_min_data,
}

def train(f_train, f_test, d_tmp="optuna-tmp", phases="lbmr", iters=100, timeout=None, inits={}):
   (xs, ys) = trains.load(f_train)
   dtrain = lgb.Dataset(xs, label=ys)
   testd = trains.load(f_test)
   os.system('mkdir -p "%s"' % d_tmp)
   redirect.module("optuna", os.path.join(d_tmp, "optuna.log"))
   
   params = dict(lgbooster.DEFAULTS)
   params.update(inits)
   pos = sum(ys)
   neg = len(ys) - pos
   #params["scale_pos_weight"] = neg / pos
   params["is_unbalance"] = neg != pos 
   if "m" in phases:
      params["feature_pre_filter"] = False
   timeout = timeout / len(phases) if timeout else None
   if iters:
      iters += iters % len(phases)
      iters = iters // len(phases)
   args = dict(dtrain=dtrain, testd=testd, d_tmp=d_tmp, iters=iters, timeout=timeout)

   best = None
   for phase in phases:
      trial = PHASES[phase](params=params, **args)
      if (not best) or (trial.user_attrs["score"] > best.user_attrs["score"]):
         best = trial
         params.update(best.params)
   
   return (best, params, pos, neg)

def lgbtune(f_train, f_test, d_tmp="optuna-tmp", phases="lbmr", iters=None, timeout=3600.0, inits={}):
   logger.setLevel(logging.DEBUG)
   logger.addHandler(logging.StreamHandler(io.TextIOWrapper(os.fdopen(sys.stdout.fileno(), "wb"))))
   (best, params, _, _) = train(f_train, f_test, d_tmp, phases, iters, timeout, inits)
   logger.info("")
   logger.info("Best model params: %s" % str(params))
   logger.info("Best model accuracy: %s" % human.humanacc(best.user_attrs["acc"]))
   logger.info("Best model file: %s" % best.user_attrs["model"])

#autotune("train.in", "test.in", "lgbtune", iters=None, timeout=60)

