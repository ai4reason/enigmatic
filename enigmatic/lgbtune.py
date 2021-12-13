#!/usr/bin/env python3

import os, sys, io, logging, time, math
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

def model(params, dtrain, testd, f_mod, barmsg="lgb"):
   f_log = f_mod + ".log"
   if barmsg:
      ProgressBar.file = None
      bar = ProgressBar(barmsg, max=params["num_round"]) if barmsg else None
   else:
      bar = None
   logger.debug("- building model %s" % f_mod)
   redir = redirect.start(f_log, bar)
   try:
      if bar: bar.start()
      begin = time.time()
      bst = lgb.train(
         params,
         dtrain, 
         valid_sets=[dtrain],
         callbacks=[lgb.log_evaluation(1)]+([lambda _: bar.next()] if bar else [])
      )
      end = time.time()
      bst.save_model(f_mod)
      if bar:
         bar.finish()
         bar.file.flush()

      (xs0, ys0) = testd
      acc = accuracy(bst, xs0, ys0)
      bst.free_dataset()
      bst.free_network()
   except Exception as e:
      redirect.finish(*redir)
      raise e
   redirect.finish(*redir)
   score = POS_ACC_WEIGHT*acc[1] + acc[2]
   return (score, acc, end-begin)
   

def check(trial, params, dtrain, testd, d_tmp, usebar, **args):
   f_mod = os.path.join(d_tmp, "model%04d.lgb" % trial.number)
   barmsg = ("[trial %d]" % trial.number) if usebar else None
   (score, acc, dur) = model(params, dtrain, testd, f_mod, barmsg)
   trial.set_user_attr(key="model", value=f_mod)
   trial.set_user_attr(key="score", value=score)
   trial.set_user_attr(key="acc", value=acc)
   trial.set_user_attr(key="time", value=dur)
   return score

def check_leaves(trial, params, min_leaves, max_leaves, **args):
   #num_leaves_base = trial.suggest_int('num_leaves_base', 16, 31)
   #num_leaves = round(2**(num_leaves_base/2))
   num_leaves = trial.suggest_int('num_leaves', min_leaves, max_leaves)
   params = dict(params, num_leaves=num_leaves)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- leaves trial %d: %s [num_leaves=%s]" % (trial.number, acc, params["num_leaves"]))
   return score

def check_bagging(trial, params, **args):
   bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
   bagging_fraction = min(trial.suggest_float("bagging_fraction", 0.4, 1.0+1e-12), 1.0)
   params = dict(params, bagging_freq=bagging_freq, bagging_fraction=bagging_fraction)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- bagging trial %d: %s [freq=%s, frac=%s]" % (trial.number, acc, params["bagging_freq"], params["bagging_fraction"]))
   return score

def check_min_data(trial, params, **args):
   min_data = trial.suggest_int("min_data", 5, 100)
   params = dict(params, min_data=min_data)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- min_data trial %d: %s [min_data=%s]" % (trial.number, acc, params["min_data"]))
   return score

def check_regular(trial, params, **args):
   lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0)
   lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0)
   params = dict(params, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
   score = check(trial, params, **args)
   acc = human.humanacc(trial.user_attrs["acc"])
   logger.debug("- regular trial %d: %s [l1=%s, l2=%s]" % (trial.number, acc, params["lambda_l1"], params["lambda_l2"]))
   return score

def tune(check_fun, nick, iters, timeout, d_tmp, sampler=None, **args):
   d_tmp = os.path.join(d_tmp, nick)
   os.system('mkdir -p "%s"' % d_tmp)
   study = optuna.create_study(direction='maximize', sampler=sampler)
   objective = lambda trial: check_fun(trial, d_tmp=d_tmp, **args)
   study.optimize(objective, n_trials=iters, timeout=timeout)
   return study.best_trial

def tune_leaves(min_leaves, max_leaves, **args):
   args = dict(args, min_leaves=min_leaves, max_leaves=max_leaves)
   min_base = round(2*math.log2(min_leaves))
   max_base = round(2*math.log2(max_leaves))
   values = list([round(2**(n/2)) for n in range(min_base, max_base+1)])
   sampler = optuna.samplers.GridSampler({"num_leaves": values})
   return tune(check_leaves, "leaves", sampler=sampler, **args)

def tune_bagging(**args):
   return tune(check_bagging, "bagging", **args)

def tune_min_data(**args):
   values = [5, 10, 25, 50, 100]
   sampler = optuna.samplers.GridSampler({"min_data": values})
   return tune(check_min_data, "min_data", sampler=sampler, **args)

def tune_regular(**args):
   return tune(check_regular, "regular", **args)

PHASES = {
   "l": tune_leaves,
   "b": tune_bagging,
   "r": tune_regular,
   "m": tune_min_data,
}

def train(
   f_train,
   f_test, 
   d_tmp="optuna-tmp", 
   phases="l:b:m:r", 
   iters=100, 
   timeout=None, 
   init_params=None, 
   usebar=True, 
   min_leaves=256, 
   max_leaves=32768
):
   (xs, ys) = trains.load(f_train)
   dtrain = lgb.Dataset(xs, label=ys)
   testd = trains.load(f_test) if f_test != f_train else (xs, ys)
   os.system('mkdir -p "%s"' % d_tmp)
   redirect.module("optuna", os.path.join(d_tmp, "optuna.log"))
   
   params = dict(lgbooster.DEFAULTS)
   if init_params: params.update(init_params)
   pos = sum(ys)
   neg = len(ys) - pos
   #params["scale_pos_weight"] = neg / pos
   params["is_unbalance"] = "true" if neg != pos else "false"
   phases = phases.split(":")
   if "m" in phases:
      params["feature_pre_filter"] = "false" 
   timeout = timeout / len(phases) if timeout else None
   iters = iters // len(phases) if iters else None
   args = dict(
      dtrain=dtrain, 
      testd=testd, 
      d_tmp=d_tmp, 
      iters=iters, 
      timeout=timeout, 
      usebar=usebar, 
      min_leaves=min_leaves, 
      max_leaves=max_leaves
   )

   if init_params is not None:
      f_mod = os.path.join(d_tmp, "init.lgb")
      (score, acc, dur) = model(params, dtrain, testd, f_mod, "[init]" if usebar else None)
      best = (score, acc, f_mod, dur)
      logger.debug("- initial model: %s" % human.humanacc(acc)) 
   else:
      best = (-1, None, None)

   for phase in phases:
      trial = PHASES[phase](params=params, **args)
      if trial.user_attrs["score"] > best[0]:
         best = tuple(trial.user_attrs[x] for x in ["score", "acc", "model", "time"])
         params.update(trial.params)
         #if "num_leaves_base" in params:
         #   params["num_leaves"] = round(2**(params["num_leaves_base"]/2))
         #   del params["num_leaves_base"]
   
   return best + (params, pos, neg)

def lgbtune(f_train, f_test, d_tmp="optuna-tmp", phases="lbmr", iters=None, timeout=3600.0, init_params={}):
   (_, acc, f_mod, _, params, _, _) = train(f_train, f_test, d_tmp, phases, iters, timeout, init_params)
   logger.info("")
   logger.info("Best model params: %s" % str(params))
   logger.info("Best model accuracy: %s" % human.humanacc(acc))
   logger.info("Best model file: %s" % f_mod)

