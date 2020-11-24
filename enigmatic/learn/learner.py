#!/usr/bin/python

import sys
import os.path
import json
import time
import logging

from pyprove import log, redirect
from pyprove.bar import ProgressBar
from enigmatic import models, trains

logger = logging.getLogger(__name__)

class Learner:
   
   def __init__(self, bar_round=None):
      self.stats = {}
      self.bar_round = bar_round 

   def efun(self):
      "E Prover weight function name."
      return "Enigma"

   def ext(self):
      "Model filename extension."
      return "ext"

   def name(self):
      return "learner"

   def desc(self):
      return "default"

   def train(self, f_in, f_mod, init_model=None, handlers=None):
      pass

   def readlog(self, f_log):
      return

   def build(self, f_in, f_mod, f_log, options=[], init_model=None):
      def atfinish():
         bar.finish()
         bar.file.flush()
      # progress bar
      if "headless" in options:
         (bar, atstart, atiter, atfinish) = (None, None, None, None)
      else:
         ProgressBar.file = None
         bar = ProgressBar("[%s]"%self.ext(), max=self.bar_round)
         atstart = bar.start
         atiter = bar.next
         handlers = (atstart, atiter, atfinish)
      logger.info("- building model %s" % f_mod)
      logger.debug(log.data("- learning parameters:", self.params))
      # standard output redirect
      redir = redirect.start(f_log, bar)
      begin = time.time()
      try:
         self.train(f_in, f_mod, init_model=init_model, handlers=handlers)
      except Exception as e:
         redirect.finish(*redir)
         raise e # raise after redirect so that stack trace is not lost
      end = time.time()
      # redirect back
      redirect.finish(*redir)
      # compute statistics
      self.readlog(f_log)
      self.stats["model.train.time"] = end-begin
      self.stats["train.size"] = trains.size(f_in)
      self.stats["train.format"] = trains.format(f_in)
      self.stats["model.size"] = os.path.getsize(f_mod)
      with open("%s-stats.json"%f_log,"w") as f: 
         json.dump(self.stats, f, indent=3, sort_keys=True)
      with open("%s-params.json"%f_log,"w") as f: 
         json.dump(self.params, f, indent=3, sort_keys=True)
      logger.info(log.data("- training statistics: ", self.stats))

   def predict(self, f_in, f_mod):
      return []

   def accuracy(self, f_in, f_mod, ret=None):
      preds = list(self.predict(f_in, f_mod))
      logger.debug("- predictions: %d", len(preds))
      acc = sum([1 for (x,y) in preds if int(x>0.5)==y])
      acc /= len(preds)
      logger.debug("- accuracy: %f" % acc)
      if ret is not None: ret["acc"] = acc
      return acc

   def refit(self, f_in, f_mod, f_log, options=[]):
      logger.info("- skipped refit of %s with %s" % (f_mod, f_in))

