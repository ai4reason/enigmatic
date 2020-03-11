#!/usr/bin/python

import sys
import os.path
import json
import time

from pyprove import log, redirect
from pyprove.bar import ProgressBar
from enigmatic import models

class Learner:
   
   def __init__(self, bar_round=None):
      self.stats = None
      # total for progress bar
      self.bar_round = bar_round if log.ENABLED else None

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

   def train(self, f_in, f_mod, iter_done=None):
      pass

   def readlog(self, f_log):
      return

   def build(self, model, f_in=None, f_mod=None, f_log=True, f_stats=True):
      f_in = models.path(model, "train.in") if not f_in else f_in
      f_mod = models.path(model, "model.%s" % self.ext()) if not f_mod else f_mod
      if f_log is True: f_log = models.path(model, "train.log")
      if f_stats is True: f_stats = models.path(model, "train.stats")
      
      self.stats = {}
      self.model = model
      bar = None
      if self.bar_round:
         bar = ProgressBar("[3/3]", max=self.bar_round)
         bar.start()
      if f_log:
         redir = redirect.start(f_log, bar)

      begin = time.time()
      ret = self.train(f_in, f_mod, iter_done=bar.next if bar else None)
      end = time.time()
      self.stats["model.train.time"] = log.humantime(end-begin)
      self.stats["model.train.seconds"] = end-begin
      self.stats["train.size"] = log.humanbytes(os.path.getsize(f_in))
      self.stats["model.size"] = log.humanbytes(os.path.getsize(f_mod))

      if bar:
         bar.finish() 
      if f_log:
         redirect.finish(*redir)
         #print()
         self.readlog(f_log)
      if f_stats:
         with open(f_stats,"w") as f: json.dump(self.stats, f)

      return ret

   def predict(self, f_in, f_mod):
      return {}

   def nobar(self):
      self.bar_round = None

