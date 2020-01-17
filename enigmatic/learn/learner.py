#!/usr/bin/python

import sys
import os.path
import json

from pyprove import log, redirect
from pyprove.bar import ProgressBar

class Learner:
   
   def __init__(self):
      self.stats = None
      self.num_round = None # set it to int to enable progress bar
      pass

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

   def build(self, f_in, f_mod, f_log=None, f_stats=None):
      self.stats = {}
      bar = None
      if self.num_round:
         bar = ProgressBar("[3/3]", max=self.num_round)
         bar.start()
      if f_log:
         redir = redirect.start(f_log, bar)

      ret = self.train(f_in, f_mod, iter_done=bar.next)
      self.stats["train.size"] = log.humanbytes(os.path.getsize(f_in))
      self.stats["model.size"] = log.humanbytes(os.path.getsize(f_mod))

      if bar:
         bar.finish() 
      if f_log:
         redirect.finish(*redir)
         print()
         self.readlog(f_log)
      if f_stats:
         with open(f_stats,"w") as f: json.dump(self.stats, f)

      return ret

   def predict(self, f_in, f_mod):
      return {}

