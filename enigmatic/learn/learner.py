#!/usr/bin/python

import sys

class Learner:
   
   def __init__(self):
      pass

   def efun(self):
      "E Prover weight function name."
      return "Enigma"

   def ext(self):
      "Model filename extension."
      return "ext"

   def name(self):
      return "learner"

   def train(self, f_in, f_mod, f_log=None, f_stats=None):
      pass

   def build(self, f_in, f_mod, f_log=None, f_stats=None):
      ret = self.train(f_in, f_mod, f_log, f_stats)
      return ret

   def predict(self, f_in, f_mod):
      return {}

