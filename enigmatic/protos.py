import re, os
from pyprove import expres, log
from . import models
import logging

logger = logging.getLogger(__name__)

def cef(freq, efun, fname, prio="PreferWatchlist", binary_weigths=1, threshold=0.5):
   cef = '%d*%s(%s,"%s",%s,%s)' % (freq,efun,prio,fname,binary_weigths,threshold)
   return cef

def solo(pid, name, mult=0, noinit=False, efun="Enigma", fullname=False, binary_weigths=1, threshold=0.5, prio="PreferWatchlist"):
   proto = expres.protos.load(pid)
   fname = os.path.join(models.DEFAULT_DIR, name)
   enigma = cef(1, efun, fname, prio, binary_weigths, threshold)
   eproto = "%s-H'(%s)'" % (proto[:proto.index("-H'")], enigma)
   if noinit:
      eproto = eproto.replace("--prefer-initial-clauses", "")
   if fullname:
      post = efun
      post += ("0M%s" % mult) if mult else "0"
      if noinit:
         post += "No" 
      epid = "Enigma+%s+%s+%s" % (name.replace("/","+"), pid, post)
   else:
      epid = "Enigma+%s+solo-%s" % (name.replace("/","+"), pid)
   expres.protos.save(epid, eproto)
   return epid

def coop(pid, name, freq=None, mult=0, noinit=False, efun="Enigma", fullname=False, binary_weigths=1, threshold=0.5, prio="PreferWatchlist"):
   proto = expres.protos.load(pid)
   fname = os.path.join(models.DEFAULT_DIR, name)
   post = efun
   if not freq:
      freq = sum(map(int,re.findall(r"(\d*)\*", proto)))
      post += "S"
   else:
      post += "F%s"% freq
   post += ("M%s" % mult) if mult else ""
   enigma = cef(freq, efun, fname, prio, binary_weigths, threshold)
   eproto = proto.replace("-H'(", "-H'(%s,"%enigma)
   if noinit:
      eproto = eproto.replace("--prefer-initial-clauses", "")
   if fullname:
      if noinit:
         post += "No"
      epid = "Enigma+%s+%s+%s" % (name.replace("/","+"), pid, post)
   else:
      epid = "Enigma+%s+coop-%s" % (name.replace("/","+"), pid)
   expres.protos.save(epid, eproto)
   return epid

def build(model, learner, pids=None, refs=None, **others):
   refs = refs if refs else pids
   logger.info("- creating Enigma strategies for model %s" % model)
   logger.debug("- base strategies: %s" % refs)
   efun = learner.efun()
   new = []
   for ref in refs:
      new.extend([
         solo(ref, model, mult=0, noinit=True, efun=efun),
         coop(ref, model, mult=0, noinit=True, efun=efun)
      ])
   logger.debug(log.lst("- %d new strategies:"%len(new), new))
   return new

