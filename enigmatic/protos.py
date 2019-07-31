import re
from pyprove import expres

def solo(pid, name, mult=0, noinit=False, efun="Enigma", fullname=False):
   proto = expres.protos.load(pid)
   enigma = "1*%s(PreferWatchlist,%s,%s)" % (efun, name, mult)
   eproto = "%s-H'(%s)'" % (proto[:proto.index("-H'")], enigma)
   if fullname:
      post = efun
      post += ("0M%s" % mult) if mult else "0"
      if noinit:
         eproto = eproto.replace("--prefer-initial-clauses", "")
         post += "No" 
      epid = "Enigma+%s+%s+%s" % (name.replace("/","+"), pid, post)
   else:
      epid = "Enigma+%s+solo" % name.replace("/","+")
   expres.protos.save(epid, eproto)
   return epid

def coop(pid, name, freq=None, mult=0, noinit=False, efun="Enigma", fullname=False):
   proto = expres.protos.load(pid)
   post = efun
   if not freq:
      freq = sum(map(int,re.findall(r"(\d*)\*", proto)))
      post += "S"
   else:
      post += "F%s"% freq
   post += ("M%s" % mult) if mult else ""
   enigma = "%d*%s(PreferWatchlist,%s,%s)" % (freq,efun,name,mult)
   eproto = proto.replace("-H'(", "-H'(%s,"%enigma)
   if fullname:
      if noinit:
         eproto = eproto.replace("--prefer-initial-clauses", "")
         post += "No"
      epid = "Enigma+%s+%s+%s" % (name.replace("/","+"), pid, post)
   else:
      epid = "Enigma+%s+coop" % name.replace("/","+")
   expres.protos.save(epid, eproto)
   return epid

