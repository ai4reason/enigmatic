import subprocess
from . import enigmap

#PERF = ["perf", "stat", "-e", "task-clock:up,page-faults:up,instructions:up"]
PERF = []

def train(f_in, f_mod, f_out=None, f_log=None):
   if f_log:
      subprocess.call(PERF+["train", "-s" , "2", f_in, f_mod], \
         stdout=file(f_log, "a"), stderr=subprocess.STDOUT)
   else:
      subprocess.call(["train", "-s" , "2", f_in, f_mod])

   if f_out:
      predict(f_in, f_mod, f_out, f_log)

def predict(f_in, f_mod, f_out, f_log):
   if f_log:
      subprocess.call(PERF+["predict", f_in, f_mod, f_out], \
         stdout=file(f_log, "a"), stderr=subprocess.STDOUT)
   else:
      subprocess.call(["predict", f_in, f_mod, f_out])

def stats(f_in, f_out):
   ins = file(f_in).read().strip().split("\n")
   ins = [int(x.split()[0]) for x in ins]

   outs = file(f_out).read().strip().split("\n")
   outs = map(int, outs)

   stat = {}
   stat["COUNT"] = 0
   stat["COUNT:POS"] = 0
   stat["COUNT:NEG"] = 0
   stat["RIGHT"] = 0
   stat["RIGHT:POS"] = 0
   stat["RIGHT:NEG"] = 0
   stat["WRONG"] = 0
   stat["WRONG:POS"] = 0
   stat["WRONG:NEG"] = 0

   for (correct,predicted) in zip(ins, outs):
      stat["COUNT"] += 1
      if correct == 1:
         stat["COUNT:POS"] += 1
      else:
         stat["COUNT:NEG"] += 1

      if correct == predicted:
         stat["RIGHT"] += 1
         if correct == 1:
            stat["RIGHT:POS"] += 1
         else:
            stat["RIGHT:NEG"] += 1
      else:
         stat["WRONG"] += 1
         if correct == 1:
            stat["WRONG:POS"] += 1
         else:
            stat["WRONG:NEG"] += 1
   
   stat["ACC"] = stat["RIGHT"] / float(stat["COUNT"])
   stat["ACC:POS"] = (stat["RIGHT:POS"] / float(stat["COUNT:POS"])) if int(stat["COUNT:POS"]) != 0 else 0
   stat["ACC:NEG"] = (stat["RIGHT:NEG"] / float(stat["COUNT:NEG"])) if int(stat["COUNT:NEG"]) != 0 else 0

   return stat

def load(f_mod, f_map):
   emap = enigmap.load(f_map)
   lines = file(f_mod)
   rev = {emap[ftr]:ftr for ftr in emap}

   header = {}
   for line in lines:
      line = line.strip()
      if line == "w":
         break
      parts = line.split()
      header[parts[0]] = " ".join(parts[1:])

   # clause features
   w1 = {}
   i = 1
   for line in lines:
      w = float(line.strip())
      w1[rev[i]] = w
      if i == len(emap):
         break
      i += 1

   # conjecture features
   w2 = {}
   i = 1
   for line in lines:
      w = float(line.strip())
      w2[rev[i]] = w
      if i == len(emap):
         break
      i += 1

   lines.close()
   return (header,w1,w2)

def save(header, w1, w2, emap, f_mod):
   rev = {emap[ftr]:ftr for ftr in emap}
   lines = file(f_mod, "w")

   info = lambda key: lines.write("%s %s\n" % (key, header[key]))
   info("solver_type")
   info("nr_class")
   info("label")
   lines.write("nr_feature %d\n" % (2*len(emap)))
   info("bias")
   lines.write("w\n")

   for i in xrange(1,len(rev)+1):
      if rev[i] in w1 and w1[rev[i]] != 0:
         lines.write("%.16f\n"%w1[rev[i]])
      else:
         lines.write("0\n")
   
   for i in xrange(1,len(rev)+1):
      if rev[i] in w2 and w2[rev[i]] != 0:
         lines.write("%.16f\n"%w2[rev[i]])
      else:
         lines.write("0\n")

   lines.close()

