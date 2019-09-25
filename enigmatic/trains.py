import math

from .enigmap import fhash

PREFIX = {
   "+": "+1",
   "-": "+0",
   "*": ""
}

BOOSTS = {
   "WRONG:POS": 1,
   "WRONG:NEG": 0
}

def count(ftrs, vector, emap, offset, strict=True):
   for ftr in ftrs:
      if not ftr:
         continue
      if ftr.startswith("$"):
         continue
      if "/" in ftr:
         parts = ftr.split("/")
         ftr = parts[0]
         inc = int(parts[1])
         if inc == 0:
            continue
      else:
         inc = 1
      if (not strict) and (ftr not in emap):
         continue
      if isinstance(emap, int): # hashing version (emap::int is the base)
         fid = fhash(ftr, emap) + offset
      else:
         fid = emap[ftr] + offset
      vector[fid] = vector[fid]+inc if fid in vector else inc

def proofstate(ftrs, vector, offset, hashing=None):
   # TODO: hash also proof numbers (modulo hashing)
   for ftr in ftrs:
      if not ftr.startswith("$"):
         continue
      (num, val) = ftr[1:].split("/")
      (num, val) = (int(num), float(val))
      if not val:
         continue
      vector[offset+num] = val

def string(sign, vector):
   ftrs = ["%s:%s"%(fid,vector[fid]) for fid in sorted(vector)] 
   ftrs = "%s %s"%(PREFIX[sign], " ".join(ftrs))
   return ftrs

def normalize(vector):
    non0 = len([x for x in vector if vector[x]])
    non0 = math.sqrt(non0)
    return {x:vector[x]/non0 for x in vector}

def encode(pr, emap, strict=True):
   (sign,clause,conj) = pr.strip().split("|")
   vector = {}
   count(clause.strip().split(" "), vector, emap, 0, strict)
   conjs = conj.strip().split(" ")
   base = emap if isinstance(emap,int) else len(emap)
   count(conjs, vector, emap, base, strict)
   proofstate(conjs, vector, 2*base, emap)
   #vector = normalize(vector)
   return string(sign, vector)

def make(pre, emap, out=None, strict=True):
   train = []
   for pr in pre:
      tr = encode(pr, emap, strict)
      if out:
         out.write(tr)
         out.write("\n")
      else:
         train.append(tr)
   return train if not out else None

def boost(f_in, f_out, out, method="WRONG:POS"):
   if method not in BOOSTS:
      raise Exception("Unknown boost method (%s)")
   CLS = BOOSTS[method]

   ins = open(f_in).read().strip().split("\n")
   outs = open(f_out).read().strip().split("\n")

   for (correct,predicted) in zip(ins,outs):
      out.write(correct)
      out.write("\n")
      cls = int(correct.split()[0])
      if cls == CLS and cls != int(predicted):
         out.write(correct)
         out.write("\n")

