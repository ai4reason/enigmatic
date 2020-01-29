from multiprocessing import Pool, Manager
from progress.bar import FillingSquaresBar as Bar
import subprocess
from pyprove import expres, eprover
import os
import traceback
from pyprove import log

TRAINS_DIR = os.getenv("EXPRES_TRAINS", "./00TRAINS")
RAMDISK_DIR = None

def path(bid, pid, problem, limit, version, hashing, ext="out"):
   tid = bid.replace("/","-")
   tid += "-%s%s" % ("T" if isinstance(limit,int) else "", limit)
   vid = "%s%s" % (version, log.humanexp(hashing))
   f_out = "%s.%s" % (problem, ext)
   f = os.path.join(TRAINS_DIR, tid, vid, f_out)
   if RAMDISK_DIR and not os.path.isfile(f):
      f = os.path.join(RAMDISK_DIR, tid, vid, f_out)
   return f

def proofstate(f_dat, f_pos, f_neg, hashing=None):
   def parse(clause):
      clause = clause[clause.rindex("proofvector")+12:].rstrip(",\n").strip().split(",")
      clause = [x.split("(")[0].split(":") for x in clause if x]
      if not hashing:
         clause = ["$%s/%s"%tuple(x) for x in clause if x]
      else:
         clause = ["%s:%s"%tuple(x) for x in clause if x]
      return " ".join(clause)
   dat = open(f_dat).read().strip().split("\n")
   dat = [x for x in dat if x]
   i = 0
   for pos in open(f_pos):
      dat[i] += " "+parse(pos)
      i += 1
   for neg in open(f_neg):
      dat[i] += " "+parse(neg)
      i += 1
   if i != len(dat):
      raise Exception("File %s does not match files %s and %s!" % (f_dat,f_pos,f_neg))
   open(f_dat, "w").write("\n".join(dat))

def prepare2(job):
   queue = job[7]
   try:
      prepare1(job)
   except:
      print("Error: "+traceback.format_exc())
   queue.put(job[2])

def prepare1(job):
   (bid,pid,problem,limit,version,force,hashing,queue) = job

   f_problem = expres.benchmarks.path(bid, problem)
   f_cnf = expres.benchmarks.path(os.path.join(bid,"cnf"), problem)
   if not os.path.isfile(f_cnf):
      open(f_cnf, "wb").write(eprover.runner.cnf(f_problem))

   result = None
   #result = rkeys[(bid,pid,problem,limit)]
   f_pos = expres.results.path(bid, pid, problem, limit, ext="pos")
   f_neg = expres.results.path(bid, pid, problem, limit, ext="neg")
   os.system("mkdir -p %s" % os.path.dirname(f_pos))
   os.system("mkdir -p %s" % os.path.dirname(f_neg))
   if force or (not (os.path.isfile(f_pos) and os.path.isfile(f_neg))):
      result = expres.results.load(bid, pid, problem, limit, trains=True, proof=True)
      if force or not os.path.isfile(f_pos):
         open(f_pos, "w").write("\n".join(result["POS"]))
      if force or not os.path.isfile(f_neg):
         open(f_neg, "w").write("\n".join(result["NEG"]))
      # extract additional positive samples from the proof
      #f_sol = expres.results.path(bid, pid, problem, limit, ext="sol")
      #open(f_sol, "w").write("\n".join(result["PROOF"]))
      #f_prf = expres.results.path(bid, pid, problem, limit, ext="prf")
      #prf = open(f_prf, "w")
      #subprocess.call(["eprover", "--free-numbers", "--cnf", f_sol], stdout=prf)
      ##subprocess.call(["eprover", "--free-numbers", "--cnf", "--no-preprocessing", f_sol], stdout=prf)
      #prf.close()
      #os.system("cat %s | grep '^cnf' >> %s" % (f_prf, f_pos))
   
   #f_dat = expres.results.path(bid, pid, problem, limit, ext="in" if hashing else "pre")
   f_dat = path(bid, pid, problem, limit, version, hashing, ext="in" if hashing else "pre")
   #f_map = expres.results.path(bid, pid, problem, limit, ext="map")
   f_map = path(bid, pid, problem, limit, version, hashing, ext="map")
   os.system("mkdir -p %s" % os.path.dirname(f_dat))
   os.system("mkdir -p %s" % os.path.dirname(f_map))
   if force or not os.path.isfile(f_dat):
      out = open(f_dat, "w")
      if not hashing:
         subprocess.call(["enigma-features", "--free-numbers", "--enigma-features=%s"%version, \
            f_pos, f_neg, f_cnf], stdout=out)
            #stdout=out, stderr=subprocess.STDOUT)
      else:
         subprocess.call(["enigma-features", "--free-numbers", "--enigma-features=%s"%version, \
            "--feature-hashing=%s"%hashing, "--enigmap-file=%s"%f_map, f_pos, f_neg, f_cnf], stdout=out)

      out.close()
      if "W" in version:
         proofstate(f_dat, f_pos, f_neg, hashing)

def prepare(rkeys, version, force=False, cores=1, hashing=None):
   pool = Pool(cores)
   m = Manager()
   queue = m.Queue()
   jobs = [rkey+(version,force,hashing,queue) for rkey in rkeys]
   bar = Bar("[1/3]", max=len(jobs), suffix="%(percent).1f%% / %(elapsed_td)s / ETA %(eta_td)s")
   bar.start()
   res = pool.map_async(prepare2, jobs, chunksize=1)
   todo = len(jobs)
   while todo:
      queue.get()
      todo -= 1
      bar.next()
   bar.finish()
   pool.close()
   pool.join()

def translate(f_cnf, f_conj, f_out):
   "deprecated?"

   out = open(f_out, "w")
   if not f_conj:
      subprocess.call(["enigma-features", "--free-numbers", f_cnf], stdout=out)
   else:   
      f_empty = "empty.tmp"
      os.system("rm -fr %s" % f_empty)
      os.system("touch %s" % f_empty)
      subprocess.call(["enigma-features", "--free-numbers", f_cnf, f_empty, f_conj], \
         stdout=out)
         #stdout=out, stderr=subprocess.STDOUT)
      os.system("rm -fr %s" % f_empty)
   out.close()

def make(rkeys, out=None, hashing=None, version=None):
   dat = []
   bar = Bar("[2/3]", max=len(rkeys), suffix="%(percent).1f%% / %(elapsed_td)s / ETA %(eta_td)s")
   bar.start()
   for (bid, pid, problem, limit) in rkeys:
      #f_dat = expres.results.path(bid, pid, problem, limit, ext="in" if hashing else "pre")
      f_dat = path(bid, pid, problem, limit, version, hashing, ext="in" if hashing else "pre")
      if out:
         tmp = open(f_dat).read().strip()
         if tmp:
            out.write(tmp)
            out.write("\n")
      else:
         dat.extend(open(f_dat).read().strip().split("\n"))
      bar.next()
   bar.finish()
   return dat if not out else None

