import os, io
import subprocess
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from pyprove import expres, par, log
from scipy.sparse import vstack, hstack
import lightgbm as lgb

#from pyprove import expres, eprover
#import traceback
#from pyprove import log

###DEFAULT_NAME = "00TRAINS"
###DEFAULT_DIR = os.getenv("ENIGMATIC_TRAINS", DEFAULT_NAME)
###RAMDISK_DIR = None
###
###TRAINS_DIR = os.getenv("EXPRES_TRAINS", "./00TRAINS") # todel
###
###def path(bid, pid, problem, limit, version, hashing, ext="out"):
###   global DEFAULT_DIR, RAMDISK_DIR
###   tid = bid.replace("/","-")
###   tid += "-%s%s" % ("T" if isinstance(limit,int) else "", limit)
###   vid = "%s%s" % (version, log.humanexp(hashing))
###   f_out = "%s.%s" % (problem, ext)
###   f = os.path.join(DEFAULT_DIR, tid, vid, pid, f_out)
###   if RAMDISK_DIR and not os.path.isfile(f):
###      f = os.path.join(RAMDISK_DIR, tid, vid, pid, f_out)
###   return f

DEFAULT_NAME = "00TRAINS"
DEFAULT_DIR = os.getenv("PYPROVE_TRAINS", DEFAULT_NAME)
#RAMDISK_DIR = None

def dirpath(bid, pid, limit, version, hashing):
   global DEFAULT_DIR, RAMDISK_DIR
   tid = "%s-%s" % (bid.replace("/","-"), limit)
   vid = "%s%s" % (version, log.humanexp(hashing))
   return os.path.join(DEFAULT_DIR, tid, vid)

def makeone(f_pos, f_neg, f_cnf, version, hashing, f_in=None, f_map=None):
   args = [
      "enigma-features", 
      "--free-numbers", 
      "--enigma-features=%s" % version,
      "--feature-hashing=%s" % hashing
   ]
   if f_map:
      args.append("--enigmap-file=%s" % f_map)
   args.extend([f_pos, f_neg, f_cnf])
   try:
      out = subprocess.check_output(args, stderr=subprocess.STDOUT)
   except subprocess.CalledProcessError as e:
      out = e.output
   if f_in:
      with open(f_in, "wb") as f: f.write(out)
   with io.BytesIO(out) as data:
      (xs, ys) = load_svmlight_file(data)
   return (xs, ys)

   
def makedir(d_out, bid, version, hashing, cores, callback, msg="[*]", d_info=None):

   def job(p):
      nonlocal d_out, bid, version, hashing
      f_pos = os.path.join(d_out, p+".pos")
      f_neg = os.path.join(d_out, p+".neg")
      f_cnf = expres.benchmarks.path(os.path.join(bid,"cnf"), p)
      f_in  = os.path.join(d_info, p+".in") if d_info else None
      f_map = os.path.join(d_info, p+".map") if d_info else None
      return (f_pos, f_neg, f_cnf, version, hashing, f_in, f_map)
   
   def select(ext):
      nonlocal d_out
      return [f[:-(1+len(ext))] for f in os.listdir(d_out) if f.endswith("."+ext)]

   pos = select("pos")
   neg = select("neg")
   jobs = [job(p) for p in pos if p in neg]
   ret = par.apply(makeone, jobs, cores=cores, barmsg=msg, callback=callback)

   return ret

def make(d_outs, bid, version, hashing, out, cores=4, **others):
   
   def save(arg, res, bar):
      nonlocal out
      (xs0, ys0) = res
      dump_svmlight_file(xs0, ys0, out)

   callback = save if out else None
   rets = []
   for (n,d_out) in enumerate(d_outs):
      msg = "[%s/%s]" % (n+1, len(d_outs))
      ret = makedir(d_out, bid, version, hashing, cores, callback, msg)
      if not out:
         rets.extend(ret)

   return rets

def build(bid, pids, limit, version, hashing, cores, out=None, **others):
   pass



##
##     
##
##
##
##
##
##def prepare2(job):
##   queue = job[7]
##   try:
##      prepare1(job)
##   except:
##      print("Error: "+traceback.format_exc())
##   queue.put(job[2])
##
##def prepare1(job):
##   (bid,pid,problem,limit,version,force,hashing,queue) = job
##
##   f_problem = expres.benchmarks.path(bid, problem)
##   f_cnf = expres.benchmarks.path(os.path.join(bid,"cnf"), problem)
##   if not os.path.isfile(f_cnf):
##      open(f_cnf, "wb").write(eprover.runner.cnf(f_problem))
##
##   result = None
##   #result = rkeys[(bid,pid,problem,limit)]
##   f_pos = expres.results.path(bid, pid, problem, limit, ext="pos")
##   f_neg = expres.results.path(bid, pid, problem, limit, ext="neg")
##   os.system("mkdir -p %s" % os.path.dirname(f_pos))
##   os.system("mkdir -p %s" % os.path.dirname(f_neg))
##   if force or (not (os.path.isfile(f_pos) and os.path.isfile(f_neg))):
##      result = expres.results.load(bid, pid, problem, limit, trains=True, proof=True)
##      if force or not os.path.isfile(f_pos):
##         open(f_pos, "w").write("\n".join(result["POS"]))
##      if force or not os.path.isfile(f_neg):
##         open(f_neg, "w").write("\n".join(result["NEG"]))
##   
##   #f_dat = expres.results.path(bid, pid, problem, limit, ext="in" if hashing else "pre")
##   f_dat = path(bid, pid, problem, limit, version, hashing, ext="in" if hashing else "pre")
##   #f_map = expres.results.path(bid, pid, problem, limit, ext="map")
##   f_map = path(bid, pid, problem, limit, version, hashing, ext="map")
##   os.system("mkdir -p %s" % os.path.dirname(f_dat))
##   os.system("mkdir -p %s" % os.path.dirname(f_map))
##   if force or not os.path.isfile(f_dat):
##      out = open(f_dat, "w")
##      if not hashing:
##         subprocess.call(["enigma-features", "--free-numbers", "--enigma-features=%s"%version, \
##            f_pos, f_neg, f_cnf], stdout=out)
##            #stdout=out, stderr=subprocess.STDOUT)
##      else:
##         subprocess.call(["enigma-features", "--free-numbers", "--enigma-features=%s"%version, \
##            "--feature-hashing=%s"%hashing, "--enigmap-file=%s"%f_map, f_pos, f_neg, f_cnf], stdout=out)
##
##      out.close()
##      if "W" in version:
##         proofstate(f_dat, f_pos, f_neg, hashing)
##
##def prepare(rkeys, version, force=False, cores=1, hashing=None):
##   pool = Pool(cores)
##   m = Manager()
##   queue = m.Queue()
##   jobs = [rkey+(version,force,hashing,queue) for rkey in rkeys]
##   bar = Bar("[1/3]", max=len(jobs), suffix="%(percent).1f%% / %(elapsed_td)s / ETA %(eta_td)s")
##   bar.start()
##   res = pool.map_async(prepare2, jobs, chunksize=1)
##   todo = len(jobs)
##   while todo:
##      queue.get()
##      todo -= 1
##      bar.next()
##   bar.finish()
##   pool.close()
##   pool.join()
##
##def translate(f_cnf, f_conj, f_out):
##   "deprecated?"
##
##   out = open(f_out, "w")
##   if not f_conj:
##      subprocess.call(["enigma-features", "--free-numbers", f_cnf], stdout=out)
##   else:   
##      f_empty = "empty.tmp"
##      os.system("rm -fr %s" % f_empty)
##      os.system("touch %s" % f_empty)
##      subprocess.call(["enigma-features", "--free-numbers", f_cnf, f_empty, f_conj], \
##         stdout=out)
##         #stdout=out, stderr=subprocess.STDOUT)
##      os.system("rm -fr %s" % f_empty)
##   out.close()
##
##def make(rkeys, out=None, hashing=None, version=None):
##   dat = []
##   bar = Bar("[2/3]", max=len(rkeys), suffix="%(percent).1f%% / %(elapsed_td)s / ETA %(eta_td)s")
##   bar.start()
##   for (bid, pid, problem, limit) in rkeys:
##      #f_dat = expres.results.path(bid, pid, problem, limit, ext="in" if hashing else "pre")
##      f_dat = path(bid, pid, problem, limit, version, hashing, ext="in" if hashing else "pre")
##      if out:
##         tmp = open(f_dat).read().strip()
##         if tmp:
##            out.write(tmp)
##            out.write("\n")
##      else:
##         dat.extend(open(f_dat).read().strip().split("\n"))
##      bar.next()
##   bar.finish()
##   return dat if not out else None
##
