import os, io
import subprocess
import logging, random
from sklearn.datasets import load_svmlight_file
import numpy, scipy
from pyprove import expres, par, log, human
from . import enigmap

DEFAULT_NAME = "00TRAINS"
DEFAULT_DIR = os.getenv("PYPROVE_TRAINS", DEFAULT_NAME)

logger = logging.getLogger(__name__)

def path(bid, limit, features, dataname, **others):
   bid = bid.replace("/","-")
   tid = "%s-%s" % (bid, limit)
   return os.path.join(DEFAULT_DIR, tid, dataname, features)

def filename(prefix=None, **others):
   return os.path.join(path(**others), "train.in" if not prefix else prefix)

def datafiles(f_in):
   z_data = f_in + "-data.npz"
   z_label = f_in + "-label.npz"
   return [z_data, z_label]

def size(f_in):
   z_data = datafiles(f_in)[0]
   if os.path.isfile(z_data):
      f_in = z_data
   return os.path.getsize(f_in)

def format(f_in):
   if os.path.isfile(datafiles(f_in)[0]):
      return "binary/npz"
   if os.path.isfile(f_in):
      return "text/svm"
   return "unknown"

def exist(f_in):
   "True iff **compressed** data files exist"
   return all(map(os.path.isfile, datafiles(f_in)))

def load(f_in):
   if exist(f_in): 
      (z_data, z_label) = datafiles(f_in)
      data = scipy.sparse.load_npz(z_data)
      label = numpy.load(z_label, allow_pickle=True)["label"]
   else:
      (data, label) = load_svmlight_file(f_in, zero_based=True)
   return (data, label)

def compress(f_in):
   logger.debug("- loading %s" % f_in)
   logger.debug("- uncompressed size: %s" % human.humanbytes(size(f_in)))
   (data, label) = load_svmlight_file(f_in, zero_based=True)
   (z_data, z_label) = datafiles(f_in)
   logger.debug("- compressing to %s" % z_data)
   scipy.sparse.save_npz(z_data, data, compressed=True)
   numpy.savez_compressed(z_label, label=label)
   logger.debug("- compressed size: %s" % human.humanbytes(size(f_in)))

def forgetting(lines, forget):
   if forget is None:
      return lines
   def duplicates(line):
      nonlocal cache, lines
      if not line in cache:
         cache[line] = lines.count(line)
      return cache[line]
   
   lines = lines.decode()
   lines = lines.strip().split("\n")
   if forget:
      cache = {}
      [duplicates(l) for l in lines] 
      lines = sorted(set(lines), key=duplicates, reverse=True)
      idx = int((1.0-forget)*len(lines))
      idx = min(len(lines),max(1,idx))
      lines = lines[:idx]
   else:
      lines = list(set(lines))
   lines = "\n".join(lines)+"\n"
   lines = lines.encode()
   return lines

def makesingle(f_list, features, f_problem=None, f_map=None, f_buckets=None, f_out=None, prefix=None, forget=0.0):
   args = [
      "enigmatic-features", 
      "--free-numbers", 
      "--features=%s" % features
   ]
   if f_map:
      args.append("--output-map=%s" % f_map)
   if f_buckets:
      args.append("--output-buckets=%s" % f_buckets)
   if f_problem:
      args.append("--problem=%s" % f_problem)
   if prefix is True:
      args.append("--prefix-pos")
   elif prefix is False:
      args.append("--prefix-neg")
   elif prefix is not None:
      args.append("--prefix=%s" % prefix)
   args.append(f_list)
   try:
      out = subprocess.check_output(args)
   except subprocess.CalledProcessError as e:
      return None
   out = forgetting(out, forget)
   if f_out:
      with open(f_out, "ab") as f: f.write(out)
   return out

def makes(posnegs, f_prfx, bid, features, cores, msg="[+/-]", d_info=None, options=[], debug=[], chunksize=None, forgets=(None,None), **others):
   def job(f_list):
      p = os.path.basename(f_list)[:-4]
      pos = f_list.endswith(".pos")
      f_problem = expres.benchmarks.path(bid, p)
      f_map = os.path.join(d_info, p+".map") if d_info else None
      f_buckets  = os.path.join(d_info, p+".json") if d_info else None
      f_out = os.path.join(d_info, p+".in") if d_info else None
      forget = forgets[int(pos)]
      return (f_list, features, f_problem, f_map, f_buckets, f_out, pos, forget)
   def save(res, bar):
      nonlocal out, count, written, parts, f_in, has_pos, has_neg
      if not res: return
      has_pos = has_pos or b"+" in res
      has_neg = has_neg or b"-" in res
      if written and chunksize and written >= chunksize and has_neg and has_neg:
         #dump(out, f_in)
         out.close()
         count += 1
         f_in = "%s.in-part%03d.in" % (f_prfx, count)
         out = open(f_in, "wb")
         out.write(res)
         parts.append(f_in)
         written = 0
         has_pos = b"+" in res
         has_neg = b"-" in res
      else:
         out.write(res)
         written += len(res)
   logger.debug("- generating trains %s" % f_prfx)
   f_in = "%s.in" % f_prfx
   with open("%s-posnegs.txt"%f_in,"w") as f: f.write("\n".join(posnegs))
   jobs = list(map(job, posnegs))
   barmsg = msg if not "headless" in options else None
   parts = [f_in]
   out = open(f_in, "wb")
   count = 0
   written = 0
   has_pos = False
   has_neg = False
   par.apply(makesingle, jobs, cores=cores, barmsg=barmsg, 
      callback=save, chunksize=100)
   out.close()
   if not (has_neg and has_pos):
      logger.debug("- last part without both pos/negs; appending to the previous")
      f_last = parts[-1]
      parts = parts[:-1]
      with open(parts[-1],"ab") as prev:
         with open(f_last,"rb") as last:
            prev.write(last.read())
      os.remove(f_last)
   if not "nozip" in debug:
      for f in parts:
         compress(f)
         os.remove(f)

def collect(d_posnegs, **others):
   posnegs = []
   for d in d_posnegs:
      fs = [f for f in os.listdir(d) if f.endswith(".pos") or f.endswith(".neg")]
      fs = [os.path.join(d,f) for f in fs]
      posnegs.extend(fs)
   logger.info("- found %s pos/neg files in %s directories" % 
      (len(posnegs), len(d_posnegs)))
   logger.debug("- directories: %s", d_posnegs)
   return posnegs

def make(d_posnegs, debug=[], split=False, **others):
   d_in = path(**others)
   f_in = filename(**others)
   logger.info("+ generating training files")
   os.system('mkdir -p "%s"' % d_in)
   if (exist(f_in) or os.path.isfile(f_in)) and not "force" in debug:
      logger.debug("- skipped generating %s" % f_in)
      return
   d_info = path(**others) if "train" in debug else None
   posnegs = collect(d_posnegs)

   if split:
      f_prfx = filename("test", **others)
      logger.debug("- generating tests %s" % f_prfx)
      random.shuffle(posnegs)
      i = int(len(posnegs) * split)
      posneg0 = posnegs[:i]
      posnegs = posnegs[i:]
      makes(posneg0, f_prfx, d_info=d_info, debug=debug, msg="[tst]", **others)

   f_prfx = filename("train", **others)
   makes(posnegs, f_prfx, d_info=d_info, debug=debug, msg="[trn]", **others)
   enigmap.build(debug=["force"], path=path, **others)

def build(pids, **others):
   d_posnegs = [expres.results.dir(pid=pid, **others) for pid in pids]
   make(d_posnegs, **others)

