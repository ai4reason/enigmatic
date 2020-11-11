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

def filename(**others):
   return os.path.join(path(**others), "train.in")

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

def makesingle(f_list, features, f_problem=None, f_map=None, f_buckets=None, f_out=None, prefix=None):
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
      out = None
   if f_out and out:
      with open(f_out, "ab") as f: f.write(out)
   return out

def makes(posnegs, bid, features, cores, callback, msg="[+/-]", d_info=None, options=[], **others):
   def job(f_list):
      p = os.path.basename(f_list)[:-4]
      pos = f_list.endswith(".pos")
      f_problem = expres.benchmarks.path(bid, p)
      f_map = os.path.join(d_info, p+".map") if d_info else None
      f_buckets  = os.path.join(d_info, p+".json") if d_info else None
      f_out = os.path.join(d_info, p+".in") if d_info else None
      return (f_list, features, f_problem, f_map, f_buckets, f_out, pos)
   jobs = list(map(job, posnegs))
   barmsg = msg if not "headless" in options else None
   par.apply(makesingle, jobs, cores=cores, barmsg=barmsg, 
      callback=callback, chunksize=300)

def make(d_posnegs, debug=[], split=False, **others):
   def save(res, bar):
      nonlocal out
      if res:
         out.write(res)
   d_in = path(**others)
   f_in = filename(**others)
   logger.info("+ generating training files")
   os.system('mkdir -p "%s"' % d_in)
   if (exist(f_in) or os.path.isfile(f_in)) and not "force" in debug:
      logger.debug("- skipped generating %s" % f_in)
      return
   posnegs = []
   d_info = path(**others) if "train" in debug else None
   for d in d_posnegs:
      fs = [f for f in os.listdir(d) if f.endswith(".pos") or f.endswith(".neg")]
      fs = [os.path.join(d,f) for f in fs]
      posnegs.extend(fs)
   logger.info("- found %s pos/neg files in %s directories" % 
      (len(posnegs), len(d_posnegs)))
   logger.debug("- directories: %s", d_posnegs)

   if split:
      f_test = f_in+"-test.in"
      logger.debug("- generating tests file %s" % f_test)
      out = open(f_test, "wb")
      random.shuffle(posnegs)
      i = int(len(posnegs) * split)
      posneg0 = posnegs[:i]
      posneg1 = posnegs[i:]
      makes(posneg0, callback=save, d_info=d_info, debug=debug, msg="[tst]", **others)
      open(f_test+"-posnegs.txt","w").write("\n".join(posneg0))
      posnegs = posneg1

   logger.debug("- generating trains file %s" % f_in)
   open(f_in+"-posnegs.txt","w").write("\n".join(posnegs))
   out = open(f_in, "wb")
   makes(posnegs, callback=save, d_info=d_info, debug=debug, msg="[trn]", **others)
   out.close()
   # compress data as numpy npz data
   if not "nozip" in debug:
      compress(f_in)
      os.remove(f_in)
   enigmap.build(debug=["force"], path=path, **others)

def build(pids, **others):
   d_posnegs = [expres.results.dir(pid=pid, **others) for pid in pids]
   make(d_posnegs, **others)

