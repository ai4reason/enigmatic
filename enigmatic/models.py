import os
import json
from multiprocessing import Process

from . import enigmap, pretrains, trains, protos
from pyprove import expres, log

ENIGMA_ROOT = os.getenv("ENIGMA_ROOT", "./Enigma")
RAMDISK_DIR = None

DEFAULTS = {
   "gzip": True,
   "force": False,
   "hashing": None,
   "version": "VHSLC",
   "eargs": "--training-examples=3 -s",
   "cores": 4,
   "hash_debug": False,
}


def path(model, filemodel=None):
   def add(f):
      return f if not filemodel else os.path.join(f, filemodel)
      
   f = add(os.path.join(ENIGMA_ROOT, model))
   if RAMDISK_DIR and not os.path.isfile(f):
      f = add(os.path.join(RAMDISK_DIR, model))
   return f


def name(bid, limit, ref, version, learner, hashing, **others):
   return "%s-%s/%s-%s/%s" % (bid.replace("/","-"), limit, ref, version, learner.desc())

def collect(model, rkeys, settings):
   version = settings["version"]
   force = settings["force"]
   cores = settings["cores"]
   hashing = settings["hashing"] if not settings["hash_debug"] else None

   f_dat = path(model, "train.%s" % ("in" if hashing else "pre"))
   if force or not os.path.isfile(f_dat):
      log.msg("+ extracting training data from results")
      pretrains.prepare(rkeys, version, force, cores, hashing)
      log.msg("+ collecting %s data" % ("training" if hashing else "pretrains"))
      pretrains.make(rkeys, out=open(f_dat, "w"), hashing=hashing)


def setup(model, rkeys, settings):
   os.system("mkdir -p %s" % path(model))
   f_pre = path(model, "train.pre")
   f_map = path(model, "enigma.map")
   f_log = path(model, "train.log")
   hashing = settings["hashing"]
  
   if os.path.isfile(f_map) and os.path.isfile(f_pre) and not settings["force"]:
      return enigmap.load(f_map) if not hashing else hashing
      
   if rkeys:
      collect(model, rkeys, settings)

   if hashing and not settings["hash_debug"]:
      open(f_map,"w").write('version("%s").\nhash_base(%s).\n' % (settings["version"], hashing))
      return hashing

   #if os.path.isfile(f_log):
   #   os.system("rm -f %s" % f_log)
   if settings["force"] or not os.path.isfile(f_map):
      log.msg("+ creating feature info")
      emap = enigmap.create(open(f_pre), hashing)
      enigmap.save(emap, f_map, settings["version"], hashing)
   else:
      if not hashing:
         emap = enigmap.load(f_map)

   return emap if not hashing else hashing


def make(model, rkeys, settings):
   
   learner = settings["learner"]

   f_pre   = path(model, "train.pre")
   f_in    = path(model, "train.in")
   f_stats = path(model, "train.stats")
   f_mod   = path(model, "model.%s" % learner.ext())
   f_log   = path(model, "train.log")

   if os.path.isfile(f_mod) and not settings["force"]:
      return True

   emap = setup(model, rkeys, settings)
   if not emap:
      os.system("rm -fr %s" % path(model))
      return False
   
   if settings["hash_debug"] or not settings["hashing"]:
      if settings["force"] or not os.path.isfile(f_in):
         log.msg("+ generating training data")
         trains.make(open(f_pre), emap, out=open(f_in, "w"))

   log.msg("+ training %s model" % learner.name())
   p = Process(target=learner.build, args=(f_in,f_mod,f_log,f_stats))
   p.start()
   p.join()

   # wait and show progress bar 
   #total = learner.rounds()
   #bar = Bar("[3/3]", max=learner.rounds(), suffix="%(percent).1f%% / %(elapsed_td)s / ETA %(eta_td)s")
   #bar.start()
   #done = 0
   #while p.is_alive():
   #   cur = learner.current(f_log)
   #   while done < cur:
   #      done += 1
   #      bar.next()
   #   p.join(1)
   #while done < total:
   #   done += 1
   #   bar.next()
   #bar.finish()

   stats = json.load(open(f_stats)) if os.path.isfile(f_stats) else {}
   log.msg("+ training statistics:\n%s" % "\n".join(["%-23s = %s"%(x,stats[x]) for x in sorted(stats)]))

   if settings["gzip"]:
      log.msg("+ compressing training files")
      os.system("cd %s; gzip -qf *.pre *.in *.out 2>/dev/null" % path(model))

   return True


def check(settings):

   for x in DEFAULTS:
      if x not in settings:
         settings[x] = DEFAULTS[x]
   for x in ["bid", "pids", "learner", "ref"]:
      if x not in settings:
         raise Exception("enigma.models: Required setting '%s' not set!" % x)   
   if "hashing" not in settings:
      settings["hashing"] = 2**15
   if "results" not in settings:
      settings["results"] = {}
   if "ramdisk" not in settings:
      settings["ramdisk"] = None

def update(results, only=None, **others):
   if only:
      others["pids"] = only
   results.update(expres.benchmarks.eval(**others))

def loop(model, settings, nick=None):
   global RAMDISK_DIR

   check(settings)
   if nick:
      model = "%s/%s" % (model, nick)
   log.msg("Building model %s" % model)

   if settings["ramdisk"]:
      RAMDISK_DIR = os.path.join(settings["ramdisk"], "Enigma")
      os.system("mkdir -p %s" % RAMDISK_DIR)
      expres.results.RAMDISK_DIR = os.path.join(settings["ramdisk"], "00RESULTS")
      os.system("mkdir -p %s" % expres.results.RAMDISK_DIR)

   update(**settings)
   if not make(model, settings["results"], settings):
      raise Exception("Enigma: FAILED: Building model %s" % model)
   efun = settings["learner"].efun()
   new = [
      protos.solo(settings["pids"][0], model, mult=0, noinit=True, efun=efun),
      protos.coop(settings["pids"][0], model, mult=0, noinit=True, efun=efun)
   ]
   settings["pids"].extend(new)

   if settings["ramdisk"]:
      os.system("mkdir -p %s" % ENIGMA_ROOT)
      os.system("cp -rf %s/* %s" % (RAMDISK_DIR, ENIGMA_ROOT))
      os.system("rm -fr %s" % RAMDISK_DIR)
      RAMDISK_DIR = None
   
   update(only=new, **settings)

   if settings["ramdisk"]:
      os.system("mkdir -p %s" % expres.results.RESULTS_DIR)
      os.system("cp -rf %s/* %s" % (expres.results.RAMDISK_DIR, expres.results.RESULTS_DIR))
      os.system("rm -fr %s" % expres.results.RAMDISK_DIR)
      expres.results.RAMDISK_DIR = None
   
   log.msg("Building model finished\n")
   return new


