import os
from . import enigmap, pretrains, trains, protos
from pyprove import expres, log

ENIGMA_ROOT = os.getenv("ENIGMA_ROOT", "./Enigma")

DEFAULTS = {
   "gzip": True,
   "force": False,
   "hashing": None,
   "version": "VHSLC",
   "eargs": "--training-examples=3 -s",
   "cores": 4,
   "hash_debug": False,
}


def path(name, filename=None):
   if filename:
      return os.path.join(ENIGMA_ROOT, name, filename)
   else:
      return os.path.join(ENIGMA_ROOT, name)


def collect(name, rkeys, settings):
   version = settings["version"]
   force = settings["force"]
   cores = settings["cores"]
   hashing = settings["hashing"] if not settings["hash_debug"] else None

   f_dat = path(name, "train.%s" % ("in" if hashing else "pre"))
   if force or not os.path.isfile(f_dat):
      log.msg("+ extracting training data from results")
      pretrains.prepare(rkeys, version, force, cores, hashing)
      log.msg("+ collecting %s data" % ("training" if hashing else "pretrains"))
      pretrains.make(rkeys, out=file(f_dat, "w"), hashing=hashing)


def setup(name, rkeys, settings):
   os.system("mkdir -p %s" % path(name))
   f_pre = path(name, "train.pre")
   f_map = path(name, "enigma.map")
   f_log = path(name, "train.log")
   hashing = settings["hashing"]
  
   if os.path.isfile(f_map) and os.path.isfile(f_pre) and not settings["force"]:
      return enigmap.load(f_map) if not hashing else hashing
      
   if rkeys:
      collect(name, rkeys, settings)

   if hashing and not settings["hash_debug"]:
      file(f_map,"w").write('version("%s").\nhash_base(%s).\n' % (settings["version"], hashing))
      return hashing

   #if os.path.isfile(f_log):
   #   os.system("rm -f %s" % f_log)
   if settings["force"] or not os.path.isfile(f_map):
      log.msg("+ creating feature info")
      emap = enigmap.create(file(f_pre), hashing)
      enigmap.save(emap, f_map, settings["version"], hashing)
   else:
      if not hashing:
         emap = enigmap.load(f_map)

   return emap if not hashing else hashing


def make(name, rkeys, settings):
   learner = settings["learner"]

   f_pre = path(name, "train.pre")
   f_in  = path(name, "train.in")
   f_mod = path(name, "model.%s" % learner.ext())
   f_log = path(name, "train.log")

   if os.path.isfile(f_mod) and not settings["force"]:
      return True

   emap = setup(name, rkeys, settings)
   if not emap:
      os.system("rm -fr %s" % path(name))
      return False
   
   if settings["hash_debug"] or not settings["hashing"]:
      if settings["force"] or not os.path.isfile(f_in):
         log.msg("+ generating training data")
         trains.make(file(f_pre), emap, out=file(f_in, "w"))

   log.msg("+ training %s" % learner.name())
   tlog = file(f_log, "a")
   learner.build(f_in, f_mod, tlog)
   tlog.close()

   if settings["gzip"]:
      log.msg("+ compressing training files")
      os.system("cd %s; gzip -qf *.pre *.in *.out 2>/dev/null" % path(name))

   return True


def check(settings):
   if ("h" in settings["version"] and not settings["hashing"]) or \
      (settings["hashing"] and "h" not in settings["version"]):
         raise Exception("enigma.models: Parameter hashing must be set to the hash base (int) iff version contains 'h'.")   
   for x in DEFAULTS:
      if x not in settings:
         settings[x] = DEFAULTS[x]
   for x in ["bid", "pids", "learner"]:
      if x not in settings:
         raise Exception("enigma.models: Required setting '%s' not set!" % x)   
   if "results" not in settings:
      settings["results"] = {}


def update(settings, pids=None):
   if not pids:
      pids = settings["pids"]
   settings["results"].update(expres.benchmarks.eval(
      settings["bid"], pids, settings["limit"], cores=settings["cores"], 
      eargs=settings["eargs"], force=settings["force"]))


def loop(model, settings, nick=None):
   check(settings)
   if nick:
      model = "%s/%s" % (model, nick)
   log.msg("Building model %s" % model)

   update(settings)
   if not make(model, settings["results"], settings):
      raise Exception("Enigma: FAILED: Building model %s" % model)
   efun = settings["learner"].efun()
   new = [
      protos.solo(settings["pids"][0], model, mult=0, noinit=True, efun=efun),
      protos.coop(settings["pids"][0], model, mult=0, noinit=True, efun=efun)
   ]
   settings["pids"].extend(new)
   update(settings, new)
   
   log.msg("Building model finished\n")
   return new


