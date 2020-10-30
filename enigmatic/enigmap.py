import os
import subprocess
import logging
from . import models

logger = logging.getLogger(__name__)

def make(f_map, features):
   logger.debug("- writing features map for %s to %s" % (features, f_map))
   args = [
      "enigmatic-features", 
      "--features=%s" % features,
      "--output-map=%s" % f_map
   ]
   try:
      subprocess.run(args)
   except Exception as e:
      print(e)
      pass

def default_path(**others):
   return models.path(**others)

def build(features, debug=[], path=default_path, **others):
   d_map = path(features=features, debug=debug, **others)
   f_map = os.path.join(d_map, "enigma.map")
   os.system('mkdir -p "%s"' % os.path.dirname(f_map))
   if os.path.isfile(f_map) and not "force" in debug:
      logger.debug("- skipped writing map %s" % f_map)
      return
   make(f_map, features)

def load(**others):
   f_map = models.pathfile("enigma.map", **others)
   with open(f_map) as f:
      lines = f.read().strip().split("\n")
      features = lines[0].lstrip("features(").rstrip(").")
      count = int(lines[1].lstrip("count(").rstrip(")."))
   return dict(features=features, count=count, lines=lines)

