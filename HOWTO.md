# Enigmatic HOWTOs #

## Running Enigmatic ##

Most of the Enigmatic functions use Python's keyword parameters to setup experiments.
This is usefull to keep all the experiment setup arguments at one same place and simplify method calls.
Typically, you will have a dictionary with the parameters as follows.

```python
run = {
   "bid"       : "tptp/PUZ",
   "pids"      : ["mzr01", "mzr02"],
   "refs"      : ["mzr02"],
   "limit"     : "T5",
   "cores"     : 32,
   "features"  : "C(l,p,x,s,r,h,v,c,d,a):G:P",
   "learner"   : LightGBM(),
   "eargs"     : "--training-examples=3 -s --free-numbers",
   "dataname"  : "mytest"
}
```

Then, you run, for example, the eval/train loops as follows.

```python
from pyprove import log
from enigmatic import models
from enigmatic.learn.lgbooster import LightGBM

run = ... # see above

logger = log.logger("loop", **run)
models.loops(**run)
```

Don't forget to setup the logger, otherwise you might not see any output.
See below for `run` parameter description.

## Enigmatic Parameters ##

This is the list of `run` parameters supported by Enigmatic and their basic description.

| name | type | description |
| ---- | ---- | ----------- |
| `bid` | `str` | benchmark id: directory relative to `$PYPROVE_BENCHMARKS` (or `.`) |
| `pids` | `[str]` | strategy id list (filenames relative to `./strats` directory) |
| `limit` | `str` | evaluation time limit (use `T5` for `5` seconds) |
| `cores` | `int` | number of cores to use for evaluation (set `$OMP_NUM_THREADS` to control model building) |
| `eargs` | `str` | additional command line arguments for `eprover` |
| `refs` | `str` | reference strategies for model building |
| `features` | `str` | [Enigmatic features](https://github.com/ai4reason/eprover/blob/enigma/README.Enigmatic.md) to use when generating training data |
| `dataname` | `str` | custom experiment name |
| `iters` | `int` | number of iteration for model looping |
| `split` | `float` | ratio to divide train/test data |
| `forgets` | `(float, float)` | randomly forget `(neg, pos)` of training samples (can be `(None, None)`) |
| `balance` | `int` | automatically keep the `pos:neg` ratio close to `1:balance` |
| `options` | `[str]` | option flags |
| `debug` | `[str]` | debugging flags |

### `options`: Option flags ###

The option flags are recognized.

| flag | description |
| - | - |
| `headless` | do not use progress bars |
| `loop-coop-only` | use `coop` strategies when looping (no `solo` strategies) |


### `debug`: Debugging flags ###

The following flags are recognized.

| flag | description |
| - | - |
| `acc` | compute train/test model accuracies (use together with `split`) |
| `train` | keep separate uncompressed train vectors for each problem file in `00TRAINS` |
| `nozip` | do not compress training data |
| `force` | do not use stored files and recompute everything |





## Tuning LightGBM model parameters ##

Enigmatic provides automatic LightGBM model parameter tuner, implemented using `optuna`, motivated by `optuna`'s LightGBM integration.
It can be used either manually (if you already have a training data), or automatically by Enigmatic model building methods like `models.build`.

### Manual usage ###

If you have train and test vectors already generated in `train.in` and `test.in` files, and you want to find a model with the best testing accuracy, use `enigmatic.lgbtune` module as follows.

```python
from pyprove import log
from enigmatic import lgbtune

logger = log.logger("tune")
lgbtune.lgbtune("train.in", "test.in", timeout=3600)
```

This will run for `3600` seconds and it will create bunch of models in a temporary directory (optional argument `d_tmp` defaults to `./optuna-tmp`).
In the end you will see an output like this:

```
Best model params: {'num_leaves': 512, ...}
Best model accuracy: 72.73% (89.74% / 64.63%)
Best model file: optuna-tmp/min_data/model0002.lgb
```

Other arguments to control the tunning are available, the same as for the automated usage described below.

### Compresssed data ###

If you have Enigmatic-compatible compressed data files like `train.in-data.npz` and `train.in-label.npz` you can also use them as well.
It is, indeed, highly recommended in order to speed up data loading.
But note that in the case of compressed data, you still pass just `train.in` and `test.in` to `lgbtune`.
It will automatically recognize compressed data.
The compressed files must be named as above (suffixes `*-data.npz` and `*-label.npz`).

To compress an SVM-Light text file `train.in` use `enigmatic.trains.compress` as follows.

```python
from pyprove import log
from enigmatic import trains

logger = log.logger("compress")
trains.compress("train.in")
```


























