# **ENIGMA** #

Inference guiding machine for saturation-based Automated Theorem Provers.

## User Install

This is enough if you plan to use Enigma.  There is no need to clone this
repo manually.

```console
$ pip install enigmatic --user
```

This installs:

* our Python packages `pyprove` and `enigmatic`, and
* other Python dependencies (`xgboost`, `numpy`, ...), and
* statically compiled binaries for `eprover` with Enigma support, Enigma
  feature extractor `enigma-features`, and LIBLINEAR training and prediction
  binaries `train` and `predict`.

## Developers

You need this only if you plan to update the source codes.

1. Install `pyprove`:

   ```console
   $ git clone https://github.com/ai4reason/pyprove.git
   $ cd pyprove
   $ pip install -e . --user
   ```

2. Install `enigmatic`:

   ```console
   $ git clone https://github.com/ai4reason/enigmatic.git
   $ cd enigmatic
   $ pip install -e . --user
   ```

After this, the `git pull` command issued in both `pyprove` and `enigmatic`
directories will automatically update both Python packages to their latest
development versions from GitHub.

## Credits

Development of this software prototype was supported by: 

+ ERC Consolidator grant no. 649043 *AI4REASON*
+ ERC Starting grant no. 714034 *SMART*
+ FWF grant P26201
+ Cost Action CA15123 *EUTypes*

