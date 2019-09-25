from setuptools import setup, find_packages

setup(name='enigmatic',
      version='0.2.1',
      description='Enigma: Automatic Inference Guiding Machine',
      url='http://github.com/ai4reason/enigmatic',
      author='ai4reason',
      license='GPL3',
      packages=find_packages(),
      #scripts=[
      data_files=[
         'bin/eprover',
         'bin/enigma-features',
         'bin/train',
         'bin/predict'
      ],
      install_requires=[
         'xgboost',
         'lightgbm',
         'pyprove==0.2'
      ],
      zip_safe=False)

