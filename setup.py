from setuptools import setup
import subprocess

with open("README.md", 'r') as f:
    long_description = f.read()

try:
    process = subprocess.check_output(['git', 'describe'], shell=False)
    __git_version__ = process.strip().decode('ascii')
except Exception as e:
    print(e)
    __git_version__ = 'unknown'
print("__git_version__ ", __git_version__)
try:
    with open('musclebeachtools/_version.py', 'w') as fp:
        fp.write("__git_version__ = '" + str(__git_version__) + "'")
except Exception as e:
    print("Error : ", e)

setup(
   name='musclebeachtools',
   version='1.2.0',
   description='This package used to analyze neuron data, from basic plotting\
               (ISI histograms, firing rate, cross corr), to hard core\
               computational work (ising models, branching ratios, etc).',
   license="",
   keywords='musclebeachtools, neuroscience, electrophysiology',
   package_dir={'musclebeachtools': 'musclebeachtools'},
   author='Keith Hengen, Sahara Ensley, Kiran Bhaskaran-Nair\
           (Hengen Lab Washington University in St. Louis)',
   author_email='',
   maintainer='Kiran Bhaskaran-Nair, Sahara Ensley, Keith Hengen,\
           (Hengen Lab Washington University in St. Louis)',
   maintainer_email='',
   url="https://github.com/hengenlab/musclebeachtools",
   download_url="https://github.com/hengenlab/musclebeachtools",
   packages=['musclebeachtools'],
   install_requires=[
    'ipython', 'numpy', 'matplotlib', 'seaborn', 'pandas',
    'joblib', 'scipy', 'scikit-learn',
    'mrestimator',
    'neuraltoolkit',
    # 'mrestimator@git+https://github.com/Priesemann-Group/mrestimator.git',
    # 'neuraltoolkit@git+https://github.com/hengenlab/neuraltoolkit.git',
    'xgboost'],
   # not added as may be users have their own changes in neuraltoolkit
   # 'neuraltoolkit@git+https://github.com/hengenlab/neuraltoolkit.git'],
   classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
   scripts=[
           ]
)
