from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='musclebeachtools',
   version='1.0',
   description='This package used to analyze neuron data, from basic plotting\
               (ISI histograms, firing rate, cross corr), to hard core\
               computational work (ising models, branching ratios, etc).',
   license="",
   long_description=long_description,
   keywords='musclebeachtools, neuroscience, electrophysiology',
   package_dir={'musclebeachtools': 'musclebeachtools'},
   author='Keith Hengen, Sahara Ensley, Kiran Bhaskran-Nair\
           (Hengen Lab Washington University in St. Louis)',
   author_email='',
   maintainer='Kiran Bhaskaran-Nair, Sahara Ensley, Keith Hengen,\
           (Hengen Lab Washington University in St. Louis)',
   maintainer_email='',
   url="https://github.com/hengenlab/musclebeachtools",
   download_url="https://github.com/hengenlab/musclebeachtools",
   packages=['musclebeachtools'],
   install_requires=['ipython', 'numpy', 'matplotlib', 'seaborn', 'pandas',
                     'scipy'],
   classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
   scripts=[
           ]
)
