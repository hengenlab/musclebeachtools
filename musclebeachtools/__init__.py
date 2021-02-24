import subprocess
try:
    process = subprocess.check_output(['git', 'describe'], shell=False)
    __git_version__ = rocess.strip().decode('ascii')
except Exception as e:
    __git_version__ = 'unknown'
from .mbt_neurons import *
from .mbt_spkinterface_out import *
