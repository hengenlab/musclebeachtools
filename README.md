# $\textcolor{#fa8072}{\textbf{Musclebeachtools}}$

<!-- 
# title  color  #fa8072
# header color  #6897bb
# warnings color #ff4040
# best color #b4eeb4
-->

---
This package used to analyze neuronal data, from basic plotting
 (ISI histograms, firing rate, cross correlation),
 to hard core computational work (ising models, branching ratios, etc).  

---
![Tests](https://github.com/hengenlab/musclebeachtools/actions/workflows/pytests.yml/badge.svg)


## $\textcolor{#6897bb}{\textbf{Installation}}$


#### $\textcolor{#81d8d0}{\textbf{Download and install musclebeachtools}}$

```
cd  ~/git/
git clone https://github.com/hengenlab/neuraltoolkit.git
git clone https://github.com/hengenlab/musclebeachtools.git  
```

```
# cd locationofneuraltoolkit/neuraltoolkit/  
# For example /home/kbn/git/neuraltoolkit
pip install .
```

```
# cd locationofmusclebeachtools/musclebeachtools/  
# For example /home/kbn/git/musclebeachtools
pip install .
```

```
# In Linux and windows
pip install xgboost

# In mac, install brew,gcc then install xgboost
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew install gcc
conda install -c conda-forge xgboost
```

<!--
---
### Installation by adding to path (not recommended)

##### Windows
```
My Computer > Properties > Advanced System Settings > Environment Variables >  
In system variables, create a new variable  
    Variable name  : PYTHONPATH  
    Variable value : location where musclebeachtools is located  
    Click OK
```

##### Linux
```
If you are using bash shell  
In terminal open .barshrc or .bash_profile  
add this line  
export PYTHONPATH=/location_of_musclebeachtools:$PYTHONPATH
```

##### Mac
```
If you are using bash shell  
In terminal cd ~/  
then open  .profile using your favourite text editor  
add this line  
export PYTHONPATH=/location_of_musclebeachtools:$PYTHONPATH
```
-->
---

#### $\textcolor{#81d8d0}{\textbf{Test import}}$

```
# Open powershell/terminal     
ipython
```
```
import musclebeachtools as mbt   
```
---


<!--
#### Load data from ntksortingb
```
import musclebeachtools as mbt

datadir = "/hlabhome/kiranbn/neuronclass/t_lit_EAB50final/"
n1 = mbt.ksout(datadir, filenum=0, prbnum=4, filt=[])
```
-->

## $\textcolor{#6897bb}{\textbf{Usage}}$

#### $\textcolor{#81d8d0}{\textbf{Load neurons group file (from spikeinteface output) and get basic properties}}$
```
import numpy as np
import musclebeachtools as mbt

neurons = np.load('neurons_group0.npy', allow_pickle=True)

# Get basic info of a neuron, here 6th in list neurons
print(neurons[6])
# Neuron with (clust_idx=6, quality=1, peak_channel=6)

# Get sampling rate for 4th neuron
neurons[4].fs

# Get sample time for 4th neuron
neurons[4].spike_time

# Get spike time in seconds for 4th neuron
neurons[4].spike_time_sec

# Get spike time in seconds for 4th neuron from on off times
neurons[4].spike_time_sec_onoff
```

#### $\textcolor{#81d8d0}{\textbf{Get a test neurons group file in case you do not have one}}$
```
import requests
import io
import numpy as np
import musclebeachtools as mbt

response = requests.get('https://biolinux2.wustl.edu/hlabdata/data/test_neuron.npy')
neurons = np.load(io.BytesIO(response.content), allow_pickle=True)
# neurons = np.load('test_neuron.npy', allow_pickle=True)

# Get basic info of a neuron, first neuron
print("\nBasics ", neurons[0])

# Get sampling rate for first neuron
print("\nSampling rate, fs ", neurons[0].fs)

# Get sample time for first neuron
print("\nSample times ", neurons[0].spike_time)

# Get spike time in seconds for the first neuron
print("\nSpike times in second ", neurons[0].spike_time_sec)
```
![checkqual](https://biolinux2.wustl.edu/hlabdata/images/mbt_loadtestneuron.png)

#### $\textcolor{#81d8d0}{\textbf{Find start and end time of the 4th neuron}}$
```
neurons[4].start_time
# 0
# 

neurons[4].end_time
# 14400.0
# 14400.0/(60*60) = 4
# 4 hours of data sorted
```

$\textcolor{#a0db8e}{\textbf{How many hours of data sorted}, end time - start time}$

#### $\textcolor{#81d8d0}{\textbf{Find start and end time of the 4th neuron}}$
```
neurons[4].rstart_time
# 2022-12-31_11-15-55
# Date and time in the filename of the first binary file used for sorting

neurons[4].rend_time
# 2022-12-31_15-10-55
# Date and time in the filename of the last binary file used for sorting
```


$\textcolor{#a0db8e}{\textbf{If the binary file contains 5 minutes of data, you must add those 5 minutes to rend time to calculate the true rend time;}}$
$\textcolor{#a0db8e}{\textbf{however, if the last binary file is shorter, add its actual length instead.}}$


#### $\textcolor{#81d8d0}{\textbf{Find ecube start and rend time of the 4th neuron}}$
```
neurons[4].estart_time
# np.int64(43228511909775)

neurons[4].eend_time
# np.int64(57628511869775)
# (np.int64(57628511869775) - np.int64(43228511909775))/ (60* 60 * 10**9) = 3.9 hours
```

$\textcolor{#a0db8e}{\textbf{If the binary file contains 5 minutes of data, you must add those 5 minutes in nano seconds to eend time to calculate the true eend time;}}$
$\textcolor{#a0db8e}{\textbf{however, if the last binary file is shorter, add its actual length instead.}}$

```
((np.int64(57628511869775)+ np.int64(300 * 10**9) ) - np.int64(43228511909775))/ (60* 60 * 10**9)
#  np.float64(4.083333322222222)# hours
```

#### $\textcolor{#81d8d0}{\textbf{Find region, birthday, sex and species of the 4th neuron}}$
```
neurons[4].region
# 'ACC'

neurons[4].birthday
# datetime.datetime(2022, 9, 20, 7, 30)

neurons[4].sex
# 'f'

neurons[4].species
# 'mouse'
```
#### $\textcolor{#81d8d0}{\textbf{Find quality of the 4th neuron}}$

```
neurons[4].quality
neurons[4].qual_prob
# qual : Quality values should be 1, 2, 3 or 4
#        1 : Good
#        2 : Good but some contamination
#        3 : Multiunit contaminated unit
#        4 : Noise unit
# qual_prob array with probabilities for each quality from autoqual.
```

#### $\textcolor{#81d8d0}{\textbf{Check total number of neurons and check quality}}$
```
import numpy as np
import musclebeachtools as mbt
# Load neurons
neurons = \
    np.load('H_2020-04-09_09-11-37_2020-04-10_01-06-37_neurons_group0.npy',
            allow_pickle=True)
print(f'Total number of neurons {len(neurons)}')
q1 = sum( 1 for neuron in neurons if neuron.quality == 1)
q2 = sum( 1 for neuron in neurons if neuron.quality == 2)
q3 = sum( 1 for neuron in neurons if neuron.quality == 3)
print(f'Quality 1 has {q1} neurons and 2  has {q2} neurons and quality 3 has {q3} neurons')
```


#### $\textcolor{#81d8d0}{\textbf{Find other properties of the 4th neuron}}$
```
neurons[4].peak_channel

neurons[4].waveform
neurons[4].waveform_tetrodes
neurons[4].waveforms

neurons[4].mean_amplitude

neurons[4].cell_type
neurons[4].peaklatency

neurons[4].clust_idx
neurons[4].spike_time_sec_onoff
```
---


#### $\textcolor{#81d8d0}{\textbf{Plot mean waveform of 4th neuron}}$
```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

neurons[4].plot_wf()
```
---

#### $\textcolor{#81d8d0}{\textbf{Check on off times of 4th neuron}}$
```
neurons[4].on_times
# [0.0]

neurons[4].off_times
# [14400.0]
```

#### $\textcolor{#81d8d0}{\textbf{Set on off times of 4th neuron}}$
```
# If we find we have noise in recording from hour 1 to 2
#
# This means from neurons[0].start_time or 0 seconds to 3600 seconds (hour 1) data is good
# Also from 7200 seconds (hour 2) to end of the recording neurons[0].end_time data is good
#
# This can be set using set_onofftimes_from_list function like this
neurons[4].set_onofftimes_from_list([neurons[0].start_time, 7200],
                                    [3600,  neurons[0].end_time])

# To verify this is done correctly print on off times
print(neurons[4].on_times)
# [0.0, 7200]
print(neurons[4].off_times)
# [3600, 14400.0]
```

#### $\textcolor{#81d8d0}{\textbf{Plot isi of 4th neuron}}$
```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

#  start : Start time (default False uses self.start_time)
#  end : End time (default False uses self.end_time)
#  isi_thresh : isi threshold (default 0.1)
#  nbins : Number of bins (default 101)
#  lplot : To plot or not (default lplot=1, plot isi)
#  lonoff : Apply on off times (default on, 1)
neurons[4].isi_hist(start=False, end=False, isi_thresh=0.1, nbins=101,
                    lplot=1, lonoff=1)
```

$\textcolor{#a0db8e}{\textbf{Here isi thresh, start, end are in seconds}}$

---

#### $\textcolor{#81d8d0}{\textbf{Plot firing rate of 4th neuron}}$

```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

# binsz : Bin size (default 60)
#  start : Start time (default False uses self.start_time)
#  end : End time (default False uses self.end_time)
# lplot : To plot or not (default lplot=1, plot firing rate)
# lonoff : Apply on off times (default on, 1)
neurons[4].plotFR(binsz=60, start=False, end=False,
                  lplot=1, lonoff=1)
```
$\textcolor{#a0db8e}{\textbf{Here binsz, start, end are in seconds}}$

---

#### $\textcolor{#81d8d0}{\textbf{calculate Fano factor of the 4th neuron}}$

```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

# binsz : Bin size (default 0.05 (50ms))
# start : Start time (default False uses self.start_time)
# end : End time (default False uses self.end_time)
# lonoff : Apply on off times (default on, 1)
#
# returns : fano factor
neurons[4].calculate_fano_factor(binsz=0.05, start=False, end=False,
                                 lonoff=1)

```
$\textcolor{#a0db8e}{\textbf{Here binsz, start, end are in seconds}}$

---

#### $\textcolor{#81d8d0}{\textbf{Calculate coefficient of variation (cv) of the ISIs of the 4th neuron}}$
```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

# start : Start time (default False uses self.start_time)
# end : End time (default False uses self.end_time)
# lonoff : Apply on off times (default on, 1)
#
# returns : coefficient of variation (cv) of the ISIs
neurons[4].calculate_cv(start=False, end=False, lonoff=1)

```
$\textcolor{#a0db8e}{\textbf{Here start, end are in seconds}}$

---

#### $\textcolor{#81d8d0}{\textbf{Calculate presence ratio of the 4th neuron}}$

```
import numpy as np
import musclebeachtools as mbt

# Load neurons
neurons = np.load('neurons_group0.npy', allow_pickle=True)

# nbins : Number of bins (default 101)
# start : Start time (default False uses self.start_time)
# end : End time (default False uses self.end_time)
# lonoff : Apply on off times (default on, 1)
neurons[4].presence_ratio(nbins=101, start=False, end=False,
                          lonoff=1)
```
$\textcolor{#a0db8e}{\textbf{Here start, end are in seconds}}$

---

#### $\textcolor{#81d8d0}{\textbf{Calculate isi contamination at various thresholds, 2 and 4 ms of the 4th neuron}}$

```
# cont_thresh_list : threshold lists for calculating isi contamination
# time_limit : count spikes upto, default np.inf. Try also 100 ms, 0.1
# start : Start time (default self.start_time)
# end : End time (default self.end_time)
# lonoff : Apply on off times (default on, 1)
neurons[4].isi_contamination(cont_thresh_list=[0.002, 0.004], time_limit=np.inf)
                             start=False, end=False, lonoff=1)
```
 $\textcolor{#a0db8e}{\textbf{Here binsz, start, end are in seconds}}$

---

### Check quality and its probability from autoqual (see below).
```
print(neurons[2].quality, "", neurons[2].qual_prob[neurons[2].quality - 1])
```
---

#### Change quality of neuron neurons[0] to 1
```
# qual : Quality values should be 1, 2, 3 or 4
#        1 : Good
#        2 : Good but some contamination
#        3 : Multiunit contaminated unit
#        4 : Noise unit
# qual_prob array with probabilities for each quality from autoqual.
# When quality is assigned manualy qual_prob is set to 100% for that quality
neurons[0].set_qual(1)
```
---

#### $\textcolor{#81d8d0}{\textbf{Get spiketimes from all neurons as a list}}$
```
import numpy as np
import musclebeachtools as mbt


neurons = np.load('neurons_group0.npy', allow_pickle=True)

# neurons : List of neurons
# start : Start time (default (False uses self.start_time))
# end : End time (default (False uses self.end_time))
# lonoff : Apply on off times (default on, 1)
spiketimes_list = mbt.n_getspikes(neurons, start=False, end=False, lonoff=1)
```
---


#### $\textcolor{#81d8d0}{\textbf{Get spikewords from all neurons in neurons as a list}}$

$\textcolor{#b4eeb4}{\textbf{Here binsz, start, end are in seconds}}$
```
import numpy as np
import musclebeachtools as mbt


neurons = np.load('neurons_group0.npy', allow_pickle=True)

# neurons : List of neurons
# binsz : Bin size (default 0.02 (20 ms))
# start : Start time (default (False uses self.start_time))
# end : End time (default (False uses self.end_time))
# binarize : Get counts (default 0) in bins,
#   if binarize is 1, binarize to 0 and 1
# lonoff : Apply on off times (default on, 1)
sw = mbt.n_spiketimes_to_spikewords(neurons, binsz=0.02,
                                    start=False, end=False,
                                    binarize=0, lonoff=1)
```

```
import numpy as np
import musclebeachtools as mbt


neurons = np.load('neurons_group0.npy', allow_pickle=True)
# Filter by quality
neuron_list_filt = mbt.n_filt_quality(neurons, maxqual=[1, 2])

# neuron_list_filt : List of neurons filtered by quality
# binsz : Bin size (default 0.02 (20 ms))
# start : Start time (default (False uses self.start_time))
# end : End time (default (False uses self.end_time))
# binarize : Get counts (default 0) in bins,
#   if binarize is 1, binarize to 0 and 1
# lonoff : Apply on off times (default on, 1)
sw = mbt.n_spiketimes_to_spikewords(neuron_list_filt, binsz=0.02,
                                    start=False, end=False,
                                    binarize=0, lonoff=1)
```

---


#### $\textcolor{#81d8d0}{\textbf{Change quality with plot of ISI, Firingrate, Waveform and amplitude}}$
```
import numpy as np
import musclebeachtools as mbt
neurons = np.load('neurons_group0.npy', allow_pickle=True)

# binsz : Bin size (default 3600)
# start : Start time (default self.start_time)
# end : End time (default self.end_time)
# lsavepng : Save checkqual results as png's
# png_outdir : Directory to save png files
#              if lsavepng=1 and png_outdir=None
#              png's will be saved in current working directory
# fix_amp_ylim : default 0, yaxis max in amplitude plot.
#                For example can be fix_amp_ylim=500 to see from 0 to 500
#                in amplitude plot.
neurons[2].checkqual(binsz=3600, start=False, end=False, lsavepng=0,
               png_outdir=None, fix_amp_ylim=0)
```
![checkqual](https://biolinux2.wustl.edu/hlabdata/images/mbt_checkqual.png)


#### Check quality is changed also there is a log from checkqual
```
print(neurons[2].quality)
```



#### Find quality using xgboost
```
import numpy as np
import musclebeachtools as mbt
# Load neurons
neurons = \
    np.load('H_2020-04-09_09-11-37_2020-04-10_01-06-37_neurons_group0.npy',
            allow_pickle=True)

# Find quality
# neuron_list : List of neurons from (usually output from ksout)
# model_file : model file with path
mbt.autoqual(neurons, '/media/HlabShare/models/xgb_model')
```




#### $\textcolor{#81d8d0}{\textbf{Filter neuron list by quality}}$

````
import numpy as np
import musclebeachtools as mbt
# Load neurons
neurons = \
    np.load('H_2020-04-09_09-11-37_2020-04-10_01-06-37_neurons_group0.npy',
            allow_pickle=True)
print(f'Total number of neurons {len(neurons)}')
q1 = sum( 1 for neuron in neurons if neuron.quality == 1)
q2 = sum( 1 for neuron in neurons if neuron.quality == 2)
q3 = sum( 1 for neuron in neurons if neuron.quality == 3)
print(f'Quality 1 has {q1} neurons and 2  has {q2} neurons and quality 3 has {q3} neurons')

# Filter to have quality 1 and 2 only
neuron_list_filt = mbt.n_filt_quality(neurons, maxqual=[1, 2])

q1 = sum( 1 for neuron in neurons if neuron.quality == 1)
q2 = sum( 1 for neuron in neurons if neuron.quality == 2)
q3 = sum( 1 for neuron in neurons if neuron.quality == 3)

print(f'Quality 1 has {q1} neurons and 2  has {q2} neurons and quality 3 has {q3} neurons')
````

#### Verify quality is correct using checkqual
```
# lsavepng : Save checkqual results as png's, default 0
# png_outdir : Directory to save png files
#              if lsavepng=1 and png_outdir=None
#              png's will be saved in current working directory
# fix_amp_ylim=0 : Default off
# fix_amp_ylim=500, to change amplitude plot ylim from 0 to 500.
for neuron in neurons:
    neuron.checkqual(savepng=0, png_outdir=None, fix_amp_ylim=500)
```
---

#### Remove spikes for neuron with large amplitudes
```
# Default method based on standard deviation,
For example for neurons of quality 1 and 2,  1.5 to 4.5 standard deviation is fine.
But quality 3 neurons make sure to use larger deviations or check manualy with lplot=True
as it contains multiple nuerons/MUA.

neurons[4].remove_large_amplitude_spikes(1.5, lstd_deviation=True, start=False, end=False, lplot=True)

# Based on threshold value, for example 1000
neurons[4].remove_large_amplitude_spikes(1000, lstd_deviation=False, start=False, end=False, lplot=True)
If you are sure and do not want to check plots
 to confirm change lplot=False (not recommended)

# Based on np percentile, 98
# lpercentile : default False,  if True, turns off lstd_deviation
neurons[4].remove_large_amplitude_spikes(98, lstd_deviation=False, lpercentile=True, start=False, end=False, lplot=True)
If you are sure and do not want to check plots
 to confirm change lplot=False (not recommended)

# Save the modified neuron list
mbt.n_save_modified_neuron_list(neurons, '/home/kbn/neuron_mod.npy')
```
---

#### Sort neurons by peak_channel in ascending order
```
import numpy as np
import musclebeachtools as mbt
# Load neurons
neurons = \
    np.load('H_2020-04-09_09-11-37_2020-04-10_01-06-37_neurons_group0.npy',
            allow_pickle=True)
neurons = sorted(neurons, key=lambda i: i.peak_channel)
```
---


#### Shuffle spike times
```
import numpy as np
import musclebeachtools as mbt
import os.path as op

# get new file name
fl = '/media/ckbn/H_2022-05-12_08-59-26_2022-05-12_17-54-28_neurons_group0.npy'
print(op.splitext(fl)[0]+'_shuffle_spiketimes.npy')
fl_new = op.splitext(fl)[0]+'_shuffle_spiketimes.npy'

# load neuons and shuffle times and save
neurons = np.load(fl, allow_pickle=True)
for indx, neuron in enumerate(neurons):
    print("indx ",indx)
    neuron.shuffle_times(shuffle_alg=1)
np.save(fl_new, neurons)
```
```
# to test whether everything worked
# check quality of some neurons
neurons = np.load(fl, allow_pickle=True)
for indx, neuron in enumerate(neurons):
        neuron.checkqual()
print("shuffled")
neurons_new = np.load(fl_new, allow_pickle=True)
for indx, neuron_new in enumerate(neurons_new):
        neuron_new.checkqual()
```


#### Add behavior to neurons
```
# behavior is a 2d array with
# first one with time in each state and
# second one with behavioral states (sleep states),
#
# As Sleepstates can be modified using Sleep_Wake_Scoring
# please remember to load these everytime in case they are modified.
behavior = np.load('Sleepstates.npy', allow_pickle=True)
mbt.Neuron.update_behavior(behavior))

# To get behavior
# tolerance : 2 (default)
#    check difference between end_time and behavior length
#    is close to tolerance for binsize
#
# binsize : 4 (default) binsize used for behavior
#     4 here means 4 second bins
#
neurons[0].get_behavior(tolerance=2, binsize=4)
# if behavior is not loaded using above step (using Sleepstates.npy)
# it will raise error ValueError
#
# if behavior length is not close to binsize w.r.t neurons[0].end_time
# it will raise error ValueError
                      
```
---

#### Plot all waveforms in a neuron list
```
# maxqual filter by quality, list
# plot waveforms for neurons with quality in maxqual
# To see plot
mbt.n_plot_neuron_wfs(neurons, maxqual=[1, 2, 3, 4], pltname="block1")
# To save plot
mbt.n_plot_neuron_wfs(neurons, maxqual=[1], pltname="block1",
                      saveloc='/home/kbn/')
```
---

#### Plot all neurons checkqual (figure) and save as pdf
```
fl = '/home/kbn/probe1/co/H_2023-12-29_23-09-54_2023-12-30_11-04-55_neurons_group0.npy'

# Load neurons
neurons = np.load(fl, allow_pickle=True)

# def n_checkqual_pdf(neurons, savepdf, maxqual=None,
#                     binsz=3600, start=False, end=False,
#                     fix_amp_ylim=1):
#
#     neurons : List of neurons
#     savepdf : filename with path to save pdf
#     maxqual : default [1, 2, 3, 4], filter by quality,
#               neuron.quality in maxqual

#     binsz : Bin size (default 3600)
#     start : Start time (default self.start_time)
#     end : End time (default self.end_time)
#     fix_amp_ylim : default 1, yaxis max in amplitude plot.
#                    For example can be fix_amp_ylim=500 to see from 0 to 500
#                    in amplitude plot.
savepdf = '/home/kbn/probe1/co/H_2023-12-29_23-09-54_2023-12-30_11-04-55_neurons_group0_checkqual.pdf'
# call n_checkqual_pdf to save as savepdf
mbt.n_checkqual_pdf(neurons, savepdf, maxqual=None,
                    binsz=3600, start=False, end=False,
                    fix_amp_ylim=1)
```
---


#### check two neurons are from same tetrode
```
# check_sametetrode_neurons(channel1, channel2,
                            ch_grp_size=4,
                            lverbose=1)
# channel1: first channel
# channel2: second channel
# ch_grp_size : default (4) for tetrodes
# lverbose: default(1) to print tetrodes check
#
# return True or False
lsamechannel = \
    check_sametetrode_neurons(neurons[0].peak_channel,
                              neurons[1].peak_channel,
                              ch_grp_size=4,
                              lverbose=1)
```
---


#### To create neuron list from sorter_interface output
You need to be in sorter_interface conda environment (spike15)
```
import numpy as np
import glob
import musclebeachtools as mbt
from datetime import datetime
co_dir = '/home/kbn/co/'
model_file = '/media/HlabShare/models/xgboost_autoqual_prob'
neurons = mbt.mbt.mbt_sorter_interface_out(co_dir,
                             model_file=model_file,
                             sex='m',
                             birthday=datetime(1970, 1, 1, 00, 00),
                             species='m',
                             animal_name='ABC12345',
                             region_loc='CA1',
                             genotype='te4',
                             expt_cond='monocular deprivation')
neurons[0].species
# 'm'

neurons[0].region
# 'CA1'

neurons[0].animal_name
# 'ABC12345'

neurons[0].sex
# 'm'

neurons[0].birthday
# datetime.datetime(1970, 1, 1, 0, 0)

neurons[0].genotype
# 'te4'

neurons[0].expt_cond
# 'monocular deprivation'
```
---

#### Add neurons
```
import numpy as np
import musclebeachtools as mbt

neurons = np.load('neuron_add.npy', allow_pickle=True)

new_neuron = neurons[51] + neurons[61]
neurons[51].checkqual()
neurons[61].checkqual()
new_neuron.checkqual()

# If you agree merge is correct
print("Length ", len(neurons))
# delete neurons used to merge
neurons = np.delete(neurons, [51, 61])
print("Length ", len(neurons))
# add merged new merged neuron
neurons = np.append(neurons, new_neuron)
print("Length ", len(neurons))
```
---

```
# branching ratio from Viola Priesemann
import numpy as np
import musclebeachtools as mbt

neurons = np.load('H_2020-12-17_13-19-30_2020-12-18_01-14-32_neurons_group0.npy', allow_pickle=True)
# filer by quality
neuron_list = mbt.n_filt_quality(neurons, maxqual=[1, 2])

# start, end are starting time and end time for br calculation, in hours
# ava_binsz binsize for branching ration, in seconds
# kmax  150-2500, remember in ms multiply ava_binsz
# plotname None no figures
# br1 branching ratio with complex fit
# br2 branching ratio with exp_offset fit
# acc1 is pearson correlation coefficient with respect to data for complex (experimental not tested fully)
# acc2 is pearson correlation coefficient with respect to data for exp_offset (experimental not tested fully)
br1, br2, acc1, acc2 = mbt.n_branching_ratio(neurons, ava_binsz=0.004,
                                kmax=500,
                                start=0, end=2,
                                binarize=1,
                                plotname='/home/kbn/hhh.png')
print("Branching ratio ", br1, " pearson corr ", acc1, flush=True)



# find keys
import numpy as np
import glob
import musclebeachtools as mbt

npath = '/media/kbn/results/'
fl = np.sort(glob.glob(npath + '*neurons_group0.npy'))
for indx, fl_file in enumerate(fl):
    print("indx ", indx, " ", fl_file)
mbt.track_blocks(fl, ch_grp_size=4, maxqual=3, corr_fact=0.97, lsaveneuron=1, lsavefig=1)

# For 4th neuron, by neuron.clust_idx (old)
n_amp = mbt.load_spike_amplitudes(n, '/home/kbn/amplitudes0.npy')
n_amp[4].spike_amplitude


# Correlograms
import numpy as np
import musclebeachtools as mbt
fl = '/media/ckbn/H_2022-05-12_08-59-26_2022-05-12_17-54-28_neurons_group0.npy'
neurons = np.load(fl, allow_pickle=True)
# Autocorr
# default savefig_loc=None, to show plot
# savefig_loc='/home/kbn/fig_crosscorr.png' to save fig
neurons[1].crosscorr(friend=None, savefig_loc=None)
# Crosscorr
neurons[10].crosscorr(friend=neurons[11], savefig_loc=None)


# zero cross corr
import numpy as np
import musclebeachtools as mbt
fl = 'H_2020-08-07_14-00-15_2020-08-08_01-55-16_neurons_group0.jake_scored_q12.npy'
neurons = np.load(fl, allow_pickle=True)
zcl_i, zcl_v, zcl_mse = mbt.n_zero_crosscorr(neurons)
print(zcl_i, "\n", zcl_v)
```


#### $\textcolor{#81d8d0}{\textbf{Check validity of neurons group0 files}}$
```
import numpy as np
import musclebeachtools as mbt
import glob

main_dir = '/media/HlabShare/Clustering_Data/CAF00080/'
neurons_group0_files_list = \
    sorted(glob.glob(main_dir + '/*/*/*/*/H*neurons_group0*'))
mbt.n_check_date_validity(neurons_group0_files_list,
                          surgeryday_time_string='2021-02-04_07-30-00',
                          sacday_time_string='2021-03-08_07-30-00',
                          birthday_time_string='2020-04-19_07-30-00')
```
---


#### Calculates the group value of a channel
```
import numpy as np
import musclebeachtools as mbt

mbt.get_group_value(channel, group_size)
# Calculates the group value of a channel
#  by finding the remainder when divided by group_size.
#
# channel (int): The channel number.
# group_size (int): The number of groups.
#
# group_size must be greater than 0.
#
# Returns
#     int: The group value (0 to group_size - 1).
```
---


## $\textcolor{#6897bb}{\textbf{FAQ}}$
```
1. spike_time vs spike_time_sec
Property spike_time is in sample times.
To to get spike time in seconds
please use spike_time_sec, neurons[4].spike_time_sec
or
For example for 4th neuron neurons[4].spike_time/n1[4].fs
Also spike_time_sec_onoff filters spike_time_sec based on on/off times

```
---
## Issues

```Please slack Kiran ```
---
