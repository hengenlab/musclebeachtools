# musclebeachtools
---
This package used to analyze neuronal data, from basic plotting
 (ISI histograms, firing rate, cross correlation),
 to hard core computational work (ising models, branching ratios, etc).  

---
![Tests](https://github.com/hengenlab/musclebeachtools/actions/workflows/pytests.yml/badge.svg)


## Installation

### Download musclebeachtools
```
git clone https://github.com/hengenlab/musclebeachtools.git   
```
### Using pip
```
cd locationofmusclebeachtools/musclebeachtools/  
For example /home/kbn/git/musclebeachtools  
pip install .

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
### Test import
```
Open powershell/terminal     
    ipython    
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

## Usage

#### Load neurons group file (from spikeinteface output) and get basic properties
```
import numpy as np
import musclebeachtools as mbt

n = np.load('neurons_group0.npy', allow_pickle=True)

# Get basic info of a neuron, here 6th in list n1
print(n1[6])
# Neuron with (clust_idx=6, quality=1, peak_channel=6)

# Get sampling rate for 4th neuron
n1[4].fs

# Get sample time for 4th neuron
n1[4].spike_time

# Get spike time in seconds for 4th neuron
n1[4].spike_time_sec

# Get spike time in seconds for 4th neuron from on off times
n1[4].spike_time_sec_onoff
```

#### Get a test neurons group file in case you do not have one.
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

#### Other properties and functions
```
n1[4].start_time
n1[4].end_time
n1[4].on_times
n1[4].off_times
n1[4].peak_channel
n1[4].quality
n1[4].qual_prob
n1[4].region
n1[4].birthday
n1[4].sex
n1[4].species
n1[4].region
n1[4].waveform
n1[4].waveform_tetrodes
n1[4].waveforms
n1[4].mean_amplitude
n1[4].cell_type
n1[4].peaklatency
n1[4].clust_idx

# Plot mean waveform of 4th neuron
n1[4].plot_wf()

# Plot isi of 4th neuron
#  start : Start time (default self.start_time)
#  end : End time (default self.end_time)
#  isi_thresh : isi threshold (default 0.1)
#  nbins : Number of bins (default 101)
#  lplot : To plot or not (default lplot=1, plot isi)
#  lonoff : Apply on off times (default on, 1)
n1[4].isi_hist(start=False, end=False, isi_thresh=0.1, nbins=101,
               lplot=1, lonoff=1)

# plot firing rate of 4th neuron
# binsz : Bin size (default 3600)
# start : Start time (default self.start_time)
# end : End time (default self.end_time)
# lplot : To plot or not (default lplot=1, plot firing rate)
# lonoff : Apply on off times (default on, 1)
n1[4].plotFR(binsz=3600, start=False, end=False,
             lplot=1, lonoff=1)

# Calculate presence ratio
# nbins : Number of bins (default 101)
# start : Start time (default self.start_time)
# end : End time (default self.end_time)
# lonoff : Apply on off times (default on, 1)
n1[4].presence_ratio(nbins=101, start=False, end=False,
                     lonoff=1)

# Calculate isi contamination at various thresholds, 2 and 4 ms
# cont_thresh_list : threshold lists for calculating isi contamination
# time_limit : count spikes upto, default np.inf. Try also 100 ms, 0.1
# start : Start time (default self.start_time)
# end : End time (default self.end_time)
# lonoff : Apply on off times (default on, 1)
n1[4].isi_contamination(cont_thresh_list=[0.002, 0.004], time_limit=np.inf)
                        start=False, end=False, lonoff=1)

# Check quality and its probability from autoqual (see below).
print(n[2].quality, "", n[2].qual_prob[n[2].quality - 1])

# Change quality of neuron n[0] to 1
# qual : Quality values should be 1, 2, 3 or 4
#        1 : Good
#        2 : Good but some contamination
#        3 : Multiunit contaminated unit
#        4 : Noise unit
# qual_prob array with probabilities for each quality from autoqual.
# When quality is assigned manualy qual_prob is set to 100% for that quality
n[0].set_qual(1)

# get spiketimes from all neurons in n1 as a list
spiketimes_list = n_getspikes(n1)

# Get spikewords from all neurons in n as a list
import numpy as np
import musclebeachtools as mbt
n = np.load('neurons_group0.npy', allow_pickle=True)
sw = mbt.n_spiketimes_to_spikewords(n)

# Set on off times
n[2].set_onofftimes_from_list([0, 3600], [900, 7200])
print(n[2].on_times)
[0, 3600]
print(n[2].off_times)
[900, 7200]

# Change quality with plot of ISI, Firingrate and Waveform
import numpy as np
import musclebeachtools as mbt
n = np.load('neurons_group0.npy', allow_pickle=True)

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
n[2].checkqual(binsz=3600, start=False, end=False, lsavepng=0,
               png_outdir=None, fix_amp_ylim=0)
```
![checkqual](https://biolinux2.wustl.edu/hlabdata/images/mbt_checkqual.png)

```
# Check quality is changed also there is a log from checkqual
print(n[2].quality)

# Find quality using xgboost
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

# Verify quality is correct using checkqual
# lsavepng : Save checkqual results as png's, default 0
# png_outdir : Directory to save png files
#              if lsavepng=1 and png_outdir=None
#              png's will be saved in current working directory
# fix_amp_ylim=0 : Default off
# fix_amp_ylim=500, to change amplitude plot ylim from 0 to 500.
for neuron in neurons:
    neuron.checkqual(savepng=0, png_outdir=None, fix_amp_ylim=500)

# Remove spikes for neuron with large amplitudes
# Default method based on standard deviation,
For example for neurons of quality 1 and 2,  1.5 to 2.5 standard deviation is fine.
But quality 3 neurons make sure to use larger deviations or check manualy with lplot=True
as it contains multiple nuerons/MUA.

n[4].remove_large_amplitude_spikes(1.5, lstd_deviation=True, start=False, end=False, lplot=True)

# Based on threshold value, for example 1000
n[4].remove_large_amplitude_spikes(1000, lstd_deviation=False, start=False, end=False, lplot=True)
If you are sure and do not want to check plots
 to confirm change lplot=False (not recommended)

# Save a modified neuron list
mbt.n_save_modified_neuron_list(n, '/home/kbn/neuron_mod.npy')
```


```
# Load behavior to neurons
# behavior has default values of np.zeros(2,6) to update behaviour of all cells
# behavior is a 2d array with
# first one with time in each state and
# second one with behavioral states (sleep states),
# As Sleepstates can be modified using Sleep_Wake_Scoring
# please remember to load these everytime in case they are modified.
behavior = np.load('Sleepstates.npy', allow_pickle=True)
mbt.Neuron.update_behavior(behavior))

# To get behavior
neurons[0].get_behavior()
# if behavior is not loaded using above step (using Sleepstates.npy)
# it will raise error ValueError: behavior not added
```

```
# Plot all waveforms in a neuron list
# maxqual filter by quality, list
# plot waveforms for neurons with quality in maxqual
# To see plot
mbt.n_plot_neuron_wfs(n, maxqual=[1, 2, 3, 4], pltname="block1")
# To save plot
mbt.n_plot_neuron_wfs(n, maxqual=[1], pltname="block1",
                      saveloc='/home/kbn/')

# To create neuron list from spikeinteface output folder in spikeinterface environmnet
import numpy as np
import glob
import musclebeachtools as mbt
from datetime import datetime
n = mbt.mbt_spkinterface_out('/home/kbn/co/',
                             '/media/HlabShare/models/xgboost_autoqual_prob',
                             sex='m', birthday=datetime(1970, 1, 1, 00, 00),
                             species='m',
                             animal_name='ABC12345',
                             region_loc='CA1',
                             genotype='te4',
                             expt_cond='monocular deprivation')
nnew[0].species
'm'

nnew[0].region
'CA1'

nnew[0].animal_name
'ABC12345'

nnew[0].sex
'm'

nnew[0].birthday
datetime.datetime(1970, 1, 1, 0, 0)

nnew[0].age_rec # age based on first file in sorting block
datetime.timedelta(days=-18625, seconds=2958)

nnew[0].genotype
'te4'

nnew[0].expt_cond
'monocular deprivation'

# Filter neuron list by quality
neuron_list_filt = mbt.n_filt_quality(neuron_list, maxqual=[1, 2])


# Add neurons
import numpy as np
import musclebeachtools as mbt

neurons = np.load('neuron_add.npy', allow_pickle=True)

new_neuron = neurons[51] + neurons[61]
neurons[51].checkqual()
neurons[61].checkqual()
new_neuron.checkqual()

# if you agree merge is right
print("Length ", len(neurons))
# delete neurons used to merge
neurons = np.delete(neurons, [51, 61])
print("Length ", len(neurons))
# add merged new merged neuron
neurons = np.append(neurons, new_neuron)
print("Length ", len(neurons))

# branching ratio from Viola Priesemann
import numpy as np
import musclebeachtools as mbt

n = np.load('H_2020-12-17_13-19-30_2020-12-18_01-14-32_neurons_group0.npy', allow_pickle=True)
# filer by quality
neuron_list = mbt.n_filt_quality(n, maxqual=[1, 2])

# start, end are starting time and end time for br calculation, in hours
# ava_binsz binsize for branching ration, in seconds
# kmax  150-2500, remember in ms multiply ava_binsz
# plotname None no figures
# br1 branching ratio with complex fit
# br2 branching ratio with exp_offset fit
# acc1 is pearson correlation coefficient with respect to data for complex (experimental not tested fully)
# acc2 is pearson correlation coefficient with respect to data for exp_offset (experimental not tested fully)
br1, br2, acc1, acc2 = mbt.n_branching_ratio(neuron_list, ava_binsz=0.004,
                                kmax=500,
                                start=0, end=2,
                                binarize=1,
                                plotname='/home/kbn/hhh.png')
print("Branching ratio ", br1, " pearson corr ", acc1, flush=True)

# check two neurons are from same tetrode
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

# Shuffle spike times
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
# to test whether everything worked
# check quality of some neurons
neurons = np.load(fl, allow_pickle=True)
for indx, neuron in enumerate(neurons):
    if neuron.quality < 3:
        neuron.checkqual()
print("shuffled")
neurons_new = np.load(fl_new, allow_pickle=True)
for indx, neuron_new in enumerate(neurons_new):
    if neuron_new.quality < 3:
        neuron_new.checkqual()

# Correlograms
import numpy as np
import musclebeachtools as mbt
fl = '/media/ckbn/H_2022-05-12_08-59-26_2022-05-12_17-54-28_neurons_group0.npy'
n = np.load(fl, allow_pickle=True)
# Autocorr
# default savefig_loc=None, to show plot
# savefig_loc='/home/kbn/fig_crosscorr.png' to save fig
n[1].crosscorr(friend=None, savefig_loc=None)
# Crosscorr
n[10].crosscorr(friend=n[11], savefig_loc=None)


# zero cross corr
import numpy as np
import musclebeachtools as mbt
fl = 'H_2020-08-07_14-00-15_2020-08-08_01-55-16_neurons_group0.jake_scored_q12.npy'
neurons = np.load(fl, allow_pickle=True)
zcl_i, zcl_v = mbt.n_zero_crosscorr(neurons)
print(zcl_i, "\n", zcl_v)


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
## FAQ
```
1. spike_time vs spike_time_sec
Property spike_time is in sample times.
To to get spike time in seconds
please use spike_time_sec, n1[4].spike_time_sec
or
For example for 4th neuron n1[4].spike_time/n1[4].fs
Also spike_time_sec_onoff filters spike_time_sec based on on/off times

```
---
## Issues

```Please slack Kiran ```
---
