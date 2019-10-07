# musclebeachtools
---
This package used to analyze neuronal data, from basic plotting
 (ISI histograms, firing rate, cross correlation),
 to hard core computational work (ising models, branching ratios, etc).  

---
## Installation

### Download musclebeachtools
git clone https://github.com/hengenlab/musclebeachtools_hlab.git   
Enter your username and password

### Using pip
cd locationofmusclebeachtools_hlab/musclebeachtools/  
pip install .


---
### Installation by adding to path

##### Windows
My Computer > Properties > Advanced System Settings > Environment Variables >  
In system variables, create a new variable  
    Variable name  : PYTHONPATH  
    Variable value : location where musclebeachtools_hlab is located  
    Click OK


##### Linux
If you are using bash shell  
In terminal open .barshrc or .bash_profile  
add this line  
export PYTHONPATH=/location_of_musclebeachtools_hlab:$PYTHONPATH


##### Mac
If you are using bash shell  
In terminal cd ~/  
then open  .profile using your favourite text editor  
add this line  
export PYTHONPATH=/location_of_musclebeachtools_hlab:$PYTHONPATH


---
##### Test import
Open powershell/terminal     
    ipython    
    import musclebeachtools.musclebeachtools as mbt   
---

## Usage
```
import musclebeachtools as mbt

datadir = "/hlabhome/kiranbn/neuronclass/t_lit_EAB50final/"
n1 = mbt.ksout(datadir, filenum=0, prbnum=4, filt=[])

# Plot mean waveform of 4th neuron
n1[4].plot_wf()

# Plot isi of 4th neuron
n1[4].isi_hist()

# plot firing rate of 4th neuron
n1[4].plotFR()

# get spiketimes from all neurons in n1 as a list
spiketimes_list = n_getspikes(n1)

```
## Issues

```Please slack Kiran ```
---
