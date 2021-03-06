import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import time
import seaborn as sns
import matplotlib
import glob
#matplotlib.use('TkAgg')
from IPython.core.debugger import Tracer

class neuron(object):
    '''Create instances (objects) of the class neuron ''' 
    def __init__(self, datafile='/Volumes/HlabShare/Clustering_Data/EAB00040/ts/', cell_idx=1, datatype='npy', start_day = 0, end_day = 1, silicon=False):
        if datatype == 'npy':
            print('You are using WashU data')
            fs = 25000
            print('working on neuron '+str(cell_idx)+'\n')

            

            # going to folder that contains processed data if it exists
            try:
                os.chdir(datafile)
            except FileNotFoundError:
                print("*** Data File does not exist *** check the path")
                return


            #if the recording is solicon pull the files based on input channel 
            #pull all of the relevant files with silicon recording
            if(silicon):
                ch = input("What channel would you like to look at?")
                f="*chg_"+str(ch)+"*"
                channelFiles=np.sort(glob.glob(f))
                #sorts spikes and clusters
                spikefiles = [channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*spike_times*.npy"))]
                #print("spike_files: ", spikefiles)
                clustfiles = [channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*spike_clusters*.npy"))]
                #sorts any peak channel files found in folder
                peakfiles = [channelFiles[i] for i in range(len(channelFiles)) if (channelFiles[i] in np.sort(glob.glob("*peakchannel*.npy")) or channelFiles[i] in np.sort(glob.glob("*max_channel*.npy")))]
                #sorts wavefiles in two forms, named "waveform" or "templates"
                wavefiles=[channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*waveform*.npy"))]
                templates_all=[channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*templates*.npy"))]
                #since there are multiple files that have "templates" at the end this pulls out only the ones we want
                templates_wf=[fn for fn in templates_all if fn not in glob.glob("*spike*.npy") and fn not in glob.glob("*similar*.npy") and fn not in glob.glob("*number*.npy")]
                #checks for amplitude files
                amplitude_files = [channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*amplitudes*.npy"))]
                #this checks for an automated quality array from the clustering algorithm
                aq = [channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*qual*.npy"))]
                #looks for scrubbed quality and loads if possible
                sq = [channelFiles[i] for i in range(len(channelFiles)) if channelFiles[i] in np.sort(glob.glob("*scrubbed*.npy"))]
                #print("peakfiles: ", peakfiles, "\nclustfiles: ", clustfiles, "\nspikefiles: ", spikefiles, "\nwavefiles: ", wavefiles)
            #pull all relevant files in non-silicon recordings
            else:
                #sorts spikes and clusters
                spikefiles = np.sort(glob.glob("*spike_times*.npy"))
                clustfiles = np.sort(glob.glob("*spike_clusters*.npy"))
                #sorts any peak channel files found in folder
                peakfiles = np.sort(glob.glob("*max_channel*.npy"))
                if(len(peakfiles)==0):
                    peakfiles=np.sort(glob.glob("*peakchannel*.npy"))
                #peakfiles.extend(np.sort(glob.glob("*max_channel*.npy")))
                #sorts wavefiles in two forms, named "waveform" or "templates"
                wavefiles = np.sort(glob.glob("*waveform*.npy"))
                templates_all=np.sort(glob.glob("*template*.npy"))
                #since there are multiple files that have "templates" at the end this pulls out only the ones we want
                templates_wf=[fn for fn in templates_all if fn not in glob.glob("*spike*.npy") and fn not in glob.glob("*similar*.npy") and fn not in glob.glob("*number_of_*.npy") and fn not in glob.glob("*templates_in_clust.npy")]
                #checks for amplitude files
                amplitude_files = np.sort(glob.glob("*amplitudes*.npy"))
                #this checks for an automated quality array from the clustering algorithm
                aq = np.sort(glob.glob("*qual*.npy"))
                #looks for scrubbed quality and loads if possible
                sq = np.sort(glob.glob("*scrubbed*.npy"))
            
            if len(peakfiles)==0:
                #if there are no peak channel files, set the flag to False and print
                has_peak_files = False
                print("Peak Channels: No")
            else:
                #if there are files, set the flag to True and print
                 print("Peak Channels: Yes")
                 has_peak_files = True

            

            if len(wavefiles)==0 and len(templates_wf)==0:
                #if there are none, set flag to False and print
                has_twf=False
                print("Waveform array: No")
            elif len(wavefiles)>0:
                #if the format was "wavefiles" then set flag to True and load those files
                has_twf=True
                w=np.load(wavefiles[0])
                print("Waveform array: Yes")
            else:
                #if the format is "templates" set flag to True and load those files
                has_twf=True
                w=np.load(templates_wf[0])
                print("Waveform array: Yes")
            

            if len(amplitude_files) > 0 :
                #if there are amplitude files then set the instance variable and print
                print("loading amplitudes")
                self.amplitudes = np.load(glob.glob("*amplitudes*.npy")[0])



            #we don't have keys yet so we can't track cells across multiple days
            if end_day-start_day > 1:

                #KEYS
                keys = np.load(glob.glob("new_key*")[0])
                length_start = [(spikefiles[i].find('length_')+7) for i in range(start_day, end_day)]
                length_end = [(spikefiles[i].find('_p')) for i in range(start_day, end_day)]
                length = [int(spikefiles[i][length_start[i]:length_end[i]]) for i in range(start_day, end_day)]
                length = np.append([0], length)
                length = np.cumsum(length)

               
            else:
                #in order to account for having to add the length once we have keys
                length = np.zeros(end_day)


            #try to load clusters
            try:
                print('loading clusters')
                curr_clust = [np.load(clustfiles[i]) for i in range(start_day, end_day)]
            except IndexError:
                print('spike cluster files do not exist for that day range')
                return

            #try to load spikes
            try:
                print('loading spikes')

                '''length[i] not necessary for single day ranges but will be for multiple days'''
                curr_spikes = [np.load(spikefiles[i])+length[i] for i in range(start_day, end_day)]
                
            except IndexError:
                print('spike time files do not exist for that day range')
                return

            #load peak channels if availible 
            if has_peak_files:
                try:
                    print('loading peak channels')
                    peak_ch = np.concatenate([np.load(peakfiles[i])for i in range(start_day, end_day)])
                except IndexError:
                    print('peak channel files do not exist for that day range')
                    return

            
            #this needs to be initialized here in order to be used in the next block
            self.peak_chans = np.zeros(end_day-start_day)

            #again this part isn't ready yet since we don't have keys
            if end_day-start_day > 1:
                spiketimes = [] 
                clusters = np.load(glob.glob('*unique_clusters*.npy')[0])
                
                for f in range(start_day, end_day):
                    
                    #KEYS
                    key_val = keys[f, int(cell_idx-1)]
                    clust_idx = clusters[int(key_val)]
                    spk_idx = np.where(curr_clust[f] == clust_idx)[0]  
                    spks = np.concatenate(curr_spikes[f][spk_idx])/fs 
                    if has_peak_files:
                        self.peak_chans[f] = peak_ch[f][int(key_val)]
                    spiketimes.append(spks)
                spiketimes = np.concatenate(spiketimes)

                
            else:
                #unique_clust_files=np.sort(glob.glob('*unique_clusters*.npy'))
                #clusters=[np.load(unique_clust_files[i]) for i in range(start_day, end_day)]

                #pulls out the unique cluster numbers
                self.unique_clusters = [np.unique(curr_clust)]
                try:
                    clust_idx = self.unique_clusters[0][int(cell_idx-1)]
                except IndexError:

                    #this error is very general, there are multiple things that could be happening
                    #most likely there is an issue with the way clusters is being indexed into
                     print("cluster file index error")
                     return
                #pulls out all indices of that cluster spiking based on cluster index and spike clusters
                spk_idx = np.where(curr_clust[0] == clust_idx)[0]
                #loads all times at those indicies
                spiketimes = curr_spikes[0][spk_idx]/fs
                #if there are peak channels this loads them into the instance variables
                if has_peak_files:
                    self.peak_chans = peak_ch[int(cell_idx-1)]



            #QUALITY STUFF
            #everything is set up for only one day right now, should be changed eventually
            #looks for automated quality array and loads if possible

            
            if len(aq)==0:
                #if there is none set the Flag and print
                print ('No automated quality file')
                self.has_aqual=False
            else:
                #if there is an array then set the Flag and load it into the instance variable
                self.has_aqual=True
                self.auto_qual_array = np.load(aq[0])


            
            if len(sq)>0:
                #set the flag and load it to the instance varaible
                self.has_squal = True
                #uneccisary but it gets used later so currently keeping it
                scrubbed_qual=np.load(sq[0])
                self.scqu_file = sq[0]
                self.scrubbed_qual_array = np.load(sq[0])

                #if that particular cell hasn't been scrubbed yet, print automated quality, otherwise print the scrubbed quality
                #set the quality based on the scrubbbed array if it's scrubbed, otherwise set it to automated
                if np.isnan(self.scrubbed_qual_array[cell_idx-1]):
                    if self.has_aqual:
                        self.quality = self.auto_qual_array[cell_idx-1][0]
                        print("\nScrubbed: NO")
                        print("\nQuality rating (automated): ", self.quality)
                else:
                    self.quality = self.scrubbed_qual_array[cell_idx-1][0]
                    print('\nScrubbed: YES')
                    print("\nQuality set at: ", self.quality)
            
            #if there is no scrubbed quality, print automated quality, otherwise set the quality to 0       
            else:
                self.has_squal = False
                #if there is an automated quality file but no scrubbed file then print the automated quality
                #if there's neither than set the quality to 0
                if self.has_aqual:
                    self.quality = self.auto_qual_array[cell_idx-1][0]
                    print("\nQuality rating (automated): ", self.quality)
                else:
                    print("\nNo automated quality rating, check the quality to set a scrubbed quality rating")
                    self.quality = 0
                
                

           #sets the time instance variables to all the spike times
            self.time = np.concatenate(spiketimes)
            #sets on an off time based on first and last numbers in the spike time array
            self.onTime = np.array([self.time[0]])
            self.offTime = np.array([self.time[-1]])

            
        
            #if there is templatewf files, set it up
            if has_twf:
                self.wf_temp = w[cell_idx-1]
                bottom      = np.argmin(self.wf_temp)
                top         = np.argmax(self.wf_temp[bottom:]) + bottom
                np_samples  = top - bottom

                seconds_per_sample  = 1/fs
                ms_per_sample       = seconds_per_sample*1e3
                self.neg_pos_time   = np_samples * ms_per_sample
                self.cell_idx = cell_idx
                if self.neg_pos_time >=0.4:
                    self.cell_type  = 'RSU'
                if self.neg_pos_time <0.4:
                    self.cell_type = 'FS' 
              

            #this is the flag to tell if there is WF array information later    
            self.wf = has_twf

        #Brandeis data
        else:
            print('You are using Brandeis data')
            print('Building neuron number {}'.format(cell_idx))
            f               = h5py.File(datafile,'r')
            try:
                self.animal     = np.array(f['neurons/neuron_'+str(cell_idx)+'/animal'][:], dtype=np.int8).tostring().decode("ascii")
            except:
                print('Stuck at line 19 in neuron_class.py')
                Tracer()()

            self.HDF5_tag   = (datafile,cell_idx)
            self.deprived   = f['neurons/neuron_'+str(cell_idx)+'/deprived'][0]
            self.channel    = np.int(f['neurons/neuron_'+str(cell_idx)+'/channel'][0])
            self.cont_stat  = f['neurons/neuron_'+str(cell_idx)+'/cont_stat'].value#bool(f['neurons/neuron_'+str(cell_idx)+'/cont_stat'][0])
            self.halfwidth  = f['neurons/neuron_'+str(cell_idx)+'/halfwidth'][0]
            self.idx        = f['neurons/neuron_'+str(cell_idx)+'/idx'][:]
            self.meantrace  = f['neurons/neuron_'+str(cell_idx)+'/meantrace'][:]
            self.scaledWF   = f['neurons/neuron_'+str(cell_idx)+'/scaledWF'][:]

            if np.size(f['neurons/neuron_'+str(cell_idx)+'/offTime'].shape) > 0:

                self.offTime    = f['neurons/neuron_'+str(cell_idx)+'/offTime'][:]
                self.onTime     = f['neurons/neuron_'+str(cell_idx)+'/onTime'][:]

            elif np.size(f['neurons/neuron_'+str(cell_idx)+'/offTime'].shape) == 0:
                self.offTime    = []
                self.onTime     = []

            try:
                self.qflag      = f['neurons/neuron_'+str(cell_idx)+'/qflag'][0]
                self.score      = f['neurons/neuron_'+str(cell_idx)+'/score'][:]
            except:
                pass

            self.tailSlope  = f['neurons/neuron_'+str(cell_idx)+'/tailSlope'][0]
            self.trem       = f['neurons/neuron_'+str(cell_idx)+'/trem'][0]
            
            self.time       = f['neurons/neuron_'+str(cell_idx)+'/time'][:]
            self.quality    = np.int(f['neurons/neuron_'+str(cell_idx)+'/quality'][0])
            self.cell_idx   = cell_idx

            # calculate the half-width correctly:
            sampling_rate   = 24414.06
            og_samples      = np.shape(self.meantrace)[0]

            if og_samples == 33 or og_samples == 21:
                interp_factor = 1
            elif og_samples == 59 or og_samples == 97 or og_samples == 100:
                interp_factor = 3
            elif og_samples == 91 or og_samples == 161:
                interp_factor = 5
            
            bottom      = np.argmin(self.meantrace)
            top         = np.argmax(self.meantrace[bottom:]) + bottom
            np_samples  = top - bottom

            seconds_per_sample  = 1/(interp_factor*sampling_rate)
            ms_per_sample       = seconds_per_sample*1e3
            self.neg_pos_time   = np_samples * ms_per_sample
        self.datatype = datatype


    def plotFR(self, axes = None, savehere=None,counter=None, binsz = 3600):
        # Plot the firing rate of the neuron in 1h bins, and add grey bars to indicate dark times. This is all based on the standard approach for our recordings and will have to be updated to accomodate any majorly different datasets (e.g. very short recordings or other L/D arrangements)
        edges   = np.arange(0,max(self.time)+binsz,binsz)
        bins    = np.histogram(self.time,edges)
        hzcount = bins[0]
        hzcount = hzcount/binsz
        hzcount[hzcount==0] = 'NaN'
        xbins   = bins[1]
        xbins   = xbins/binsz

        plt.ion()
        if axes:
            currentAxis = axes
        else:
            with sns.axes_style("white"):
                fig1        = plt.figure()
                currentAxis = plt.gca()
                plt.plot(xbins[:-1],hzcount)     

        plt.gca().set_xlim([0,edges[-1]/binsz])
        ylim    = plt.gca().get_ylim()[1]
        
        # make an array of the lights-off times
        if self.datatype == 'hdf5':
            lt_off = np.arange(12*3600,max(self.time)+12*3600,24*3600)
            # cycle through the lights-off times and plot a transparent grey bar to indicate the dark hours
            for p in lt_off/binsz:
                currentAxis.add_patch(patches.Rectangle((p, 0), 12, ylim, facecolor="grey",alpha=0.1, edgecolor="none"))

        # deal with on/off times
        if np.size(self.onTime) == 1:
            # if there is only one on/off time, just plot as dashed lines
            plt.plot((self.onTime[0]/binsz,self.onTime[0]/binsz),(0,ylim),'g--')
            plt.plot((self.offTime[0]/binsz,self.offTime[0]/binsz),(0,ylim),'r--')

        elif np.size(self.onTime) > 1:
            # in this case, plot the start and end of the cell as a dashed line, and add a red shaded box around any remaining periods of "off" times
            count = 0
            for ee in self.offTime[:-1]/binsz:
                count += 1
                wdth = self.onTime[count]/binsz - ee
            
            plt.plot((self.onTime[0]/binsz,self.onTime[0]/binsz),(0,ylim),'g--')
            plt.plot((self.offTime[-1]/binsz,self.offTime[-1]/binsz),(0,ylim),'r--')

        elif np.size(self.onTime) == 0:
            print('I did not find any on/off times for this cell.')


        # if self.deprived:
        #     plt.text(12,0.7*ylim,'Deprived')
        # else:
        #     plt.text(12,0.7*ylim,'Control')


        plt.ion()
        sns.set()
        sns.despine()
        plt.gca().set_xlabel('Time (hours)')
        plt.gca().set_ylabel('Firing rate (Hz)')
        plt.show()

        if savehere:
            fig1.savefig(savehere+'NRN_'+str(counter)+'_FRplot.pdf')
            


    def isi_hist(self, start = 0, end = False, isi_thresh = 0.1, nbins = 101):
        '''Return a histogram of the interspike interval (ISI) distribution. This is a method built into the class "neuron", so input is self (). This is typically used to evaluate whether a spike train exhibits a refractory period and is thus consistent with a single unit or a multi-unit recording. '''
        # For a view of how much this cell is like a "real" neuron, calculate the ISI distribution between 0 and 100 msec. This function will plot the bar histogram of that distribution and calculate the percentage of ISIs that fall under 2 msec. 
        if end == False:
            end = max(self.time)
        idx = np.where(np.logical_and(self.time>=start, self.time<=end))[0]
        ISI = np.diff(self.time[idx])
        edges = np.linspace(0,isi_thresh,nbins)
        hist_isi        = np.histogram(ISI,edges)
        contamination   = 100*(sum(hist_isi[0][0:int((0.1/isi_thresh)*(nbins-1)/50)])/sum(hist_isi[0]))
        contamination   = round(contamination,2)
        cont_text       = 'Contamination is {} percent.' .format(contamination)

        plt.ion()
        with sns.axes_style("white"):
            fig1            = plt.figure()  
            ax              = fig1.add_subplot(111)
            ax.bar(edges[1:]*1000-0.5, hist_isi[0],color='#6a79f7')
            ax.set_ylim(bottom = 0)
            ax.set_xlim(left = 0)
            ax.set_xlabel('ISI (ms)')
            ax.set_ylabel('Number of intervals')
            ax.text(30,0.7*ax.get_ylim()[1],cont_text)
        sns.despine()
        return ISI

    def crosscorr(self,friend=None,dt=1e-3,tspan=1.0,nsegs=None):
        # if "friend" argument is not give, compute autocorrelation
        if friend is None:
            print('Computing autocorrelation for cell {:d}.'.format(self.cell_idx))
            print('  Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'.format(int(tspan*1000),dt*1000))
            # select spike timestamps within on/off times for this cell
            stimes1     = self.time[ (self.time>self.onTime[0]) & (self.time<self.offTime[-1]) ]
            # remove spikes at the edges (where edges are tspan/2)
            t_start     = time.time()
            subset      = [ (stimes1 > stimes1[0]+tspan/2) & (stimes1 < stimes1[-1] - tspan/2) ]
            # line above returns an array of booleans. Convert to indices
            subset      = np.where(subset)[1]
            # Take a subset of indices. We want "nsegs" elements, evenly spaced. "segindx" therefore contains nsegs spike indices, evenly spaced between the first and last one in subset
            if nsegs is None:
                nsegs   = np.int(np.ceil(np.max(self.time/120)))
            print('\tUsing {:d} segments.'.format(nsegs))
            segindx     = np.ceil( np.linspace( subset[0], subset[-1], nsegs) )
            # The spikes pointed at by the indices in "segindx" are our reference spikes for autocorrelation calculation

            # initialize timebins
            timebins    = np.arange(0,tspan+dt,dt)
            # number of bins
            nbins       = timebins.shape[0] - 1

            # initialize autocorrelation array
            ACorrs      = np.zeros((nsegs,2*nbins-1),float)

            # ref is the index of the reference spike
            for i,ref in enumerate(segindx):
                ref = int(ref)
                # "t" is the timestamp of reference spike
                t = stimes1[ref]
                # find indices of spikes between t and t+tspan
                spikeindx = np.where((stimes1>t) & (stimes1 <= t+tspan))[0]
                # get timestamps of those and subtract "t" to get "tau" for each spike
                # "tau" is the lag between each spike
                # divide by "dt" step to get indices of bins in which those spikes fall
                spikebins = np.ceil((stimes1[spikeindx] - t)/dt)
                if spikebins.any():
                    # if any spikes were found using this method, create a binary array of spike presence in each bin
                    bintrain    = np.zeros(nbins,int)
                    bintrain[spikebins.astype(int)-1] = 1
                    # the auto-correlation is the convolution of that binary sequence with itself
                    # mode="full" ensures np.correlate uses convolution to compute correlation
                    ACorrs[i,:] = np.correlate(bintrain,bintrain,mode="full")

            # sum across all segments to get auto-correlation across dataset
            Y = np.sum(ACorrs,0)
            # remove artifactual central peak
            Y[nbins-1] = 0

            # measure time elapsed for benchmarking
            elapsed = time.time() - t_start
            print('Elapsed time: {:.2f} seconds'.format(elapsed))

            # plotting ----------------------
            # -------------------------------
            plt.ion()
            fig = plt.figure(facecolor='white')
            fig.suptitle('Auto-correlation, cell {:d}'.format(self.cell_idx))
            ax2 = fig.add_subplot(211, frame_on=False)

            ax2.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
            xlim2       = 100 # in milliseconds
            tickstep2   = 20 # in milliseconds
            ax2.set_xlim(0,xlim2)
            ax2_ticks   = [i for i in range(0,xlim2+1,tickstep2)]
            ax2_labels  = [str(i) for i in ax2_ticks]
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels(ax2_labels)
            ax2.set_ylabel('Counts')

            ax3         = fig.add_subplot(212, frame_on=False)
            ax3.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
            xlim3       = int(tspan*1000) # milliseconds - set to tspan
            tickstep3   = int(xlim3/5) # milliseconds
            ax3_ticks   = [i for i in range(-xlim3,xlim3+1,tickstep3)]
            ax3_labels  = [str(i) for i in ax3_ticks]
            ax3.set_xticks(ax3_ticks)
            ax3.set_xticklabels(ax3_labels)
            ax3.set_ylabel('Counts')
            ax3.set_xlabel('Lag (ms)')

            fig.show()
            plt.show()

            # -------------------------------

        # if friend is an instance of class "neuron", compute cross-correlation
        elif isinstance(friend, neuron):
            print('Computing cross correlation between cells {:d} and {:d}.'.format(self.cell_idx,friend.cell_idx))
            print('  Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'.format(int(tspan*1000),dt*1000))
            # select spike timestamps within on/off times for self cell
            stimes1 = self.time[ (self.time>self.onTime[0]) & (self.time<self.offTime[-1]) ]
            # select spike timestamps within on/off times for self cell
            stimes2 = friend.time[ (friend.time>friend.onTime[0]) & (friend.time<friend.offTime[-1]) ]
            # start timer for benchmarking
            t_start = time.time()
            # remove spikes at the edges (where edges are tspan/2)
            subset1      = [ (stimes1 > stimes1[0]+tspan/2) & (stimes1 < stimes1[-1] - tspan/2) ]
            subset2      = [ (stimes2 > stimes2[0]+tspan/2) & (stimes2 < stimes2[-1] - tspan/2) ]
            # line above returns an array of booleans. Convert to indices
            subset1      = np.where(subset1)[1]
            subset2      = np.where(subset2)[1]
            # check to see if nsegs is user provided or default
            if nsegs is None:
                nsegs1  = np.int(np.ceil(np.max(self.time/120)))
                nsegs2  = np.int(np.ceil(np.max(friend.time/120)))
                nsegs   = max(nsegs1,nsegs2)
            print('\tUsing {:d} segments.'.format(nsegs))
            # Take a subset of indices. We want "nsegs" elements, evenly spaced. "segindx" therefore contains nsegs spike indices, evenly spaced between the first and last one in subset
            segindx1     = np.ceil(np.linspace(subset1[0], subset1[-1], nsegs))
            segindx2     = np.ceil(np.linspace(subset2[0], subset2[-1], nsegs))

            # The spikes pointed at by the indices in "segindx" are our reference spikes for autocorrelation calculation

            # initialize timebins
            timebins = np.arange(0, tspan+dt,   dt)
            # number of bins
            nbins    = timebins.shape[0] - 1

            # initialize autocorrelation array
            XCorrs = np.zeros((nsegs,2*nbins-1),float)

            # ref is the index of the reference spike
            for i, ref in enumerate(segindx1):
                ref = int(ref)
                # "t" is the timestamp of reference spike
                t = stimes1[ref]
                # find indices of spikes between t and t+tspan, for cell SELF
                spikeindx1 = np.where((stimes1>t) & (stimes1 <= t+tspan))[0]
                # same thing but for cell FRIEND
                spikeindx2 = np.where((stimes2>t) & (stimes2 <= t+tspan))[0]
                # get timestamps of those and subtract "t" to get "tau" for each spike
                # "tau" is the lag between each spike
                spikebins1 = np.ceil((stimes1[spikeindx1] - t)/dt)
                spikebins2 = np.ceil((stimes2[spikeindx2] - t)/dt)
                if spikebins1.any() & spikebins2.any():
                    # binary sequence for cell SELF
                    bintrain1  = np.zeros(nbins, int)
                    bintrain1[spikebins1.astype(int)-1] = 1
                    # binary sequence for cell FRIEND
                    bintrain2   = np.zeros(nbins, int)
                    bintrain2[spikebins2.astype(int)-1] = 1
                    XCorrs[i, :] = np.correlate(bintrain1, bintrain2, mode="full")

            Y = np.sum(XCorrs,0)

            elapsed     = time.time() - t_start
            print('Elapsed time: {:.2f} seconds'.format(elapsed))
            plt.ion()
            figx        = plt.figure(facecolor='white')
            figx.suptitle('Cross-correlation, cells {:d} and {:d}'.format(self.cell_idx,friend.cell_idx))

            ax2         = figx.add_subplot(211, frame_on=False)

            ax2.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
            xlim2       = int(200) # in milliseconds
            tickstep2   = int(xlim2/5) # in milliseconds
            ax2.set_xlim(-xlim2,xlim2)
            ax2_ticks   = [i for i in range(-xlim2,xlim2+1,tickstep2)]
            ax2_labels  = [str(i) for i in ax2_ticks]
            #ax2_labels = str(ax2_labels)
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels(ax2_labels)
            ax2.set_ylabel('Counts')

            ax3         = figx.add_subplot(212, frame_on=False)
            ax3.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
            xlim3       = int(tspan*1000) # milliseconds - set to tspan
            tickstep3   = int(xlim3/5) # milliseconds
            ax3_ticks   = [i for i in range(-xlim3,xlim3+1,tickstep3)]
            ax3_labels  = [str(i) for i in ax3_ticks]
            ax3.set_xticks(ax3_ticks)
            ax3.set_xticklabels(ax3_labels)
            ax3.set_ylabel('Counts')
            ax3.set_xlabel('Lag (ms)')

            figx.show()
            plt.show()

        elif not isinstance(friend, neuron):
            # error out
            raise TypeError('ERROR: Second argument to crosscorr should be an instance of ''neuron'' class.')


    def checkqual(self, save_update=False):

        #binsz set as elapsed time/100
        elapsed = self.time[-1] - self.time[0]
        binsz = elapsed/100

        if np.size(np.shape(self.offTime)) == 2:
            offts = np.squeeze(self.offTime)
        else:
            offts = self.offTime

        if np.size(np.shape(self.onTime)) == 2:
            onts = np.squeeze(self.onTime)
        else:
            onts = self.onTime

        try:
            ontimes         = self.time[ (self.time>onts[0]) & (self.time<offts[-1]) ]
        except:
            Tracer()()

        #determine contamination and set up ISI histogram
        isis            = np.diff(ontimes)
        edges           = np.linspace(0,1.0,1001)
        oldedges        = np.linspace(0,0.1,101)
        hist_isi        = np.histogram(isis,edges)
        oldhist         = np.histogram(isis,oldedges)
        contamination   = 100*(sum(hist_isi[0][0:2])/sum(hist_isi[0]))
        contamination   = round(contamination,2)
        oldcont         = 100*(sum(oldhist[0][0:2])/sum(oldhist[0]))
        oldcont         = round(oldcont,2)
        cont_text       = 'Contamination: {} percent.' .format(contamination)
        oldconttext     = 'Prior measure: {} percent.'.format(oldcont)



        # Check contamination by bin size and then return a descriptive statistic of the contamination in bins of bin size:
        hrcont  = np.repeat(999.0, 260)
        hct = 0
        tcount  = 0
        for ee in onts:

            iedge   = np.arange( ee, offts[tcount], binsz)

            for i in iedge[:-1]:
                tmpisi          = np.diff(self.time[ (self.time>i) & (self.time<(i+binsz)) ]) # ISIs/binsz
                hist_isi        = np.histogram(tmpisi, edges)
                hrcont[hct]     = 100*(sum(hist_isi[0][0:2])/sum(hist_isi[0]))
                hct += 1

            tcount += 1

        hrcont          = np.delete( hrcont,[np.where(hrcont==999.0)] )
        
        if self.time[-1] < binsz:
            print ("Not enough data for firing rate, check bin size")
            newcont_text    = 'Mean Contamination by hour: --not enough data--' 
            newcont_text2   = 'Median Contamination by hour: --not enough data--'
        else:
            newcont_text    = 'Mean Contamination by bin size: {0:.{1}} percent.' .format(np.nanmean(hrcont),4)
            newcont_text2   = 'Median Contamination by bin size: {0:.{1}} percent.' .format(np.nanmedian(hrcont),4)

        plt.ion()
        plt.rcParams['font.family'] = 'serif'
        fig8            = plt.figure(8,figsize=(12, 6), dpi=100,facecolor='white')
        sns.set(style="ticks")
        sns.despine()

        
        numGraphs=2
        #self.wf=False
        if self.wf: 
            numGraphs = 3


        # PLOT THE ISI HISTOGRAM:
        ax1             = plt.subplot(1,numGraphs,1,frame_on=False)
        # create ISI historgram over entire period of cell on, and plot that here
        tmpisi          = np.diff(self.time[ (self.time>self.onTime) & (self.time<self.offTime) ]) # ISIs/binsz
        hist_isi        = np.histogram(tmpisi, edges)
        ax1.bar(edges[1:]*1000-0.5, hist_isi[0],color=(0.7, 0.7, 0.7))
        ax1.set_xlim(0,100)
        ax1.set_ylim(bottom = 0)
        ax1.set_xlabel('ISIs (msec)')
        ax1.set_ylabel('$Number of intervals$')
        ax1.text(5,0.90*ax1.get_ylim()[1], oldconttext, fontsize="small")
        ax1.text(5,0.85*ax1.get_ylim()[1], cont_text, fontsize="small")
        ax1.text(5,0.8*ax1.get_ylim()[1], newcont_text, fontsize="small")
        ax1.text(5,0.75*ax1.get_ylim()[1], newcont_text2, fontsize="small")


        # PLOT THE MEAN TRACE (template waveform):
        if self.wf:
            ax1.text(0,.96, "Cell type: " + self.cell_type, transform=ax1.transAxes, color='black', fontsize='medium')
            sns.set(style="ticks")
            sns.despine()
            ax2             = plt.subplot(1,numGraphs,2,frame_on=False)
            if self.datatype == 'hdf5':
                ax2.plot(self.meantrace)
            else:
                 ax2.plot(self.wf_temp)
            ax2.set_ylabel('$Millivolts$')
        else:
            ax1.text(0,.96,"No waveform template found", transform=ax1.transAxes, color='red')

        quality_rating = "Quality set at: %s" % self.quality if self.quality!=None else "Quality NOT RATED yet"
        ax1.text(0,1,quality_rating, transform=ax1.transAxes, color='green')
        # if self.wf:
        #     ax1.text(0,.96, "Cell type: " + self.cell_type, transform=ax1.transAxes, color='black', fontsize='medium')

        # PLOT THE FIRING RATE TRACE
        edges   = np.arange(0,max(self.time)+2*binsz,binsz)
        bins    = np.histogram(self.time,edges)
        hzcount = bins[0]
        hzcount = hzcount/binsz
        hzcount[hzcount==0] = 'NaN'
        xbins   = bins[1]
        xbins   = xbins/binsz
        ax3     = plt.subplot(1,numGraphs,numGraphs,frame_on=False)
        with sns.axes_style("white"):
            ax3.plot(xbins[:-1],hzcount)

        ax3.set_xlim([0,edges[-1]/3600])
        ylim    = ax3.get_ylim()[1]

        if self.datatype == 'hdf5':
            # make an array of the lights-off times
            lt_off  = np.arange(12*3600,max(self.time)+12*3600,24*3600)

            # cycle through the lights-off times and plot a transparent grey bar to indicate the dark hours
            for p in lt_off/3600:
                ax3.add_patch(patches.Rectangle((p, 0), 12, ylim, facecolor="grey",alpha=0.1, edgecolor="none"))


        # deal with on/off times
        if np.size(onts) == 1:
            # if there is only one on/off time, just plot as dashed lines
            ax3.plot((onts[0]/3600,onts[0]/3600),(0,ylim),'g--')
            ax3.plot((offts[0]/3600,offts[0]/3600),(0,ylim),'r--')
        else:
            # in this case, plot the start and end of the cell as a dashed line, and add a red shaded box around any remaining periods of "off" times
            count = 0
            for ee in offts[:-1]/3600:
                count += 1
                wdth = onts[count]/3600 - ee
                ax3.add_patch(patches.Rectangle((ee, 0), wdth, ylim, facecolor="red",alpha=0.5, edgecolor="none"))

            ax3.plot((onts[0]/3600,onts[0]/3600),(0,ylim),'g--')
            ax3.plot((offts[-1]/3600,offts[-1]/3600),(0,ylim),'r--')


        # if self.deprived == 1:
        #     ax3.text(12,0.7*ylim,'Deprived')
        # else:
        #     ax3.text(12,0.7*ylim,'Control')

        sns.set()
        sns.despine()
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Firing rate (Hz)')

        plt.subplots_adjust(wspace=.3)


        fig8.show()

        g = None
        g = input('What is the quality? ')

        #QUALITY UPDATES
        flag = None
        while flag is None:
            time.sleep(0.1)
            if g in (['1', '2', '3']):
                #currently assuming there might not be a quality file
                if save_update:
                    #always update quality if the save_update flag is true
                    self.quality=g

                    if self.has_squal:
                        #if there is already a scrubbed quality array - update the index of the cell
                        self.scrubbed_qual_array[self.cell_idx-1] = self.quality
                        np.save(self.scqu_file, self.scrubbed_qual_array)
                    else:
                        #make array if there isn't one already

                        self.scrubbed_qual_array = np.full(np.shape(self.unique_clusters), np.NaN)
                        self.scrubbed_qual_array[self.cell_idx-1] = self.quality
                        np.save("scrubbed_quality.npy", self.scrubbed_qual_array)

                flag = 1
            else:
                g = input('What is the quality? ')


        plt.close(fig8)

        # Convert g to number and compare this to the current quality. if it's different, overwrite the HDF5 file and update the quality field there. 
        if self.datatype == 'hdf5':
            if self.quality != float(g):
                f1          = h5py.File(self.HDF5_tag[0], 'r+')
                data        = f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/quality']
                data[...]   = float(g)
                f1.close()
                print('Updated cell quality from {} to {}. Thanks, boss!'.format(self.quality, g))

        return(g)

    def updatecontstat(self, t0 = 36*3600, t1 = 144*3600):

        try:
            cstat = self.cont_stat[0]
        except:
            cstat = self.cont_stat

        if np.size(np.shape(self.onTime))>1:
            onTs    = np.min(np.squeeze(self.onTime))
            offTs   = np.max(np.squeeze(self.offTime))
        else:
            onTs    = np.min(self.onTime)
            offTs   = np.max(self.offTime)

        try:
            tcheck = onTs <= t0 and offTs >= t1
        except:
            Tracer()()

        try:
            if not tcheck and cstat == 1:
            # change this cell to NOT continuous
                f1          = h5py.File(self.HDF5_tag[0], 'r+')
                data        = f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/cont_stat']
                data[...]   = float(0)
                f1.close()
                print('Changed this cell from continuous to not.')

            elif tcheck and cstat == 0:
                # change this cell to continuous
                f1          = h5py.File(self.HDF5_tag[0], 'r+')
                data        = f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/cont_stat']
                data[...]   = float(1)
                f1.close()
                print('Changed this cell to continuous.')
            else:
                print('Continuous status stays as {}.'.format(cstat))
        except:
            print('Caught at 535')
            Tracer()()

    def setonoff(self,keepprior=1, binsz = 3600):
        '''Plot the firing rate of the neuron in 1h bins and ask the user to click on the figure to indicate the on time and then again for the off time. If KEEPPRIOR is set as 1, this will preserve any currently existent on/off times that are between the newly indicated on/off time. If KEEPPRIOR is set as 0, any currently saved on/off times will be erased and replaced with whatever the user selects when running this code.'''
        
        global ix, ix2, cid
        ix  = None
        ix2 = None

        def __onclick(event):
            global ix, ix2, cid

            if ix is None:
                ix, iy = event.xdata, event.ydata
                print ( ' You clicked at x = {} hours. '.format(ix) )
            elif ix >0:
                ix2, iy = event.xdata, event.ydata
                print ( ' You clicked at x = {} hours. '.format(ix2) )

            

            bbb = plt.gcf()
            bbb.canvas.mpl_disconnect(cid)

            return ix, ix2

        def __pltfr(self):
            edges   = np.arange(0,max(self.time)+binsz,binsz)
            bins    = np.histogram(self.time,edges)
            hzcount = bins[0]
            hzcount = hzcount/3600
            hzcount[hzcount==0] = 'NaN'
            xbins   = bins[1]
            xbins   = xbins/3600

            plt.ion()
            with sns.axes_style("white"):
                fig1    = plt.figure()
                ax1     = plt.subplot(1,1,1)
                ax1.plot(xbins[:-1],hzcount)
            
            ax1.set_xlim(0,edges[-1]/3600)

            ylim    = ax1.get_ylim()[1]
            ylim    = plt.gca().get_ylim()[1]
        
            # make an array of the lights-off times
            lt_off = np.arange(12*3600,max(self.time)+12*3600,24*3600)
            #currentAxis = plt.gca()

            # cycle through the lights-off times and plot a transparent grey bar to indicate the dark hours
            for p in lt_off/3600:
                ax1.add_patch(patches.Rectangle((p, 0), 12, ylim, facecolor="grey",alpha=0.1, edgecolor="none"))

            # if self.deprived:
            #     plt.text(12,0.7*ylim,'Deprived')
            # else:
            #     plt.text(12,0.7*ylim,'Control')

            sns.set()
            sns.despine()
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Firing rate (Hz)')
            plt.show()

            # Set the on time for this cell:
 
            xlim    = plt.xlim()[1]
            temptxt = ax1.text(xlim*0.1, ylim*0.9, 'Click to set the on time:')
            
            global cid
            cid = fig1.canvas.mpl_connect('button_press_event', __onclick)


            return fig1


        #global ix, ix2
        fign = __pltfr(self)

        # set the on time  - - - - - - - - - - - - 
        while ix is None:
            fign.canvas.flush_events()
            time.sleep(0.05)

        newax   = fign.get_axes()[0]
        ylim    = newax.get_ylim()[1]
        xlim    = newax.get_xlim()[1]

        newax.set_ylim(0, ylim )
        newax.plot((ix,ix),(0,ylim),'g--')

        ontime  = ix 

        # And now deal with the off time: - - - - - 
        txt = newax.texts[1]
        txt.set_text('And now select the off time:')

        fign.show()
        fign.canvas.flush_events()

        cid = fign.canvas.mpl_connect('button_press_event', __onclick)

        while ix2 is None:
            fign.canvas.flush_events()
            time.sleep(0.05)

        offtime = ix2
        txt.set_visible(False)
        newax.plot((ix2,ix2),(0,ylim),'r--')

        fign.show()
        fign.canvas.flush_events()

        # overwrite the old on/off time data with the updated information you just provided:
        f1      = h5py.File(self.HDF5_tag[0], 'r+')

        ontime  *= 3600
        offtime *= 3600

        if keepprior == 1:

            if np.shape(f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime']) != ():
                # Case in which there are already on/off times (there _should_ be). 
                onTs    = np.squeeze(f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime'][:])
                offTs   = np.squeeze(f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime'][:])

                if np.size(onTs) > 1:
                    onTs    = onTs[1:]
                    offTs   = offTs[0:-1]
                    # get rid of any remaining on/off times prior to the new on time
                    onTs    = np.delete(onTs, onTs[ np.where( onTs<ontime )]) 
                    offTs   = np.delete(offTs, offTs[ np.where( offTs<ontime )])

                    # get rid of any remaining on/off times following the new off time
                    onTs    = np.delete(onTs, onTs[ np.where( onTs>offtime )]) 
                    offTs   = np.delete(offTs, offTs[ np.where( offTs>offtime )])
                elif np.size(onTs) == 1:
                    onTs    = []
                    offTs   = []
                elif np.size(onTs) == 0:
                    offTs   = []
                else:
                    print('line 660 neuron class')
                    Tracer()()

                ontdat  = np.append(ontime, onTs)
                ontdat  = np.squeeze([ontdat])
                offtdat = np.append(offTs, offtime)
                offtdat = np.squeeze([offtdat]) 

                del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime']
                del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime']

                f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime', data = [ontdat] )
                f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime', data = [offtdat] )

            elif np.shape(f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime']) == ():
                # Case in which there is no on/off time
                ontdat  = ontime
                offtdat = offtime

                del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime']
                del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime']

                f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime', data = [ontdat] )
                f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime', data = [offtdat] )


            

        elif keepprior == 0:
            # this is the case in which you want to WIPE the prior on/off times and start with a clean slate. 
            ontdat  = ontime
            offtdat = offtime

            del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime']
            del f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime']

            f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime', data = [ontdat] )
            f1.create_dataset('neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime', data = [offtdat] )

        f1.close()
        

        f1          = h5py.File(self.HDF5_tag[0], 'r')
        try:
            newontime   = f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/onTime'][:]
            newofftime  = f1['neurons/neuron_'+str(self.HDF5_tag[1])+'/offTime'][:]
            print('Updated to on: {} and off: {} '.format(newontime/3600, newofftime/3600))
            f1.close()

        except:
            print('Problem writing on/off times to disc. Line 696 in neuron_class.py')
            Tracer()()

        plt.close(fign)

        print('Saved new on/off times to disk.')














