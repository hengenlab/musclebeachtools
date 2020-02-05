#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to read ks outputs

Hengen Lab
Washington University in St. Louis
Version:  0.1


List of functions/class in mbt_neurons
load_np(filename, lpickle=False)
Neuron(sp_c, sp_t, qual, mwf, mwfs, max_channel)
ksout(datadir, filenum=0, prbnum=1, filt=None)

'''


import os.path as op
import glob
# import copy
try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')
from matplotlib import pyplot as plt
try:
    import seaborn as sns
except ImportError:
    raise ImportError('Run command : conda install seaborn')
import logging
import re


# start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#  console/file handlers
c_loghandler = logging.StreamHandler()
# f_loghandler = logging.FileHandler('mbt.log')
c_loghandler.setLevel(logging.INFO)
# c_loghandler.setLevel(logging.DEBUG)
# f_loghandler.setLevel(logging.WARNING)

# log formats
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
c_logformat = logging.Formatter(LOG_FORMAT)
# f_logformat = logging.Formatter(LOG_FORMAT)
c_loghandler.setFormatter(c_logformat)
# f_loghandler.setFormatter(f_logformat)

# add console/file handlers to logger
logger.addHandler(c_loghandler)
# logger.addHandler(f_loghandler)

# logger.info('Creating neuron list i ')
# logger.debug('Creating neuron list d ')
# logger.warning('Creating neuron list w ')
# logger.error('Creating neuron list e ')


# Util functions
def load_np(filename, lpickle=False):

    '''
    Function/wrapper to load numpy arrays

    load_np(filename, lpickle=False)

    Parameters
    ----------
    filename : Numpy filename with path
    lpickle : pickle True or False

    Returns
    -------
    np_out : numpy values

    Raises
    ------
    NameError
    filename not defined

    See Also
    --------

    Notes
    -----

    Examples
    --------
    spike_times = load_np('/home/kiranbn/spike_time.npy')

    '''

    try:
        np_out = np.load(filename, allow_pickle=lpickle)
    except NameError:
        print("load_np: filename is not defined")
        print("load_np: Error loading", filename)
        raise
    return np_out


class Neuron:
    '''
    Neuron class for use in Hengen Lab
    '''
    # fs, datatype, species, sex, age, start_time, end_time, clust_idx,
    # spike_time, quality, waveform, waveforms, peak_channel, region,
    # cell_type, mean_amplitude, behavior
    # fs = 25000
    datatype = 'npy'
    # species = str('')
    # sex = str('')
    # age = np.int16(0)
    # start_time = 0
    # end_time = 12 * 60 * 60
    behavior = None

    def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
                 fs=25000, start_time=0, end_time=12 * 60 * 60,
                 mwft=None,
                 sex=None, age=None, species=None,
                 on_time=None, off_time=None):
        '''
        The constructor for Neuron class.

        __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
                 fs=25000, start_time=0, end_time=12 * 60 * 60,
                 mwft=None,
                 sex=None, age=None, species=None)

        Parameters
        ----------
        sp_c : spike clusters
        sp_t : spike times
        qual : quality
        mwf : mean waveform
        mwfs : mean waveform spline
        max_channel : peak channel
        fs : sampling rate
        start_time : start time in seconds
        end_time : end time in seconds
        mwft : mean waveform tetrodes
        sex : sex of animal ('m' or 'f')
        age : age in days from birth
        species : "r" for rat or "m" for mouse
        on_time : start times of cell was activity
        off_time : end times of cell activity


        Returns
        -------

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        (Neuron(sp_c, sp_t, qual, mwf, mwfs, max_channel,
                fs=25000, start_time=0, end_time=12 * 60 * 60,
                mwft=None,
                sex=None, age=None, species=None))

        '''

        logger.debug('Neuron %d', sp_c)
        self.fs = fs
        self.start_time = start_time
        self.end_time = end_time
        self.sex = sex
        self.age = age
        self.species = species

        self.clust_idx = np.int16(sp_c)[0]
        self.spike_time = np.int64(sp_t)
        self.quality = np.int8(qual)
        self.waveform = mwf
        self.waveforms = mwfs
        self.waveform_tetrodes = mwft
        self.peak_channel = np.int16(max_channel)[0]
        self.region = str("")
        # Give on_time and off_time default values from start_time and end_time
        self.on_times = list([start_time])
        self.off_times = list([end_time])
        self.cell_type, self.mean_amplitude = \
            self.__find_celltypewithmeanamplitude()

    def __repr__(self):
        '''
        how to generate neuron
        '''
        return str("'Neuron(sp_cluster, sp_times, quality, meanwaveform,") +\
            str("meanwaveformspline, peak_channel)'")

    def __str__(self):
        '''
        Description of neuron
        '''
        return 'Neuron with (clust_idx=%d, quality=%d, peak_channel=%d)' \
               % \
               (self.clust_idx, self.quality, self.peak_channel)

    def __find_celltypewithmeanamplitude(self):
        '''
        Calculate cell type from mean waveform and mean amplitude

        __find_celltypewithmeanamplitude(self)

        Parameters
        ----------

        Returns
        -------
        cell_type : RSU or FSU
        mean_amplitude : mean amplitude of neuron

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        cell_type, mean_amplitude = __find_celltypewithmeanamplitude()

        '''

        temp = self.waveform
        fs = self.fs

        # To deal with both positive and negative spikes
        maxvalueidx = np.argmax(np.abs(temp))
        if temp[maxvalueidx] <= 0:
            bottom = np.argmin(temp)
            top = np.argmax(temp[bottom:]) + bottom
            peaklatency = ((top - bottom) * 1e3) / fs
            # Find mean amplitude
            mean_amplitude = np.abs(temp[bottom])
            # mean_amplitude = temp[top] - temp[bottom]
        elif temp[maxvalueidx] > 0:
            bottom = np.argmax(temp)
            top = np.argmin(temp[bottom:]) + bottom
            peaklatency = ((top - bottom) * 1e3) / fs
            # Find mean amplitude
            mean_amplitude = np.abs(temp[bottom])
            # mean_amplitude = temp[bottom] - temp[top]

        # Find cell type @Keith's paper
        cell_type = 'RSU' if peaklatency >= 0.4 else 'FS'

        return cell_type, mean_amplitude

    def get_behavior(self):
        '''
        Get sleep wake behavioral states of animal

        get_behavior(self)

        Parameters
        ----------

        Returns
        -------
        self.behavior : Get sleep wake states

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''
        logger.info('Not implemented')
        pass

    def shuffle_times(self, shuffle_alg=1, time_offset=10):
        '''
        Shuffle spike times of a neuron

        shuffle_times(self, type=1, time_offset)

        Parameters
        ----------
        shuffle_alg : type 1 (create random values of
          same length between min and max sample time)
        time_offset : This option is not implemented

        Returns
        -------
        self.spike_time : Returns shuffled spike times

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        self.spike_time = n1[0].shuffle_times(self, shuffle_alg=1,
                                              time_offset=10)

        '''

        logging.info('Shuffling spike times')
        assert shuffle_alg in [1], 'Unkown type in shuffle_times()'
        max_t = np.max(self.spike_time)
        min_t = np.min(self.spike_time)
        # add offset
        if shuffle_alg == 1:
            logging.debug('Shuffling spike times using random')
            self.spike_time = \
                np.sort(
                        np.random.randint(low=min_t,
                                          high=max_t,
                                          size=np.size(self.spike_time),
                                          dtype='int64'))

    def plot_wf(self):
        '''
        Plot mean waveform of a neuron

        plot_wf(self)

        Parameters
        ----------

        Returns
        -------

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].plot_wf()

        '''

        plt.ion()
        with sns.axes_style("white"):
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.plot(self.waveforms, color='#6a88f7')
            # ax.set_ylim(bottom=0)
            # ax.set_xlim(left=0)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.text(30, 0.7*ax.get_ylim()[1],
                    str("idx ") + str(self.clust_idx))
        sns.despine()

    def isi_hist(self, start=False, end=False, isi_thresh=0.1, nbins=101,
                 lplot=1):
        # copied from musclebeachtools
        '''
        Return a histogram of the interspike interval (ISI) distribution.
        This is typically used to evaluate whether a spike train exhibits
        a refractory period and is thus consistent with a
        single unit or a multi-unit recording.
        This function will plot the bar histogram of that distribution
        and calculate the percentage of ISIs that fall under 2 msec.

        isi_hist(self, start=False, end=False, isi_thresh=0.1, nbins=101)

        Parameters
        ----------
        start : Start time (default self.start_time)
        end : End time (default self.end_time)
        isi_thresh : isi threshold (default 0.1)
        nbin : Number of bins (default 101)
        lplot : To plot or not (default lplot=1, plot isi)

        Returns
        -------
        ISI : spike time difference (a[i+1] - a[i]) along axis

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].isi_hist(isi_thresh=0.1, nbins=101, lplot=1)

        '''

        # For a view of how much this cell is like a "real" neuron,
        # calculate the ISI distribution between 0 and 100 msec.

        logger.info('Calculating ISI')
        # Sample time to time in seconds
        time_s = self.spike_time/self.fs

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        # Calulate isi
        idx = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
        ISI = np.diff(time_s[idx])

        # plot histogram and calculate contamination
        edges = np.linspace(0, isi_thresh, nbins)
        hist_isi = np.histogram(ISI, edges)

        # Calculate contamination percentage
        contamination = 100*(sum(hist_isi[0][0:int((0.1/isi_thresh) *
                             (nbins-1)/50)])/sum(hist_isi[0]))
        contamination = round(contamination, 2)
        cont_text = 'Contamination is {} percent.' .format(contamination)
        logger.info('Contamination is {} percent.' .format(contamination))

        plt.ion()
        if lplot:
            with sns.axes_style("white"):
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                # ax.bar(edges[1:]*1000-0.5, hist_isi[0], color='#6a79f7')
                ax.bar(edges[1:]*1000-0.5, hist_isi[0], color='#0b559f')
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                ax.set_xlabel('ISI (ms)')
                ax.set_ylabel('Number of intervals')
                ax.text(30, 0.7*ax.get_ylim()[1], cont_text)
            sns.despine()
        return ISI

    def plotFR(self, binsz=3600, start=False, end=False):
        # copied from musclebeachtools
        '''
        This will produce a firing rate plot for all loaded spike times
        unless otherwise specified binsz, start, end are in seconds

        plotFR(self, binsz=3600, start=False, end=False)

        Parameters
        ----------
        binsz : Bin size (default 3600)
        start : Start time (default self.start_time)
        end : End time (default self.end_time)

        Returns
        -------
        hzcount : count per bins

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].plotFR(binsz=3600, start=False, end=False)

        '''

        logger.info('Plotting firing rate')
        # Sample time to time in seconds
        time_s = (self.spike_time / self.fs)

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        edges = np.arange(start, end + binsz, binsz)
        bins = np.histogram(time_s, edges)
        hzcount = bins[0]/binsz
        # hzcount[hzcount == 0] = 'NaN'
        xbins = bins[1]/binsz

        plt.ion()
        with sns.axes_style("white"):
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            # print(hzcount)
            ax.plot(xbins[:-1], hzcount, color='#703be7')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Firing rate (Hz)')

        sns.despine()
        # plt.show()
        return hzcount

    def set_qual(self, qual):

        '''
        This function allows to change quality of neuron

        set_qual(self, qual)

        Parameters
        ----------
        qual : Quality values should be 1, 2, 3 or 4
               1 : Good
               2 : Good but some contamination
               3 : Multiunit contaminated unit
               4 : Noise unit

        Returns
        -------

        Raises
        ------
        ValueError if qual is not 1, 2, 3 or 4

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n[0].set_qual(qual)

        '''

        logger.info('Changing quality')

        # Check qual value
        if not ((qual >= 1) and (qual <= 4)):
            logger.error('1 : Good')
            logger.error('2 : Good but some contamination')
            logger.error('3 : Multiunit contaminated unit')
            logger.error('4 : Noise unit')
            raise \
                ValueError('Error:quality values are 1, 2, 3 or 4 given {}'
                           .format(qual))

        logger.debug('Quality is of unit %d is %d', self.clust_idx, qual)
        self.quality = np.int8(qual)

    def set_onofftimes(self, ontimes, offtimes):

        '''
        This function allows to change on off time of neuron

        set_onofftimes(self, ontimes, offtimes)

        Parameters
        ----------
        ontimes : list of ontimes
        offtimes : list of offtimes

        Returns
        -------

        Raises
        ------
        ValueError if on off times is empty
        ValueError if on off time has not equal size
        ValueError if ontime > offtime
        ValueError if ontime or offtime list not contain integer or float

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n[0].set_onofftimes(ontimes, offtimes)

        '''

        logger.info('Changing on off times')

        # numpy array to list
        if isinstance(ontimes, np.ndarray):
            ontimes = ontimes.tolist()
        if isinstance(offtimes, np.ndarray):
            offtimes = offtimes.tolist()
        print("ontimes type ", type(ontimes))
        print("offtimes type ", type(offtimes))

        # convert to list
        if not isinstance(ontimes, list):
            if ((isinstance(ontimes, float)) or (isinstance(ontimes, int))):
                ontimes = list([ontimes])
            elif (len(ontimes) > 1):
                ontimes = list(ontimes)
            logger.info('ontimes type type(ontimes)')
        if not isinstance(offtimes, list):
            if ((isinstance(offtimes, float)) or
                    (isinstance(offtimes, int))):
                offtimes = list([offtimes])
            elif (len(offtimes) > 1):
                offtimes = list(offtimes)
            logger.info('ontimes type type(offtimes)')

        # check ontimes is not empty
        if (len(ontimes) == 0):
            raise ValueError('Error : ontimes is empty')
        if (len(offtimes) == 0):
            raise ValueError('Error : offtimes is empty')

        # Check ontimes has a corresponding offtimes value
        if not ((len(ontimes)) == (len(offtimes))):
            raise \
                ValueError('Error: on off times not same size given {} and {}'
                           .format(len(ontimes), len(offtimes)))

        # Check time is ascending
        for on_tmp, off_tmp in zip(ontimes, offtimes):
            if (on_tmp > off_tmp):
                raise ValueError('Error: ontime {} > offtime {}'
                                 .format(on_tmp, off_tmp))
            if not ((isinstance(on_tmp, float)) or (isinstance(on_tmp, int))):
                raise ValueError('Error: ontime values not float')
            if not ((isinstance(off_tmp, float))
                    or (isinstance(off_tmp, int))):
                raise ValueError('Error: ontime values not float')

        self.on_times = ontimes
        self.off_times = offtimes


def n_getspikes(neuron_list, start=False, end=False):

    '''
    Extracts spiketimes to a list from neuron_list
    Unless otherwise specified start, end are in seconds

    n_getspikes(neuron_list, start=False, end=False)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    start : Start time (default self.start_time)
    end : End time (default self.end_time)

    Returns
    -------
    spiketimes_allcells : List of all spiketimes

    Raises
    ------
    ValueError if neuron list is empty

    See Also
    --------

    Notes
    -----

    Examples
    --------
    n_getspikes(neuron_list, start=False, end=False)

    '''

    logger.info('Extracting spiketimes to a list from neuron_list')
    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.info('start and end is %s and %s', start, end)

    # Create empty list
    spiketimes_allcells = []

    # Loop through and get spike times
    for i, _ in enumerate(neuron_list):
        logger.debug('Getting spiketimes for cell %d', str(i))

        # get spiketimes for each cell and append
        spiketimes = neuron_list[i].spike_time / neuron_list[i].fs
        spiketimes = spiketimes[(spiketimes > start) & (spiketimes < end)]
        spiketimes_allcells.append(spiketimes)

    return spiketimes_allcells


def n_spiketimes_to_spikewords(neuron_list, binsz=0.02,
                               start=False, end=False,
                               binarize=0):
    '''
    This function converts spiketimes to spikewords
    Unless otherwise specified binsz, start, end are in seconds

    n_spiketimes_to_spikewords(neuron_list, binsz=0.02, binarize=0)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    binsz : Bin size (default 0.02 (20 ms))
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    binarize : Get counts (default 0) in bins,
    if binarize is 1,binarize to 0 and 1

    Returns
    -------
    hzcount : count per bins
    spikewords_array : array of spikewords row x column (time bins x cells)

    Raises
    ------
    ValueError if neuron list is empty

    See Also
    --------

    Notes
    -----

    Examples
    --------
    n_spiketimes_to_spikewords(neuron_list, binsz=0.02, binarize=0)
    '''

    # Constants
    conv_mills = 1000.0

    logger.info('Converting spiketime to spikewords')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')
    # check binsize is not less than 1ms
    if (binsz < 0.001):
        raise ValueError('Bin size is less than 1millisecond')
    # binarize is only 0 or 1
    if (binarize not in [0, 1]):
        raise ValueError('Binarize takes only values 0 or 1')

    # Get time
    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.debug('start and end is %s and %s', start, end)

    # Get spiketime list
    spiketimes = n_getspikes(neuron_list, start=start, end=end)

    # convert time to milli seconds
    start = start * conv_mills
    end = end * conv_mills
    binsz = binsz * conv_mills

    # startime in bins
    binrange = np.arange(start, end, binsz)
    n_cells = len(spiketimes)

    # initialize array
    spikewords_array = np.zeros([n_cells, binrange.shape[0]-1])

    # loop over cells and find counts/binarize
    for i in range(n_cells):

        # spiketimes in seconds to ms
        spiketimes_cell = np.asarray(spiketimes)[i] * conv_mills
        counts, bins = np.histogram(spiketimes_cell, bins=binrange)

        # binarize the counts
        if binarize == 1:
            counts[counts > 0] = 1
        spikewords_array[i, :] = counts

    return(spikewords_array.astype(np.int16))


# loading function
def ksout(datadir, filenum=0, prbnum=1, filt=None):
    '''
    load Kilosort output from ntksorting
    returns list of Neuron class objects
    For example
    n1[0].quality gives neuron 0's quality

    Function to load Kilosort output

    ksout(datadir, filenum=1, prbnum=1, filt=None)


    Parameters
    ----------
    datadir : Location of output files
    filenum : File number if there is many blocks (default 0)
    prbnum : Probe number (default 1). Range 1-10.
    filt : filter by quality. filt=[1], loads only quality 1 neurons.


    Returns
    -------
    n1 : All neurons as a list. For example n1[0] is first neuron.

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    datadir = "/hlabhome/kiranbn/Animalname/final/"
    n1 = ksout(datadir, filenum=0, prbnum=1, filt=[1, 3])


    '''

    # filt to empty list
    if filt is None:
        filt = []

    # constants
    max_filenum = 100

    # datadir
    logger.info("datadir %s", datadir)
    assert op.exists(datadir), "Please recheck datadir"

    # filenum
    logger.info("filenum %s", filenum)
    assert 0 <= filenum <= max_filenum, "Please check filenum"

    # prbnum
    logger.info("prbnum %s", prbnum)
    assert 1 <= prbnum <= 10, "We have maximum 10 probes currently"

    flist = np.sort(glob.glob1(datadir, "*spike_times.npy"))[filenum]
    logger.debug("flist %s", flist)
    basename = flist.replace("spike_times.npy", "")
    logger.debug("basename %s", basename)
    basenametmp = re.sub('chg_'r'\d_', 'chg_#_', basename)
    logger.debug("basenametmp %s", basenametmp)
    basename = re.sub('_#', '_'+str(prbnum), basenametmp)
    assert op.exists(op.join(datadir, basename+"spike_times.npy")), \
        "Please recheck probe number, cannot find spike_times file"
    logger.info("basename %s", basename)

    if basename:
        spike_times = \
            load_np(op.join(datadir,
                    basename+"spike_times.npy"))
        spike_clusters = \
            load_np(op.join(datadir,
                    basename+"spike_clusters.npy"))
        cluster_quals = \
            load_np(op.join(datadir,
                    basename+"basicclusterqual.npy"))
        # print(cluster_quals)
        cluster_mwf = \
            load_np(op.join(datadir,
                    basename+"mean_waveform.npy"))
        cluster_mwfs = \
            load_np(op.join(datadir,
                    basename+"mean_waveform_spline.npy"))
        cluster_maxchannel = \
            load_np(op.join(datadir,
                    basename+"max_channel.npy"))
        sampling_rate = \
            load_np(op.join(datadir,
                    basename+"sampling_rate.npy"))[0][0]

    unique_clusters = np.unique(spike_clusters)

    # Sampling rate
    logger.debug('Finding sampling rate')
    fs_list = [20000, 25000]
    assert sampling_rate in fs_list, "fs (sampling rate) not in list"
    # Neuron.fs = sampling_rate
    n = []

    # Start and end time
    logger.debug('Finding start and end time')
    start_time = re.search('%s(.*)%s' % ('_times_', '-timee_'),
                           basename).group(1)
    end_time = re.search('%s(.*)%s' % ('-timee_', '_length_'),
                         basename).group(1)
    # Convert to seconds
    end_time = (np.double(np.int64(end_time) - np.int64(start_time))/1e9)
    # reset start to zero
    start_time = np.double(0.0)
    print((end_time - start_time))
    assert (end_time - start_time) > 1.0, \
        'Please check start and end time is more than few seconds apart'
    logger.info('Start and end times are %f and %f', start_time, end_time)
    # Neuron.start_time = start_time
    # Neuron.end_time = end_time

    # Loop through unique clusters and make neuron list
    for i in unique_clusters:
        # print("i ", i)
        # print("qual ", cluster_quals[i])
        if len(filt) == 0:
            lspike_clust = np.where(spike_clusters == i)
            # sp_c = spike_clusters[lspike_clust]
            sp_c = np.unique(spike_clusters[lspike_clust])
            sp_t = spike_times[lspike_clust]
            qual = cluster_quals[i]
            mwf = cluster_mwf[i]
            mwfs = cluster_mwfs[i]
            max_channel = cluster_maxchannel[i]
            # def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
            #              fs=25000, start_time=0, end_time=12 * 60 * 60,
            #              mwft=None,
            #              sex=None, age=None, species=None):
            n.append(Neuron(sp_c, sp_t, qual, mwf, mwfs, max_channel,
                            fs=sampling_rate,
                            start_time=start_time, end_time=end_time))
        elif len(filt) > 0:
            if cluster_quals[i] in filt:
                lspike_clust = np.where(spike_clusters == i)
                sp_c = np.unique(spike_clusters[lspike_clust])
                sp_t = spike_times[lspike_clust]
                qual = cluster_quals[i]
                mwf = cluster_mwf[i]
                mwfs = cluster_mwfs[i]
                max_channel = cluster_maxchannel[i]
                n.append(Neuron(sp_c, sp_t, qual, mwf, mwfs, max_channel,
                                fs=sampling_rate,
                                start_time=start_time, end_time=end_time))

    logger.info('Found %d neurons', len(n))
    # neurons = n.copy()
    # neurons = copy.deepcopy(n)
    # return neurons
    return n


if __name__ == "__main__":

    datadir = "/hlabhome/kiranbn/neuronclass/t_lit_EAB50final/"
    n1 = ksout(datadir, filenum=0, prbnum=4, filt=[1, 3])
