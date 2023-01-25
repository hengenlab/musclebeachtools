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
import os
# import copy
try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')
try:
    import neuraltoolkit as ntk
except ImportError:
    raise \
        ImportError('''Run command :
                       git clone https://github.com/hengenlab/neuraltoolkit.git
                       cd neuraltoolkit
                       pip install .''')
from matplotlib import pyplot as plt
# from matplotlib.widgets import TextBox
# from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
try:
    import seaborn as sns
except ImportError:
    raise ImportError('Run command : conda install seaborn')
import logging
import re
from datetime import datetime
import sys
import time
try:
    import scipy as sc
except ImportError:
    raise ImportError('Run command : pip install scipy')
from scipy.signal import fftconvolve
# try:
#     import joblib
# except ImportError:
#     raise ImportError('Run command : pip install joblib')
try:
    # from sklearn import datasets
    from sklearn.model_selection import train_test_split
    # from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.model_selection import cross_val_score
    # from sklearn.externals import joblib
except ImportError:
    raise ImportError('Run command : pip install sklearn')
try:
    import xgboost as xgb
except ImportError:
    m = 'https://raw.githubusercontent.com/Homebrew/install/master/install.sh'
    if sys.platform == "darwin":
        print('\n\nIn mac please install brew and gcc first')
        print('bash -c "$(curl -fsSL', m, ')"')
        print('Run command : brew install gcc')
        print('Run command : conda install -c conda-forge xgboost\n\n')
        raise \
            ImportError('Run: conda install -c conda-forge xgboost'
                        + '\nFirst check message above install brew and gcc')
    else:
        raise \
            ImportError('Run: conda install -c conda-forge xgboost')

# start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#  console/file handlers
c_loghandler = logging.StreamHandler()
# f_loghandler = logging.FileHandler('mbt.log')
c_loghandler.setLevel(logging.ERROR)
# c_loghandler.setLevel(logging.INFO)
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


def wf_comparison(waveform_1, waveform_2):

    '''
    This function calculate wf comparison of same size

    sse, mse, rmse = wf_comparison(waveform_1, waveform_2)

    Parameters
    ----------
    waveform_1 : First waveform
    waveform_2 : Second waveform

    Returns
    -------
    sse, mse, rmse

    Raises
    ------
    ValueError : waveform shapes are not equal
    ValueError : waveform shape not 1
    ValueError : waveform size is 0

    See Also
    --------

    Notes
    -----

    Examples
    --------
    sse, mse, rmse = wf_comparison(waveform_1, waveform_2)

    '''

    logger.info('Calculating waveform comparison')

    # Waveform size is not zero
    if (waveform_1.size == 0):
        raise ValueError('waveform_1 size is 0')
    if (waveform_2.size == 0):
        raise ValueError('waveform_2 size is 0')
    # Waveform shape is 1
    if (waveform_1.ndim != 1):
        raise ValueError('waveform_1 shape is not 1')
    if (waveform_2.ndim != 1):
        raise ValueError('waveform_2 shape is not 1')
    # Check wf has same length
    if (waveform_1.shape[0] != waveform_2.shape[0]):
        raise ValueError('waveform_1 and waveform_2 are not same shape')

    # Calculate all
    # ckbn todo add more properties
    sse = np.sum((waveform_1 - waveform_2)**2)
    mse = ((waveform_1 - waveform_2)**2).mean(axis=0)
    rmse = np.sqrt(np.mean((waveform_1 - waveform_2)**2))

    return sse, mse, rmse


def euclidean_distance(row1, row2, ch=None, corr_fact=0.95):

    '''
    Calculate distance between row1 and row2

    distance = euclidean_distance(row1, row2, ch=None, corr_fact=0.95)

    Parameters
    ----------
    row1 : first vector
    row2 : second vector
    ch : channel list
    corr_fact : correlation lower limit

    Returns
    -------
    distance

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    distance = euclidean_distance(row1, row2, ch=None, corr_fact=0.95)

    '''

    # print("row 1 ", row1)
    # print("row 2 ", row2)
    # print("sh row1 ", row1.shape, " sh ", row2.shape)
    # print("len ", len(row1)-1)
    distance = 0.0

    # if ((stats.ks_2samp(row1[5:-7], row2[5:-7])[1] >  0.1) and
    #    (np.corrcoef(row1[5:-7], row2[5:-7])[0, 1] >  0.1)):
    # if ((stats.ks_2samp(row1[5:-7], row2[5:-7])[1] >  0.5)):
    if ((np.corrcoef(row1[5:-7], row2[5:-7])[0, 1] > corr_fact)):
        # if (int(row1[-2] + ch) == int(row2[-2] + ch)):
        # print("int(row2[-2] ", int(row2[-2]), " int(row1[-2] ",
        #       int(row1[-2]), " ch ", ch)
        if (int(row2[-2]) in ch):
            # print("\tint(row2[-2] ", int(row2[-2]),
            #       " int(row1[-2] ", int(row1[-2]), " ch ", ch)
            for i in range(len(row1)-2):
                # print("i ", i, " row1 ", row1[i], " row2 ", row2[i],
                #       " ch ", int(row1[-2]), " ", int(row2[-2]))
                distance += (row1[i] - row2[i])**2
                # print("distance ", distance)
                # print("ch ", int(row1[-2]), " ", int(row2[-2]))
        else:
            distance = 1000000000.0
    else:
        distance = 1000000000.0
    return np.sqrt(distance)


def track_blocks(fl, ch_grp_size=4, maxqual=3, corr_fact=0.97,
                 lsaveneuron=0, lsavefig=0):

    '''
    Calculate key between blocks

    track_list = \
        track_blocks(fl, ch_grp_size=4, maxqual=3, corr_fact=0.97,
                     lsaveneuron=0, lsavefig=0)


    Parameters
    ----------
    fl : file list
    group size : 4 for tetrodes
    maxqual : maximum quality
    corr_fact : correlation lower limit
    lsaveneuron : 1 to save neuron, 0 no not save
    lsavefig : 1 to save fig, 0 not savefig

    Returns
    -------
    saves new neron file with name "_withkeys.npy"
    also returns key value pair as list

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    track_list = \
        track_blocks(fl, ch_grp_size=4, maxqual=3, corr_fact=0.97,
                     lsaveneuron=0, lsavefig=0)

    '''
    track_list = None
    track_list = []

    if lsavefig:
        plt.rcParams.update({'font.size': 4})
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False

    print("File list ", fl)
    for fl_idx in range(len(fl)-1):
        print("Loading neurons")
        neurons1 = None
        neurons2 = None
        # print("fl[fl_idx] ", fl[fl_idx])
        # print("fl[fl_idx+1] ", fl[fl_idx+1])
        neurons1 = np.load(fl[fl_idx], allow_pickle=True)
        neurons2 = np.load(fl[fl_idx+1], allow_pickle=True)

        print("Filtering neurons")
        wf_e =\
            [np.append(np.mean(neuron.wf_b, axis=1),
             [int(neuron.peak_channel), int(neuron.clust_idx)])
             for neuron in neurons1 if int(neuron.quality) <= maxqual]
        wf_e = np.asarray(wf_e).T
        # wf_e_l = [neuron.clust_idx
        #           for neuron in neurons1 if neuron.quality < 3]
        # print("sh neurons1 ", len(neurons1), " sh wf_e ", wf_e.shape)

        wf_b =\
            [np.append(np.mean(neuron2.wf_b, axis=1),
             [int(neuron2.peak_channel), int(neuron2.clust_idx)])
             for neuron2 in neurons2 if int(neuron2.quality) <= maxqual]
        wf_b = np.asarray(wf_b).T
        # print("sh neurons2 ", len(neurons2), " sh wf_b ", wf_b.shape)
        # wf_b_l = [neuron.clust_idx for neuron in neurons2 if
        #           neuron.quality < 3]
        # print("sh wf_b ", wf_b.shape)
        # print("sh wf_b0 ", wf_b[0,:].shape)
        # print("sh wf_b1 ", wf_b[:,0].shape)
        # print("sh wf_e ", wf_e.shape)
        # print("sh wf_e0 ", wf_e[0,:].shape)
        # print("sh wf_e1 ", wf_e[:,0].shape)
        # wf_b_mean = np.mean(wf_b, axis=0)
        dist_list = np.zeros((wf_b.shape[1], wf_e.shape[1]))
        for j in range(wf_b.shape[1]):
            for i in range(wf_e.shape[1]):
                # print(" ", i, " j ", j)
                ch_i = wf_e[-2, i]
                # if wf_b[-2, j] in np.arange(ch_i -(ch_i%ch_grp_size), ch_i +
                # (ch_grp_size - (ch_i%ch_grp_size))):
                # print("ch_i ", ch_i)
                dist_list[j, i] = \
                    euclidean_distance(wf_e[:, i],
                                       wf_b[:, j],
                                       np.arange(ch_i -
                                                 (ch_i % ch_grp_size),
                                                 ch_i +
                                                 (ch_grp_size -
                                                  (ch_i % ch_grp_size))),
                                       corr_fact=corr_fact)

        # print("dist_list ", dist_list)
        X = np.asarray(dist_list)
        # print("X ", X)
        # print("X ", np.int32(X))
        # print("sh X ", X.shape)

        keys = np.argmin(X, axis=0)
        # print("keys ", keys)
        # print("sh keys ", keys.shape)
        # print(np.where(np.asarray(keys) >0))
        K = np.where(np.asarray(keys) > 0)[0]

        # check Key is unique
        _, K_counts = np.unique(K, return_counts=True)
        if (len(np.where(K_counts > 1)[0]) == 0):
            if 0:
                print("unique")
        else:
            # print("not unique")
            raise RuntimeError('Not unique')

        V = keys[K]
        K1 = []
        V1 = []
        for k_i, v_i in zip(K, V):
            # print(neurons1[k_i].clust_idx, " ", neurons2[v_i].clust_idx,
            #       " q ", neurons1[k_i].quality, " ", neurons2[v_i].quality)
            # print("check ", wf_e[-1:, k_i], " ", wf_b[-1:, v_i])
            K1.extend(wf_e[-1:, k_i])
            V1.extend(wf_b[-1:, v_i])

        # print(K, V)
        # print(K1, V1)
        # K = K1.copy()
        # V = V1.copy()
        K = np.array(K1, dtype="int")
        V = np.array(V1, dtype="int")
        print("Keys")
        print(K)
        print(V)
        track_list.append(K)
        track_list.append(V)
        plt.figure(fl_idx + 1, figsize=(20, 20))
        # print("key_order ", key_order)
        key_order = None
        key_order = []
        for ind, i in enumerate(K):
            plt.subplot(int(np.ceil(np.sqrt(len(K)))),
                        int(np.ceil(np.sqrt(len(K)))), ind+1)
            c1 = [cell for cell in neurons1 if cell.clust_idx == int(K[ind])]
            c2 = [cell for cell in neurons2 if cell.clust_idx == int(V[ind])]
            # plt.plot(neurons1[i].waveform)
            plt.plot(c1[0].waveform)
            # plt.plot(neurons2[V[ind]].waveform,
            #          label=str("ch " +
            #                    str(neurons1[i].peak_channel) +
            #                    " " +
            #                    str(neurons2[V[ind]].peak_channel) +
            #                    " id " +
            #                    str(neurons1[i].clust_idx) +
            #                    " " +
            #                    str(neurons2[V[ind]].clust_idx) +
            #                    " q " +
            #                    str(neurons1[i].quality) +
            #                    " " +
            #                    str(neurons2[V[ind]].quality)))
            plt.plot(c2[0].waveform,
                     label=str("c " +
                               str(c1[0].peak_channel) +
                               " " +
                               str(c2[0].peak_channel) +
                               "\ni " +
                               str(c1[0].clust_idx) +
                               " " +
                               str(c2[0].clust_idx) +
                               "\nq " +
                               str(c1[0].quality) +
                               " " +
                               str(c2[0].quality)))

            # key_order.append([neurons1[i].clust_idx,
            #                   neurons2[V[ind]].clust_idx,
            #                   str(fl[fl_idx+1])])
            key_order.append([c1[0].clust_idx, c2[0].clust_idx,
                              str(fl[fl_idx+1])])
            # plt.plot(neuron.waveform,
            #          label=str(str(neuron.peak_channel) +
            #                    " " +
            #                    str(neuron.clust_idx)))
            # if ind == 0:
            #     plt.title("block " + str(fl_indx + 1))
            # # plt.legend(frameon=False, loc='best', prop={'size': 2})
            # plt.legend(frameon=False,
            #            loc='lower right', prop={'size': 4,
            #                                     'weight':'bold'})
            plt.legend(frameon=False, loc='best', prop={'size': 6,
                                                        'weight': 'bold'})
            plt.xticks([])
            # # plt.yticks([])
            plt.xlabel("")
            # plt.box(False)
        # print("nnnnnn ", op.splitext(fl[fl_idx]))
        # np.save(op.splitext(fl[fl_idx])[0] + "_" + str(fl_idx) +
        # "_new" + ".npy", np.asarray(key_order))

        # Add keys
        for i in neurons1:
            for k in key_order:
                # print(k[0], i.clust_idx)
                if int(i.clust_idx) == int(k[0]):
                    # print("i.clust_idx ", i.clust_idx, " k[0] ", k[0])
                    i.key = k
        if lsaveneuron:
            np.save(op.splitext(fl[fl_idx])[0] + "_withkeys.npy", neurons1)
        if lsavefig:
            # plt.savefig(op.splitext(fl[fl_idx])[0] + "_withkeys.png",
            #             bbox_inches='tight',pad_inches=-1)
            plt.savefig(op.splitext(fl[fl_idx])[0] + "_withkeys.png",
                        pad_inches=-1)

    # print("key_order ", key_order)
    if not lsavefig:
        plt.show()
    plt.clf()
    plt.close('all')
    return track_list


def wf_sim(wf1, wf2, ltype=1):

    '''
    Find similarity between two waveforms

    wf_sim(wf1, wf2, ltype=1)

    Parameters
    ----------
    wf1 : First waveform
    wf2 : Second waveform
    ltype : Default 1 (ks_2samp), 2 corrcoef

    Returns
    -------
    sim_fact : Similarity factor

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    sim_fact = wf_sim(wf1, wf2, ltype=1)

    '''

    if ltype == 1:
        w, p = sc.stats.ks_2samp(wf1, wf2)
        return p
    elif ltype == 2:
        p = np.corrcoef(wf1, wf2)[0][1]
        return p
    else:
        raise ValueError('Unknown ltype {}'.format(ltype))


def ecube_realtimestring_to_epoch(e_time_string):

    '''
    Convert ecube time string to epoch

    ecube_realtimestring_to_epoch(e_time_string)

    Parameters
    ----------
    e_time_string : ecube time string

    Returns
    -------
    unix time, or seconds/milliseconds since the 1970 epoch

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    t_seconds = ecube_realtimestring_to_epoch('2019-07-01_00-40-01')

    '''

    dt = datetime.strptime(e_time_string.replace("_", " "),
                           "%Y-%m-%d %H-%M-%S")
    return time.mktime(dt.timetuple())


class Neuron:
    '''
    Neuron class for use in Hengen Lab
    '''
    # fs, datatype, species, sex, birthday, animal_name,
    # start_time, end_time, clust_idx,
    # spike_time, quality, waveform, waveforms, peak_channel, region,
    # cell_type, mean_amplitude, behavior
    # fs = 25000
    datatype = 'npy'
    # species = str('')
    # sex = str('')
    # genotype,
    # notes,
    # birthday datetime.datetime(2019, 12, 24, 7, 30)
    # animal_name = str('')
    # start_time = 0
    # end_time = 12 * 60 * 60
    # behavior = None
    behavior = np.zeros((2, 6))

    @classmethod
    def update_behavior(cls, behavior):
        cls.behavior = behavior
        return cls.behavior

    def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
                 fs=25000, start_time=0, end_time=12 * 60 * 60,
                 mwft=None,
                 sex=None, birthday=None, species=None,
                 animal_name=None,
                 genotype=None,
                 expt_cond=None,
                 on_time=None, off_time=None,
                 rstart_time=None, rend_time=None,
                 estart_time=None, eend_time=None,
                 sp_amp=None,
                 region_loc=None,
                 wf_b=None, wf_e=None,
                 key=None,
                 qual_prob=None):
        '''
        The constructor for Neuron class.

        __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
                 fs=25000, start_time=0, end_time=12 * 60 * 60,
                 mwft=None,
                 sex=None, birthday=None, species=None,
                 animal_name=None,
                 genotype=None,
                 expt_cond=None,
                 on_time=None, off_time=None,
                 rstart_time=None, rend_time=None,
                 estart_time=None, eend_time=None,
                 sp_amp=None,
                 region_loc=None,
                 wf_b=None, wf_e=None,
                 key=None,
                 qual_prob=None)


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
        birthday : birthday datetime.datetime(2019, 12, 24, 7, 30)
        species : "r" for rat or "m" for mouse
        animal_name : name of animal 8 characters, UUU12345
        genotype : "wt", te4
        expt_cond : "monocular deprivation",
        on_time : start times of cell was activity
        off_time : end times of cell activity
        rstart_time : real start time of recorded file in this block, string
        rend_time : real end time of last recorded file in this block, string
        estart_time : start time in nano seconds ecube
        eend_time : end time in nano seconds ecube
        sp_amp : spike amplitudes
        region_loc : implant location
        wf_b : waveforms begin
        wf_e : waveforms end
        key : link clusters
        qual_prob : quality probability


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
                sex=None, birthday=None, species=None,
                animal_name=None,
                genotype=None,
                expt_cond=None,
                on_time=None, off_time=None,
                rstart_time=None, rend_time=None,
                estart_time=None, eend_time=None,
                sp_amp=None,
                region_loc=None,
                wf_b=None, wf_e=None,
                key=None,
                qual_prob=None)

        '''

        logger.debug('Neuron %d', sp_c)
        self.clust_idx = np.int16(sp_c)[0]
        self.spike_time = np.int64(sp_t)
        self.quality = np.int8(qual)
        self.waveform = mwf
        self.waveforms = mwfs
        self.peak_channel = np.int16(max_channel)[0]

        self.fs = fs
        self.start_time = start_time
        self.end_time = end_time

        if mwft is not None:
            self.waveform_tetrodes = mwft

        if sex is not None:
            self.sex = sex
        else:
            self.sex = ''
        if birthday is not None:
            self.birthday = birthday
        else:
            self.birthday = datetime(1970, 1, 1, 00, 00)
        if species is not None:
            self.species = species
        else:
            self.species = ''
        if animal_name is not None:
            self.animal_name = animal_name
        else:
            self.animal_name = 'UUU12345'
        if genotype is not None:
            self.genotype = genotype
        else:
            self.genotype = ''

        if expt_cond is not None:
            self.expt_cond = expt_cond
        else:
            self.expt_cond = 'experimental condition'

        # Give on_time and off_time default values from start_time and end_time
        if on_time is None:
            self.on_times = list([start_time])
        else:
            self.on_times = on_time
        if off_time is None:
            self.off_times = list([end_time])
        else:
            self.off_times = off_time

        if rstart_time is not None:
            self.rstart_time = str(rstart_time)
        if rend_time is not None:
            self.rend_time = str(rend_time)

        if estart_time is not None:
            self.estart_time = np.int64(estart_time)
        if eend_time is not None:
            self.eend_time = np.int64(eend_time)

        if sp_amp is not None:
            self.spike_amplitude = np.int32(sp_amp)

        if region_loc is not None:
            self.region = region_loc

        if wf_b is not None:
            self.wf_b = wf_b
        if wf_e is not None:
            self.wf_e = wf_e

        if key is not None:
            self.key = np.asarray([np.int16(key[0]), np.int16(key[1]),
                                   str(key[2])])
        else:
            self.key = np.asarray([np.int16(-1), np.int16(-1), str("")])

        if qual_prob is not None:
            self.qual_prob = qual_prob
        else:
            self.qual_prob = -1.0

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
        return '''Neuron with (animal_name=%s, region=%s,
             genotype=%s, species=%s, sex=%s,
             clust_idx=%d, quality=%d, peak_channel=%d)''' \
               % \
               (self.animal_name, self.region, self.genotype,
                self.species, self.sex,
                self.clust_idx, self.quality, self.peak_channel)

    def __add__(self, new):
        # def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
        #              fs=25000, start_time=0, end_time=12 * 60 * 60,
        #              mwft=None,
        #              sex=None, birthday=None, species=None,
        #              animal_name=None,
        #              genotype=None,
        #              expt_cond=None,
        #              on_time=None, off_time=None,
        #              rstart_time=None, rend_time=None,
        #              estart_time=None, eend_time=None,
        #              sp_amp=None,
        #              region_loc=None,
        #              wf_b=None, wf_e=None,
        #              key=None,
        #              qual_prob=None):

        self_dict = self.__dict__
        new_dict = new.__dict__
        # ik  clust_idx  jk  clust_idx
        # ik  spike_time  jk  spike_time
        # ik  quality  jk  quality
        # ik  waveform  jk  waveform
        # ik  waveforms  jk  waveforms
        # ik  peak_channel  jk  peak_channel
        # ik  fs  jk  fs
        # ik  start_time  jk  start_time
        # ik  end_time  jk  end_time
        # ik  waveform_tetrodes  jk  waveform_tetrodes
        # ik  on_times  jk  on_times
        # ik  off_times  jk  off_times
        # ik  rstart_time  jk  rstart_time
        # ik  rend_time  jk  rend_time
        # ik  estart_time  jk  estart_time
        # ik  eend_time  jk  eend_time
        # ik  spike_amplitude  jk  spike_amplitude
        # ik  wf_b  jk  wf_b
        # ik  wf_e  jk  wf_e
        # ik  key  jk  key
        # ik  qual_prob  jk  qual_prob
        # ik  cell_type  jk  cell_type
        # ik  mean_amplitude  jk  mean_amplitude
        for ik, iv, jk, jv in zip(self_dict.keys(), self_dict.values(),
                                  new_dict.keys(), new_dict.values()):
            if ik == jk:
                print("ik ", ik, " jk ", jk)
                if ik == "clust_idx":
                    new_clust_idx = np.array([10000 + iv])
                    print(np.int16(new_clust_idx)[0])
                    print("new_clust_idx ", new_clust_idx)
                elif ik == "spike_time":
                    new_spike_time = np.concatenate((iv, jv), axis=0)
                elif ik == "quality":
                    new_quality = -1
                elif ik == "waveform":
                    new_waveform = list((np.asarray(iv) + np.asarray(jv)) / 2)
                elif ik == "waveforms":
                    new_waveforms = list((np.asarray(iv) + np.asarray(jv)) / 2)
                elif ik == "peak_channel":
                    new_peak_channel = np.array([iv])
                elif ik == "fs":
                    new_fs = iv
                elif ik == "start_time":
                    new_start_time = iv
                elif ik == "end_time":
                    new_end_time = iv
                elif ik == "waveform_tetrodes":
                    new_waveform_tetrodes = (np.asarray(iv) + np.asarray(jv))/2
                elif ik == "on_times":
                    new_on_times = iv
                elif ik == "off_times":
                    new_off_times = iv
                elif ik == "rstart_time":
                    new_rstart_time = iv
                elif ik == "rend_time":
                    new_rend_time = iv
                elif ik == "estart_time":
                    new_estart_time = iv
                elif ik == "eend_time":
                    new_eend_time = iv
                elif ik == "spike_amplitude":
                    new_spike_amplitude = np.concatenate((iv, jv), axis=0)
                elif ik == "wf_b":
                    # may be add them lets keep it with iv
                    new_wf_b = iv
                elif ik == "wf_e":
                    # may be add them lets keep it with iv
                    new_wf_e = iv
                elif ik == "key":
                    new_key = None
                elif ik == "qual_prob":
                    new_qual_prob = np.array([0.00, 0.00, 0.00, 0.00])

        #        # def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
        #              fs=25000, start_time=0, end_time=12 * 60 * 60,
        #              mwft=None,
        #              sex=None, birthday=None, species=None,
        #              animal_name=None,
        #              genotype=None,
        #              expt_cond=None,
        #              on_time=None, off_time=None,
        #              rstart_time=None, rend_time=None,
        #              estart_time=None, eend_time=None,
        #              sp_amp=None,
        #              region_loc=None,
        #              wf_b=None, wf_e=None,
        #              key=None,
        #              qual_prob=None):
        return Neuron(new_clust_idx, new_spike_time, new_quality,
                      new_waveform, new_waveforms,
                      new_peak_channel, fs=new_fs,
                      start_time=new_start_time, end_time=new_end_time,
                      mwft=new_waveform_tetrodes, sex=self.sex,
                      birthday=self.birthday,
                      species=self.species, animal_name=self.animal_name,
                      genotype=self.genotype,
                      expt_cond=self.expt_cond,
                      on_time=new_on_times, off_time=new_off_times,
                      rstart_time=new_rstart_time, rend_time=new_rend_time,
                      estart_time=new_estart_time, eend_time=new_eend_time,
                      sp_amp=new_spike_amplitude, region_loc=self.region,
                      wf_b=new_wf_b, wf_e=new_wf_e, qual_prob=new_qual_prob,
                      key=new_key)

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

    @property
    def spike_time_sec(self):

        '''
        This will convert spike_time to seconds

        spike_time_sec(self)

        Parameters
        ----------

        Returns
        -------
        spike_time_sec : spike_time in seconds

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''

        logger.debug('Converting spike_time to seconds')
        # Sample time to time in seconds
        time_s = (self.spike_time / self.fs)

        return time_s

    @property
    def spike_time_sec_onoff(self):

        '''
        This will convert spike_time to seconds and
        remove spikes based on off times

        spike_time_sec(self)

        Parameters
        ----------

        Returns
        -------
        spike_time_sec : spike_time in seconds and
        remove spikes based on off times

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''

        logger.debug('Converting spike_time to seconds and apply on off times')

        # Sample time to time in seconds
        time_s_onoff = self.spike_time / self.fs
        good_times = None
        good_times = []
        if (hasattr(self, 'on_times') and hasattr(self, 'off_times')):
            for on_1, off_1 in zip(self.on_times, self.off_times):
                # print(on_1, " ", off_1)
                # a = np.delete(a, np.logical_and(a >= on_1, a <= off_1), 0)
                # a = np.delete(a, np.where((a >= on_1) & (a <= off_1))[0], 0)
                good_times.extend(np.argwhere((time_s_onoff >= on_1) &
                                              (time_s_onoff < off_1)))
                # print(good_times)
            # print("g0 ", good_times)
            # print("g1 ", np.asarray(good_times).flatten())
            # print("g2 ", np.int64(np.asarray(good_times).flatten()))

            # check good_times is not zero
            if len(good_times) < 1:
                raise ValueError('No spikes for this cell with onoff times')
            else:
                good_times = np.int64(np.asarray(good_times).flatten())
                # print("g0 ", good_times)
                good_times = good_times[(np.where(good_times >= 0)[0])]
                # print("g0 ", good_times)
                # time_s_onoff = time_s_onoff[np.int64(np.asarray(good_times)
                # .flatten())]
                time_s_onoff = time_s_onoff[np.int64(good_times)]
                # print("time_s_onoff ", time_s_onoff)
                # check whether length of on off time is greater
                if (len(time_s_onoff) > len(self.spike_time)):
                    raise RuntimeError('Error: more spikes after on/off times')
                else:
                    return time_s_onoff
        else:
            raise AttributeError('No attribute on_times or off_times')

    @property
    def age_rec(self):
        '''
        Calculate age at recording based on rstart_time

        age_sorted(self)

        Parameters
        ----------

        Returns
        -------
        age_rec : age at recording based on rstart_time

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''
        dt = datetime.strptime(self.rstart_time.replace("_", " "),
                               "%Y-%m-%d %H-%M-%S")
        return dt - self.birthday

    @property
    def peaklatency(self):
        '''
        Calculate peaklatency from mean waveform

        peaklatency(self)

        Parameters
        ----------

        Returns
        -------
        peaklatency : peaklatency of neuron waveform

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------

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
            # mean_amplitude = np.abs(temp[bottom])
            # mean_amplitude = temp[top] - temp[bottom]
        elif temp[maxvalueidx] > 0:
            bottom = np.argmax(temp)
            top = np.argmin(temp[bottom:]) + bottom
            peaklatency = ((top - bottom) * 1e3) / fs
            # Find mean amplitude
            # mean_amplitude = np.abs(temp[bottom])
            # mean_amplitude = temp[bottom] - temp[top]

        # Find cell type @Keith's paper
        # cell_type = 'RSU' if peaklatency >= 0.4 else 'FS'

        return peaklatency

    @property
    def mean_wf_b(self):

        '''
        Calculate mean waveform from wf_b

        mean_wf_b(self)

        Parameters
        ----------

        Returns
        -------
        mean_wf_b : calculate mean waveform from wf_b

        Raises
        ------
        ValueError if neuron has not attribute wf_b

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''

        logger.debug('Calculate mean waveform from wf_b')

        # Mean waveform
        if hasattr(self, 'wf_b'):
            mean_wf_b = np.mean(self.wf_b, axis=1)
            return mean_wf_b
        else:
            raise ValueError('No attribute wf_b')

    @property
    def mean_wf_e(self):

        '''
        Calculate mean waveform from wf_e

        mean_wf_e(self)

        Parameters
        ----------

        Returns
        -------
        mean_wf_e : calculate mean waveform from wf_e

        Raises
        ------
        ValueError if neuron has not attribute wf_e

        See Also
        --------

        Notes
        -----

        Examples
        --------

        '''

        logger.debug('Calculate mean waveform from wf_e')

        # Mean waveform
        if hasattr(self, 'wf_e'):
            mean_wf_e = np.mean(self.wf_e, axis=1)
            return mean_wf_e
        else:
            raise ValueError('No attribute wf_e')

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
        # logger.info('Not implemented')
        if np.array_equal(self.behavior, np.zeros((2, 6))):
            raise ValueError('behavior not added')
        else:
            return self.behavior

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

    def crosscorr(self, friend=None, dt=1e-3, tspan=1.0, nsegs=None,
                  savefig_loc=None):
        # if "friend" argument is not give, compute autocorrelation
        if friend is None:
            print('Computing autocorrelation for cell {:d}.'
                  .format(self.clust_idx))
            print('Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'
                  .format(int(tspan*1000), dt*1000))
            # select spike timestamps within on/off times for this cell
            # #ckbn stimes1 = \
            # #ckbn  self.spike_time[(self.spike_time
            # #ckbn > self.on_times[0]) &
            # #ckbn  (self.spike_time < self.off_times[-1])]
            stimes1 = self.spike_time_sec_onoff
            # remove spikes at the edges (where edges are tspan/2)
            t_start = time.time()
            subset = \
                [(stimes1 > stimes1[0]+tspan/2) &
                 (stimes1 < stimes1[-1] - tspan/2)]
            # line above returns an array of booleans. Convert to indices
            subset = np.where(subset)[1]
            # Take a subset of indices. We want "nsegs" elements,
            # evenly spaced.
            # "segindx" therefore contains nsegs spike indices, evenly spaced
            # between the first and last one in subset
            if nsegs is None:
                # #ckbn nsegs = np.int(np.ceil(np.max(self.spike_time/120)))
                nsegs = np.int(np.ceil(np.max(self.spike_time_sec/120)))
            print('\tUsing {:d} segments.'.format(nsegs))
            segindx = np.ceil(np.linspace(subset[0], subset[-1], nsegs))
            # The spikes pointed at by the indices in "segindx" are
            # our reference spikes for autocorrelation calculation

            # initialize timebins
            timebins = np.arange(0, tspan+dt, dt)
            # number of bins
            nbins = timebins.shape[0] - 1

            # initialize autocorrelation array
            ACorrs = np.zeros((nsegs, 2*nbins-1), float)

            # ref is the index of the reference spike
            print("len segindx ", len(segindx), flush=True)
            for i, ref in enumerate(segindx):
                ref = int(ref)
                # "t" is the timestamp of reference spike
                t = stimes1[ref]
                # find indices of spikes between t and t+tspan
                spikeindx = np.where((stimes1 > t) & (stimes1 <= t+tspan))[0]
                # get timestamps of those and subtract "t" to get "tau"
                # for each spike
                # "tau" is the lag between each spike
                # divide by "dt" step to get indices of bins in which
                # those spikes fall
                spikebins = np.ceil((stimes1[spikeindx] - t)/dt)
                if spikebins.any():
                    # if any spikes were found using this method,
                    # create a binary array of spike presence in each bin
                    bintrain = np.zeros(nbins, int)
                    bintrain[spikebins.astype(int)-1] = 1
                    # the auto-correlation is the convolution of that binary
                    # sequence with itself
                    # mode="full" ensures np.correlate uses convolution
                    # to compute correlation
                    ACorrs[i, :] = np.correlate(bintrain, bintrain,
                                                mode="full")

            # sum across all segments to get auto-correlation across dataset
            Y = np.sum(ACorrs, 0)
            # remove artifactual central peak
            Y[nbins-1] = 0
            print("Y ", Y, flush=True)

            # measure time elapsed for benchmarking
            elapsed = time.time() - t_start
            print('Elapsed time: {:.2f} seconds'.format(elapsed))

            # plotting ----------------------
            # -------------------------------
            # plt.ion()
            fig = plt.figure(facecolor='white')
            fig.suptitle('Auto-correlation, cell {:d}'.format(self.clust_idx))
            ax2 = fig.add_subplot(311, frame_on=False)

            ax2.bar(1000*np.arange(-tspan+dt, tspan, dt), Y,
                    width=0.5, color='k')
            xlim2 = 100  # in milliseconds
            tickstep2 = 20  # in milliseconds
            ax2.set_xlim(0, xlim2)
            ax2_ticks = [i for i in range(0, xlim2+1, tickstep2)]
            ax2_labels = [str(i) for i in ax2_ticks]
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels(ax2_labels)
            ax2.set_ylabel('Counts')

            ax3 = fig.add_subplot(312, frame_on=False)
            ax3.bar(1000*np.arange(-tspan+dt, tspan, dt), Y,
                    width=0.5, color='k')
            xlim3 = int(tspan*1000)  # milliseconds - set to tspan
            tickstep3 = int(xlim3/5)  # milliseconds
            ax3_ticks = [i for i in range(-xlim3, xlim3+1, tickstep3)]
            ax3_labels = [str(i) for i in ax3_ticks]
            ax3.set_xticks(ax3_ticks)
            ax3.set_xticklabels(ax3_labels)
            ax3.set_ylabel('Counts')
            ax3.set_xlabel('Lag (ms)')

            ax4 = fig.add_subplot(313, frame_on=False)
            ax4.plot(self.waveform)
            plt.tight_layout()

            if savefig_loc is None:
                # fig.show()
                plt.show()
            else:
                plt.savefig(savefig_loc)

            # -------------------------------

        # if friend is an instance of class "neuron", compute cross-correlation
        elif isinstance(friend, Neuron):
            print('Computing cross correlation between cells {:d} and {:d}.'
                  .format(self.clust_idx, friend.clust_idx))
            print('Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'
                  .format(int(tspan*1000), dt*1000))
            # select spike timestamps within on/off times for self cell
            # #ckbn stimes1 = \
            # #ckbn self.spike_time[(self.spike_time > self.on_times[0])
            # #ckbn & (self.spike_time < self.off_times[-1])]
            stimes1 = self.spike_time_sec_onoff
            # select spike timestamps within on/off times for self cell
            # #ckbn stimes2 = \
            # #ckbn friend.spike_time[(friend.spike_time > friend.on_times[0])
            # #ckbn & (friend.spike_time < friend.off_times[-1])]
            stimes2 = friend.spike_time_sec_onoff
            # start timer for benchmarking
            t_start = time.time()
            # remove spikes at the edges (where edges are tspan/2)
            subset1 =\
                [(stimes1 > stimes1[0]+tspan/2) &
                 (stimes1 < stimes1[-1] - tspan/2)]
            subset2 =\
                [(stimes2 > stimes2[0]+tspan/2) &
                 (stimes2 < stimes2[-1] - tspan/2)]
            # line above returns an array of booleans. Convert to indices
            subset1 = np.where(subset1)[1]
            subset2 = np.where(subset2)[1]
            # check to see if nsegs is user provided or default
            if nsegs is None:
                # #ckbn nsegs1 = np.int(np.ceil(np.max(self.spike_time/120)))
                # #ckbn nsegs2 = np.int(np.ceil(np.max(friend.spike_time/120)))
                nsegs1 = np.int(np.ceil(np.max(self.spike_time_sec/120)))
                nsegs2 = np.int(np.ceil(np.max(friend.spike_time_sec/120)))
                nsegs = max(nsegs1, nsegs2)
            print('\tUsing {:d} segments.'.format(nsegs))
            # Take a subset of indices. We want "nsegs" elements,
            # evenly spaced. "segindx" therefore contains nsegs
            # spike indices, evenly spaced between the first and
            # last one in subset
            segindx1 = np.ceil(np.linspace(subset1[0], subset1[-1], nsegs))
            # segindx2 = np.ceil(np.linspace(subset2[0], subset2[-1], nsegs))

            # The spikes pointed at by the indices in "segindx"
            # are our reference spikes for autocorrelation calculation

            # initialize timebins
            timebins = np.arange(0, tspan+dt,   dt)
            # number of bins
            nbins = timebins.shape[0] - 1

            # initialize autocorrelation array
            XCorrs = np.zeros((nsegs, 2*nbins-1), float)

            # ref is the index of the reference spike
            for i, ref in enumerate(segindx1):
                ref = int(ref)
                # "t" is the timestamp of reference spike
                t = stimes1[ref]
                # find indices of spikes between t and t+tspan, for cell SELF
                spikeindx1 = np.where((stimes1 > t) & (stimes1 <= t+tspan))[0]
                # same thing but for cell FRIEND
                spikeindx2 = np.where((stimes2 > t) & (stimes2 <= t+tspan))[0]
                # get timestamps of those and subtract
                # "t" to get "tau" for each spike
                # "tau" is the lag between each spike
                spikebins1 = np.ceil((stimes1[spikeindx1] - t)/dt)
                spikebins2 = np.ceil((stimes2[spikeindx2] - t)/dt)
                if spikebins1.any() & spikebins2.any():
                    # binary sequence for cell SELF
                    bintrain1 = np.zeros(nbins, int)
                    bintrain1[spikebins1.astype(int)-1] = 1
                    # binary sequence for cell FRIEND
                    bintrain2 = np.zeros(nbins, int)
                    bintrain2[spikebins2.astype(int)-1] = 1
                    XCorrs[i, :] = np.correlate(bintrain1, bintrain2,
                                                mode="full")

            Y = np.sum(XCorrs, 0)

            elapsed = time.time() - t_start
            print('Elapsed time: {:.2f} seconds'.format(elapsed))
            # plt.ion()
            figx = plt.figure(facecolor='white')
            figx.suptitle('Cross-correlation, cells {:d} and {:d}'
                          .format(self.clust_idx, friend.clust_idx))

            ax2 = figx.add_subplot(311, frame_on=False)

            ax2.bar(1000*np.arange(-tspan+dt, tspan, dt), Y,
                    width=0.5, color='k')
            xlim2 = int(200)  # in milliseconds
            tickstep2 = int(xlim2/5)  # in milliseconds
            ax2.set_xlim(-xlim2, xlim2)
            ax2_ticks = [i for i in range(-xlim2, xlim2+1, tickstep2)]
            ax2_labels = [str(i) for i in ax2_ticks]
            # ax2_labels = str(ax2_labels)
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels(ax2_labels)
            ax2.set_ylabel('Counts')

            ax3 = figx.add_subplot(312, frame_on=False)
            ax3.bar(1000*np.arange(-tspan+dt, tspan, dt), Y,
                    width=0.5, color='k')
            xlim3 = int(tspan*1000)  # milliseconds - set to tspan
            tickstep3 = int(xlim3/5)  # milliseconds
            ax3_ticks = [i for i in range(-xlim3, xlim3+1, tickstep3)]
            ax3_labels = [str(i) for i in ax3_ticks]
            ax3.set_xticks(ax3_ticks)
            ax3.set_xticklabels(ax3_labels)
            ax3.set_ylabel('Counts')
            ax3.set_xlabel('Lag (ms)')

            ax4 = figx.add_subplot(313, frame_on=False)
            ax4.plot(self.waveform)
            ax4.plot(friend.waveform)
            plt.tight_layout()

            if savefig_loc is None:
                # fig.show()
                plt.show()
            else:
                plt.savefig(savefig_loc)

        elif not isinstance(friend, Neuron):
            # error out
            msg = '''ERROR: Second argument to crosscorr should be an
                            instance of neuron class.'''
            raise TypeError(msg)

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
            ax.plot(self.waveform, color='#6a88f7')
            # ax.set_ylim(bottom=0)
            # ax.set_xlim(left=0)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.text(30, 0.7*ax.get_ylim()[1],
                    str("idx ") + str(self.clust_idx))
        sns.despine()

    def isi_hist(self, start=False, end=False, isi_thresh=0.1, nbins=101,
                 lplot=1, lonoff=1):
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
        nbins : Number of bins (default 101)
        lplot : To plot or not (default lplot=1, plot isi)
        lonoff : Apply on off times (default on, 1)

        Returns
        -------
        ISI : spike time difference (a[i+1] - a[i]) along axis
        edges
        hist_isi

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
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

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

        if lplot:
            plt.ion()
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
        return ISI, edges, hist_isi

    def plotFR(self, binsz=3600, start=False, end=False,
               lplot=1, lonoff=1):
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
        lplot : To plot or not (default lplot=1, plot firing rate)
        lonoff : Apply on off times (default on, 1)

        Returns
        -------
        hzcount : count per bins
        xbins

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

        # logger.debug('Plotting firing rate')
        # Sample time to time in seconds
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        # range
        idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
        time_s = time_s[idx_l]

        edges = np.arange(start, end + binsz, binsz)
        bins = np.histogram(time_s, edges)
        hzcount = bins[0]/binsz
        # hzcount[hzcount == 0] = 'NaN'
        xbins = bins[1]/binsz

        if lplot:
            plt.ion()
            with sns.axes_style("white"):
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                # print(hzcount)
                ax.plot(xbins[:-1], hzcount, color='#703be7')
                ax.set_xlabel('Time (hours)')
                ax.set_ylabel('Firing rate (Hz)')

        if lplot:
            sns.despine()
        # plt.show()
        return hzcount, xbins

    def isi_contamination_over_time(self, cont_thresh_list, binsz=300,
                                    lonoff=1):

        '''
        This function calculates isi contamination of a neuron over time
        from cont_thresh_list
        Unless otherwise specified cont_thresh_list is in seconds

        isi_contamination_over_time(self, cont_thresh_list=[0.003],
                                    binsz=300)

        Parameters
        ----------
        cont_thresh_list : threshold lists for calculating isi contamination
        binsz : binsize
        lonoff : Apply on off times (default on, 1)

        Returns
        -------
        isi_contamin : contamination over time

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].isi_contamination_over_time(cont_thresh_list=[0.003, 0.004])

        '''

        logger.debug("Calculating isi contamination over time")
        # tic = time.time()
        time_limit = np.inf

        # check cont_thresh_list is not empty
        if (len(cont_thresh_list) == 0):
            raise ValueError('cont_thresh_list list is empty')

        # Sample time to time in seconds
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

        start_times = np.arange(0, time_s[-1], binsz)  # finds the bin edges
        end_times = np.append(start_times[1:], start_times[-1]+binsz)
        all_values = []
        for idx in range(len(start_times)):
            # isi_cont = \
            #     self.isi_contamination(cont_thresh_list=cont_thresh_list,
            #                            start=start_times[idx],
            #                            end=end_times[idx])
            # range
            idx_l = np.where(np.logical_and(time_s >= start_times[idx],
                                            time_s <= end_times[idx]))[0]
            time_s_r = time_s[idx_l]

            # Calculate isi
            isi = np.diff(time_s_r)

            # Loop and calculate contamination at various cont_thesh
            isi_contamin = []
            for cont_thresh in cont_thresh_list:
                isi_contamin.append(100.0 * (np.sum(isi < cont_thresh) /
                                    np.sum(isi < time_limit)))

            all_values.append(isi_contamin)

        all_cont = np.array(all_values)

        cont_lines = []
        for idx in np.arange(len(cont_thresh_list)):
            cont_lines.append(all_cont[:, idx])

        cont_lines = np.array(cont_lines)
        # toc = time.time()
        # print("Time taken isicont {}".format(toc-tic))

        return cont_lines

    def isi_contamination(self, cont_thresh_list=None,
                          time_limit=np.inf,
                          start=False, end=False,
                          lonoff=1):

        '''
        This function calculates isi contamination of a neuron
        from cont_thresh_list
        unless otherwise specified cont_thresh_list, time_limit,
        start and end are in seconds

        isi_contamination(self, cont_thresh_list=[0.003],
                          time_limit=np.inf,
                          start=False, end=False)

        Parameters
        ----------
        cont_thresh_list : threshold lists for calculating isi contamination
        time_limit : count spikes upto, default np.inf. Try also 100 ms, 0.1
        start : Start time (default self.start_time)
        end : End time (default self.end_time)
        lonoff : Apply on off times (default on, 1)

        Returns
        -------
        isi_contamin : contamination percentage based on cont_thresh_list

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].isi_contamination(cont_thresh_list=[0.003, 0.004],
                                time_limit=0.1,
                                start=False, end=False)

        '''

        logger.debug('Calculating isi contamination')

        # check cont_thresh_list is not empty
        if (len(cont_thresh_list) == 0):
            raise ValueError('cont_thresh_list list is empty')

        # Sample time to time in seconds
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        # range
        idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
        time_s = time_s[idx_l]

        # Calculate isi
        isi = np.diff(time_s)

        # Loop and calculate contamination at various cont_thesh
        isi_contamin = []
        for cont_thresh in cont_thresh_list:
            isi_contamin.append(100.0 * (np.sum(isi < cont_thresh) /
                                         np.sum(isi < time_limit)))

        logger.debug('isi contaminations {}'.format(isi_contamin))

        return isi_contamin

    def presence_ratio(self, nbins=101, start=False, end=False,
                       lonoff=1):

        '''
        This will calculate ratio of time an unit is present in this block
        Unless otherwise specified  start and end are in seconds

        presence_ratio(self, nbins=101, start=False, end=False)

        Parameters
        ----------
        nbins : Number of bins (default 101)
        start : Start time (default self.start_time)
        end : End time (default self.end_time)
        lonoff : Apply on off times (default on, 1)

        Returns
        -------
        presence_ratio : ratio of time an unit is present in this block

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].presence_ratio(nbins=101, start=False, end=False)

        '''

        logger.debug('Calculating presence ratio')
        # Sample time to time in seconds
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        # range
        idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
        time_s = time_s[idx_l]

        # Calculation
        p_tmp, _ = np.histogram(time_s, np.linspace(start, end, nbins))
        prsc_ratio = (np.sum(p_tmp > 0) / (nbins - 1))
        logger.debug('Prescence ratio is %f', prsc_ratio)

        return prsc_ratio

    def remove_large_amplitude_spikes(self, threshold,
                                      lstd_deviation=True,
                                      start=False, end=False,
                                      lplot=True, lonoff=0):

        '''
        This function will remove large spike time from large amplitude spikes
        based on threshold

        remove_large_amplitude_spikes(self, threshold,
                                      lstd_deviation=True,
                                      start=False, end=False,
                                      lplot=True)

        Parameters
        ----------
        threshold : Threshold, based on lstd_deviation
        lstd_deviation : If True (default) threshold is
            in standard deviation 1.5 for example. If
            lstd_deviation is False then threshold is value
            above which amplitudes are to be remove.
        start : Start time (default self.start_time)
        end : End time (default self.end_time)
        lplot : Show plots (default True) allows selection
            else remove amplitudes above threshold without plotting
        lonoff : Apply on off times (default off, 0)

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
        n1[0].remove_large_amplitude_spikes(threshold,
                                            lstd_deviation=True,
                                            start=False, end=False,
                                            lplot=True)

        '''

        logger.info('Removing large amplitude spikes')
        # Sample time to time in seconds
        if lonoff:
            time_s = self.spike_time_sec_onoff
        else:
            time_s = self.spike_time_sec

        if start is False:
            start = self.start_time
        if end is False:
            end = self.end_time
        logger.debug('start and end is %s and %s', start, end)

        # range
        idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
        time_s = time_s[idx_l]

        # amp = amp[abs(amp - np.mean(amp)) < 1.5 * np.std(amp)]
        # Remove spikes based on threshold
        len_spks = len(self.spike_time)
        amps = self.spike_amplitude * 1.0
        if lstd_deviation:
            amps_m = abs(amps - np.mean(amps))
            idx_normalamps = np.where(amps_m < (threshold * np.std(amps)))[0]

        else:
            idx_normalamps = np.where(amps < threshold)[0]
        if lplot:
            with plt.style.context('seaborn-dark-palette'):
                fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=False,
                                       sharex=False, sharey=False,
                                       figsize=(8, 9),
                                       num=1, dpi=100, facecolor='w',
                                       edgecolor='w')
                # fig.tight_layout(pad=1.0)
                fig.tight_layout(pad=5.0)
                for i, row in enumerate(ax):
                    for j, col in enumerate(row):
                        if i == 0:
                            col.plot(amps, 'b.')
                            # col.set_xlim(left=-100, right=len(amps) + 100)
                            col.set_title('With large amplitudes')
                            col.set_xlabel('Time')
                            col.set_ylabel('Amplitude')
                        elif i == 1:
                            col.plot(amps[idx_normalamps], 'g.')
                            # col.set_xlim(left=-100,
                            #              right=len(amps[idx_normalamps]) +
                            #              100)
                            if lstd_deviation:
                                col.set_title('With large amplitudes removed '
                                              + 'above standard deviation ' +
                                              str(threshold))
                            else:
                                col.set_title('With large amplitudes removed '
                                              + ' above threshold ' +
                                              str(threshold))
                            col.set_xlabel('Time')
                            col.set_ylabel('Amplitude')
                        elif i == 2:
                            col.plot([1], [2])
                            plt.xticks([])
                            plt.yticks([])
                            col.spines['right'].set_visible(False)
                            col.spines['top'].set_visible(False)
                            col.spines['bottom'].set_visible(False)
                            col.spines['left'].set_visible(False)
                            axbox = plt.axes([0.128, 0.04, 0.17, 0.17])
                            radio = RadioButtons(axbox, ('Yes', 'No'),
                                                 active=(0, 0))

                            def remove_ampl(yes_no):
                                if yes_no == 'Yes':
                                    self.spike_time = \
                                        self.spike_time[idx_normalamps]
                                    self.spike_amplitude = \
                                        self.spike_amplitude[idx_normalamps]
                                    logger.info('Removed %d spikes',
                                                (len_spks -
                                                 len(idx_normalamps)))
                            radio.on_clicked(remove_ampl)

                            col.yaxis.set_label_coords(0.0, 0.15)
                            col.set_ylabel('Apply changes?')
                            col.set_xlabel("Press 'q' to exit")
                            col.xaxis.set_label_coords(0.1, -0.37)

                plt.show(block=True)
        else:
            self.spike_time = \
                self.spike_time[idx_normalamps]
            self.spike_amplitude = \
                self.spike_amplitude[idx_normalamps]
            # print(f'Total spikes {len_spks} spikes')
            # print(f'Normal spikes {len(idx_normalamps)} spikes')
            # print(f'Removed {len_spks - len(idx_normalamps)} spikes')
            logger.info('Removed %d spikes', (len_spks - len(idx_normalamps)))

    def signal_to_noise(self, file_name):

        '''
        This function will calculate singal to noise ratio from the output of
        spike interface rec_channels_std*.npy file

        signal_to_noise(self, file_name)

        Parameters
        ----------
        file_name : Output of spike interface rec_channels_std*.npy
                    file with path

        Returns
        -------
        signal_to_noise : ratio of time an unit is present in this block

        Raises
        ------
        FileNotFoundError if file_name not found

        See Also
        --------

        Notes
        -----

        Examples
        --------
        n1[0].signal_to_noise('/home/kbn/rec_channels_std0.npy')

        '''

        logger.info('Calculating signal to noise ratio')

        # check file exist
        if not (op.exists(file_name) and op.isfile(file_name)):
            raise FileNotFoundError("File {} not found".format(file_name))

        # Load rec_channels_std
        rec_channels_std = load_np(file_name, lpickle=True)

        sn_ratio = (np.max(np.abs(self.waveform)) /
                    rec_channels_std[self.peak_channel])

        logger.info('Signal to noise ratio is %f', sn_ratio)

        return sn_ratio

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

        # convert to int from string
        if isinstance(qual, str):
            qual = int(qual)

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
        tmp_qual_prob = np.zeros(4)
        tmp_qual_prob[qual - 1] = 100.00
        self.qual_prob = tmp_qual_prob
        self.quality = np.int8(qual)
        logger.info('Changing quality of unit {} is now {}'
                    .format(self.clust_idx, self.quality))

    def set_onofftimes(self):
        '''
        GUI based approach to setting on/off times for a neuron.
        Based on PLOT_SETONOFF_TIMES.PY. Function that allows a user to
        manually label times in which recordings should be considered
        "on" and "off" in a GUI based format. This program assumes that
        the user is loading a .npy file with multiple "neuron" class objects
        that should have fields for spike times, spike amplitudes, and
        cell quality. The user can either manually enter the file name
        and location or a GUI file selector will automatically pop up.

        NB: this is all very slow because it's built in matplotlib,
        but due to the necessity of matplotlib on everyone's machines,
        this should run without much trouble on almost any system
        within the lab.
        '''
        import seaborn as sns
        # KIRAN we need to test this function.
        plt.ion()
        global nidx, subp, subp1, tag  # , cid1, cid2
        nidx = 0  # This will index the selected neurons from tcells
        tag = np.NaN

        sp_t = []

        def onclick(event):
            # capture the x coordinate (time) of the user click input.
            # This will only matter if the user has already made a
            # keyboard press to indicate on or off time (1 or 0, respectively).
            global ix, iy, tag
            ix, iy = event.xdata, event.ydata
            print("ix ", ix, " iy ", iy)

            ylims = subp.get_ylim()
            ylims1 = subp1.get_ylim()

            if tag == 'on':
                subp.vlines(ix, ylims[0], ylims[1], colors='xkcd:seafoam')
                subp1.vlines(ix, ylims1[0], ylims1[1], colors='xkcd:seafoam')
                # plt.pause(0.01)
                sp_t.append([ix, 1])
                tag = np.NaN
                tag_event()

            elif tag == 'off':
                subp.vlines(ix, ylims[0], ylims[1], colors='xkcd:vermillion')
                subp1.vlines(ix, ylims1[0], ylims1[1],
                             colors='xkcd:vermillion')
                # plt.pause(0.01)
                sp_t.append([ix, 0])
                tag = np.NaN
                tag_event()

        def press(event):
            # Respond to the user's keyboard input. User can select right or
            # left keys to advance/retreat through the individual neurons that
            # meet the initial filtering criteria. User can also press 1 to
            # indicate ON time and 0 to indicate OFF time. Pressing 0 or 1 will
            # then make the code "listen" for a mouse input that is used to log
            # the on/off timestamp. Alternatively, the user can press "z" to
            # delete the most recent time input (mouse click).
            sys.stdout.flush()

            global nidx, subp, ky, tag  # , cid1, cid2
            ky = event.key

            if ky == '0':
                tag = 'off'
                tag_event()
                # set off time

            elif ky == '1':
                tag = 'on'
                tag_event()
            #     # set on time

            elif ky == 'z':
                tag = 'del'
                tag_event()

            elif ky == 'd':
                # This button will save the indicated on/off times
                # to the applicable neurons
                savefunc()

        def savefunc():
            # KIRAN - Does this work? Save func was a way to exit the program
            # and save the data. In this case, the user should be able to enter
            # a bunch of on off times and then press "D" to complete the
            # process and exit the GUI with the new on/off times added to the
            # base dataset
            return (self)

        def tag_event():
            # Change the sup title at the top of the window to reflect the
            # user's recent selection. This is mostly to show the user that the
            # code has registered their input, however, if the user has
            # selected "z" (delete most recent time stamp), this will clear the
            # last entry in the time stamp list and call the plotting code to
            # refresh the data w/ timestamp deleted.
            global tag

            if tag == 'off':
                top_title.update({'text': 'Click OFF time.'})
                plt.pause(0.01)
            elif tag == 'on':
                top_title.update({'text': 'Click ON time.'})
                plt.pause(0.01)
            elif tag == 'del':
                del sp_t[-1]
                subp.cla()
                subp1.cla()
                top_title.update({'text': 'Deleted last time selection.'})
                plotcell(self)
            elif np.isnan(tag):
                top_title.update({'text': 'Ready to continue.'})
                plt.pause(0.01)

        def plotcell(neuron):
            # Core plotting code to show two subplots. Top subplot is amplitude
            # versus time, and the bottom is firing rate versus time. If there
            # are on/off times, this will also process those and display them
            # accordingly.

            # Amplitudes subplot:
            meanamp = np.mean(neuron.spike_amplitude)
            stdevamp = np.std(neuron.spike_amplitude)

            subp.scatter(neuron.spike_time_sec/3600, neuron.spike_amplitude,
                         color=(0.5, 0.5, 0.5), marker='.', alpha=0.075)
            # set reasonable x and y lims for display.
            # y min is 3rd percentile and max is 5 std
            subp.set_ylim(np.percentile(neuron.spike_amplitude, 3),
                          meanamp+4*stdevamp)
            subp.set_xlim(np.floor(neuron.spike_time_sec[0]/3600),
                          neuron.spike_time_sec[-1]/3600)

            xlims = subp.get_xlim()
            ylims = subp.get_ylim()

            txtx = xlims[0]+0.1*(xlims[1] - xlims[0])
            txty = ylims[0]+0.5*(ylims[1] - ylims[0])
            subp.text(txtx, txty, f'cluster index: {neuron.clust_idx}')
            subp.set_xlabel('Time (hours)')
            subp.set_ylabel('Amplitude (uV)')
            sns.despine()
            # plt.draw()

            # Firing rate subplot:
            t0 = neuron.spike_time_sec[0] / 3600
            t1 = neuron.spike_time_sec[-1] / 3600
            step = 120
            edges = np.arange(t0, t1, step / 3600)
            fr = np.histogram(neuron.spike_time_sec/3600, edges)
            subp1.plot(fr[1][0:-1], fr[0]/step, color='xkcd:periwinkle',
                       linewidth=3)
            # Set the limits so they stop drifting when adding vertical lines.
            subp1.set_ylim(0, np.ceil((np.max(fr[0])/step)*1.05))
            subp1.set_xlabel('Time (hours)')
            subp1.set_ylabel('Firing Rate (Hz)')
            sns.despine()

            if np.size(sp_t) > 0:
                # Add on/off lines if they exist
                # take the list of on/off times and
                # convert to an array for searching.
                oots = np.stack(sp_t, axis=0)
                tempons = np.squeeze(np.where(oots[:, 1] == 1))
                tempoffs = np.squeeze(np.where(oots[:, 1] == 0))
                # pull ylims for the FR plot
                ylims1 = subp1.get_ylim()
                # add the lines to the amplitude plot
                subp.vlines(oots[tempons, 0], ylims[0], ylims[1],
                            colors='xkcd:seafoam')
                subp1.vlines(oots[tempons, 0], ylims1[0], ylims1[1],
                             colors='xkcd:seafoam')
                # and add the same x vals to the FR plot with the proper ylims
                subp.vlines(oots[tempoffs, 0], ylims[0], ylims[1],
                            colors='xkcd:vermillion')
                subp1.vlines(oots[tempoffs, 0], ylims1[0], ylims1[1],
                             colors='xkcd:vermillion')
            else:
                pass

            plt.draw()

        # Set up the figure and connect it to click and press utilities.
        fig = plt.figure(constrained_layout=True, figsize=(14, 7))
        top_title = fig.suptitle('Placeholder', fontsize=14)
        fig.canvas.mpl_connect('key_press_event', press)
        fig.canvas.mpl_connect('button_press_event', onclick)
        gs = fig.add_gridspec(2, 1)
        subp = fig.add_subplot(gs[0, 0])
        subp1 = fig.add_subplot(gs[1, 0], sharex=subp)

        # Call the initial plotting:
        plotcell(self)

    def set_onofftimes_from_list(self, ontimes, offtimes):
        '''
        This function allows to change on off time of neuron

        set_onofftimes_from_list(self, ontimes, offtimes)

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
        n[0].set_onofftimes_from_list(ontimes, offtimes)

        '''

        logger.info('Changing on off times from list')

        # numpy array to list
        if isinstance(ontimes, np.ndarray):
            ontimes = ontimes.tolist()
        if isinstance(offtimes, np.ndarray):
            offtimes = offtimes.tolist()

        # print("ontimes type ", type(ontimes))
        # print("offtimes type ", type(offtimes))

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

        # Check on off is ascending
        allonofftimes = None
        allonofftimes = []
        for on_tmp, off_tmp in zip(ontimes, offtimes):
            allonofftimes.extend([on_tmp, off_tmp])
            if sorted(allonofftimes) != allonofftimes:
                raise ValueError('Error: on {} off {} times not ascending'
                                 .format(ontimes, offtimes))

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

    def checkqual(self, binsz=3600, start=False, end=False, lsavepng=0,
                  png_outdir=None, fix_amp_ylim=0):
        # copied from musclebeachtools

        '''
        Change quality of a neuron
        This will produce a figure with ISI, Firing rate, waveform and
        radio button to change quality.

        checkqual(self, binsz=3600, start=False, end=False)

        Parameters
        ----------
        binsz : Bin size (default 3600)
        start : Start time (default self.start_time)
        end : End time (default self.end_time)
        lsavepng : Save checkqual results as png's
        png_outdir : Directory to save png files
                     if lsavepng=1 and png_outdir=None
                     png's will be saved in current working directory
        fix_amp_ylim : default 0, yaxis max in amplitude plot.
                       For example can be fix_amp_ylim=500 to see from 0 to 500
                       in amplitude plot.

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
        n1[0].checkqual(binsz=3600, start=False, end=False)

        '''

        logger.info('Plotting figures for checking quality')
        if lsavepng:
            if png_outdir is not None:
                if not os.path.exists(png_outdir):
                    raise \
                        NotADirectoryError('Folder {} not found'
                                           .format(png_outdir))

        # sharex=True, sharey=True,  figsize=(4, 4),
        with plt.style.context('seaborn-dark-palette'):
            # unsure what the constrained layout thing does
            fig = plt.figure(constrained_layout=True, figsize=(11, 8))
            gs = fig.add_gridspec(4, 3)

            waveform_ax = fig.add_subplot(gs[1:, 0])
            waveform_ax.set_title("Waveform")

            isi_ax = fig.add_subplot(gs[0, 1:])
            isi_ax.set_title("ISI Hist")

            amp_ax = fig.add_subplot(gs[-1, 1:])
            amp_ax.set_title("Amplitudes")

            fr_ax = fig.add_subplot(gs[1, 1:])
            fr_ax.set_title("Firing Rate")

            isi_time_ax = fig.add_subplot(gs[-2, 1:])
            isi_time_ax.set_title("Isi Contamination over time")

            qual_ax = fig.add_subplot(gs[0, 0])
            qual_ax.set_title("Set Quality")

            # raw_ax = fig.add_subplot(gs[1:, -1])
            # raw_ax.set_title("Raw Trace")

            # ISI HIST plot
            _, edges, hist_isi = self.isi_hist(lplot=0)

            isi_contamin = \
                self.isi_contamination(cont_thresh_list=[0.001,
                                                         0.002,
                                                         0.003,
                                                         0.005])
            r = np.arange(0, 100)
            colors = np.where(r < 5, "orangered", '#0b559f')
            isi_ax.bar(edges[1:]*1000-0.5, hist_isi[0],
                       color=colors)
            isi_ax.set_xlabel('ISI (ms)')
            isi_ax.set_ylabel('Number of intervals')
            isi_ax.set_xlim(left=0)
            isi_ax.set_ylim(bottom=0)
            isi_ax.set_xlim(left=0, right=101)
            isi_ax.axvline(1, linewidth=2, linestyle='--', color='r')
            isi_contamin_str = '\n'.join((
                r'$@1ms=%.2f$' % (isi_contamin[0], ),
                r'$@2ms=%.2f$' % (isi_contamin[1], ),
                r'$@3ms=%.2f$' % (isi_contamin[2], ),
                r'$@5ms=%.2f$' % (isi_contamin[3], )))
            props = dict(boxstyle='round', facecolor='wheat',
                         alpha=0.5)
            isi_ax.text(0.73, 0.95, isi_contamin_str,
                        transform=isi_ax.transAxes,
                        fontsize=8,
                        verticalalignment='top', bbox=props)

            # FR plot
            total_fr =\
                (len(self.spike_time) /
                    (self.end_time - self.start_time))
            logger.info('Total FR is %f', total_fr)
            prsc_ratio = self.presence_ratio()
            if self.end_time > 3600 * 4:
                hzcount, xbins = self.plotFR(binsz=300, lplot=0)
                fr_ax.plot(xbins[:-1], hzcount, color='#703be7')
                fr_ax.set_xlim(left=self.start_time)
                fr_ax.set_xticks([])
                fr_ax.set_xlabel('Time')
                fr_ax.set_ylabel('Firing rate (Hz)')
            else:
                hzcount, xbins = \
                    self.plotFR(binsz=self.end_time/60,
                                lplot=0)
                fr_ax.plot(xbins[:-1], hzcount, color='#703be7')
                fr_ax.set_xlim(left=self.start_time)
                fr_ax.set_xticks([])
                fr_ax.set_xlabel('Time')
                fr_ax.set_ylabel('Firing rate (Hz)')
            fr_stats_str = '\n'.join((
                r'$Nspikes=%d$' % (len(self.spike_time), ),
                r'$TotalFr=%.2f$' % (total_fr, ),
                r'$Pratio=%.2f$' % (prsc_ratio, )))
            props = dict(boxstyle='round', facecolor='wheat',
                         alpha=0.5)
            fr_ax.text(0.73, 0.95, fr_stats_str,
                       transform=fr_ax.transAxes,
                       fontsize=8,
                       verticalalignment='top', bbox=props)

            # ISI TIME plot
            contamination_lines =  \
                self.isi_contamination_over_time(cont_thresh_list=[0.001,
                                                                   0.002,
                                                                   0.003,
                                                                   0.005])

            isi_time_ax.plot(contamination_lines.T, alpha=0.7)
            isi_time_ax.set_ylim((0, 20))
            # isi_time_ax.set_xlim((self.start_time, self.end_time/300))
            isi_time_ax.set_xlim(left=self.start_time)
            isi_time_ax.set_xlabel('Time')
            isi_time_ax.set_ylabel('Perc. contamination')
            isi_time_ax.legend(['@1ms', '@2ms', '@3ms', '@5ms'],
                               loc="upper right", fontsize="small")
            # isi_time_ax.set_xticks(np.arange(self.start_time,
            #                                  self.end_time/300, 12))
            # isi_time_ax.set_xticklabels(np.arange(0,
            #                                       int(self.end_time/3600)))
            isi_time_ax.set_xticks([])

            # WAVEFORM plot
            if hasattr(self, 'waveform_tetrodes'):
                wf_sh = self.waveform_tetrodes.shape[0]
                waveform_ax.plot(np.linspace(0, (wf_sh * 1000.0) / self.fs,
                                             wf_sh),
                                 self.waveform_tetrodes,
                                 color='#6a88f7')
                waveform_ax.plot(np.linspace(0, (wf_sh * 1000.0) / self.fs,
                                             wf_sh),
                                 self.waveform,
                                 'green')
                waveform_ax.plot(np.linspace(0, (wf_sh * 1000.0) / self.fs,
                                             wf_sh), self.waveform, 'g.')

            else:
                wf_sh = self.waveform.shape[0]
                waveform_ax.plot(np.linspace(0, (wf_sh * 1000.0) / self.fs,
                                             wf_sh),
                                 self.waveform,
                                 color='#6a88f7')
                waveform_ax.plot(np.linspace(0, (wf_sh * 1000.0) / self.fs,
                                             wf_sh),
                                 self.waveform,
                                 'green')

            waveform_ax.set_xlabel('Time (ms)')
            waveform_ax.set_ylabel('Amplitude', labelpad=-3)
            waveform_ax.set_xlim(left=0,
                                 right=((wf_sh * 1000.0) / self.fs))
            wf_stats_str = '\n'.join((
                r'$Cluster Id=%d$' % (self.clust_idx, ),
                r'$Cell type=%s$' % (self.cell_type, ),
                r'$Channel=%d$' % (self.peak_channel, )))
            # print("wf_stats_str ", wf_stats_str)
            props = dict(boxstyle='round', facecolor='wheat',
                         alpha=0.5)
            waveform_ax.text(0.53, 0.10, wf_stats_str,
                             transform=waveform_ax.transAxes,
                             fontsize=8,
                             verticalalignment='top', bbox=props)

            # RAW TRACE plot

            # AMP plot
            if hasattr(self, 'spike_amplitude'):
                # calculate zscore
                th_zs = 2
                z = np.abs(sc.stats.zscore(self.spike_amplitude))
                # amp_ax.plot((self.spike_time / self.fs),
                #             self.spike_amplitude, 'bo',
                #             markersize=1.9, alpha=0.2)
                amp_ax.plot((self.spike_time[np.where(z <= th_zs)] / self.fs),
                            self.spike_amplitude[np.where(z <= th_zs)],
                            'bo',
                            markersize=1.0, alpha=0.1)
                amp_ax.plot((self.spike_time[np.where(z > th_zs)] / self.fs),
                            self.spike_amplitude[np.where(z > th_zs)],
                            'ro',
                            markersize=1.0, alpha=0.1)
                if (hasattr(self, 'on_times') and hasattr(self, 'off_times')):
                    for on_1, off_1 in zip(self.on_times, self.off_times):
                        # print(on_1, " ", off_1)
                        amp_ax.axvspan(on_1, off_1, ymin=0, ymax=0.1,
                                       facecolor='turquoise', alpha=0.7,
                                       zorder=10)

                amp_ax.set_xlabel('Time (s)')
                amp_ax.set_ylabel('Amplitudes', labelpad=-3)
                amp_ax.set_xlim(left=(self.start_time),
                                right=(self.end_time))
                if fix_amp_ylim:
                    amp_ax.set_ylim(bottom=0)
                else:
                    amp_ax.set_ylim(bottom=0,
                                    top=(min((np.mean(self.spike_amplitude) +
                                             (6*np.std(self.spike_amplitude))),
                                             np.max(self.spike_amplitude))))
                amp_stats_str = '\n'.join((
                    r'$Min: %d, Max: %d$' % (np.min(self.spike_amplitude),
                                             np.max(self.spike_amplitude), ),
                    r'$Mean:%d, Med:%d, Std:%d$'
                    % (np.mean(self.spike_amplitude),
                       np.median(self.spike_amplitude),
                       np.std(self.spike_amplitude), )))
                props = dict(boxstyle='round', facecolor='wheat',
                             alpha=0.5)
                amp_ax.text(0.73, 0.27, amp_stats_str,
                            transform=amp_ax.transAxes,
                            fontsize=8,
                            verticalalignment='top', bbox=props)

            else:
                amp_ax.plot([1], [2])
                plt.xticks([])
                plt.yticks([])
                amp_ax.spines['right'].set_visible(False)
                amp_ax.spines['top'].set_visible(False)
                amp_ax.spines['bottom'].set_visible(False)
                amp_ax.spines['left'].set_visible(False)
                amp_ax.axis('off')
                logger.info('No attribute .spike_amplitude')

            # SET QUAL plot
            qual_ax.plot([1], [2])
            plt.xticks([])
            plt.yticks([])
            qual_ax.spines['right'].set_visible(False)
            qual_ax.spines['top'].set_visible(False)
            qual_ax.spines['bottom'].set_visible(False)
            qual_ax.spines['left'].set_visible(False)
            axbox = plt.axes(qual_ax)
            radio = RadioButtons(axbox, ('1', '2', '3', '4'),
                                 active=(0, 0, 0, 0))
            if self.quality in list([1, 2, 3, 4]):
                radio.set_active((self.quality - 1))
                logger.info('Quality is now %d',
                            self.quality)
            else:
                logger.info('Quality not set')
            radio.on_clicked(self.set_qual)
            if hasattr(self, 'qual_prob'):
                if self.quality == 1:
                    qual_ax.text(0.5, 0.8,
                                 '{:.2f}'.format(self.qual_prob[0]) + '%',
                                 fontsize=10,
                                 color='#006374',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                else:
                    qual_ax.text(0.5, 0.8,
                                 '{:.2f}'.format(self.qual_prob[0]) + '%',
                                 fontsize=10,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                if self.quality == 2:
                    qual_ax.text(0.5, 0.6,
                                 '{:.2f}'.format(self.qual_prob[1]) + '%',
                                 fontsize=10,
                                 color='#006374',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                else:
                    qual_ax.text(0.5, 0.6,
                                 '{:.2f}'.format(self.qual_prob[1]) + '%',
                                 fontsize=10,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                if self.quality == 3:
                    qual_ax.text(0.5, 0.4,
                                 '{:.2f}'.format(self.qual_prob[2]) + '%',
                                 fontsize=10,
                                 color='#006374',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                else:
                    qual_ax.text(0.5, 0.4,
                                 '{:.2f}'.format(self.qual_prob[2]) + '%',
                                 fontsize=10,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                if self.quality == 4:
                    qual_ax.text(0.5, 0.2,
                                 '{:.2f}'.format(self.qual_prob[3]) + '%',
                                 fontsize=10,
                                 color='#006374',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
                else:
                    qual_ax.text(0.5, 0.2,
                                 '{:.2f}'.format(self.qual_prob[3]) + '%',
                                 fontsize=10,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=qual_ax.transAxes)
            else:
                print("Warning: No attribute qual_prob, update neuron file")
                print("\t check mbt.mbt_spkinterface_out")

            qual_ax.set_ylabel('Select quality')
            qual_ax.set_xlabel("Press 'q' to exit")
            qual_ax.xaxis.set_label_coords(0.1, -0.1)

            if lsavepng:
                if png_outdir is None:
                    png_outdir = os.getcwd()
                try:
                    png_filename = \
                        op.join(png_outdir,
                                str('checkqual_clust_idx_') +
                                str(self.clust_idx) +
                                str('.png'))
                    plt.savefig(png_filename)
                except Exception as e:
                    print("Error ", e)
                    raise RuntimeError('Error saving {}'.format(png_filename))
            else:
                plt.show(block=True)


def autoqual(neuron_list, model_file,
             ltrain=0):

    '''
    Find automatic quality of a neuron

    quality_predictions, cluster_idx = \
        autoqual(neuron_list, model_file
                 ltrain=0)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    model_file : model file with path
        '/media/HlabShare/models/xgboost_autoqual'
    ltrain : 0 (default 0) predict quality, 1 update model only
        update model if accuracy is above 90%. Please remember
        original model was created using a large dataset using
        gridsearch not using this function.

    Returns
    -------
    quality_predictions : quality predicted from model
    cluster_idx : cluster ids corresponding to quality_predictions

    Raises
    ------
    ValueError if neuron list is empty
    FileNotFoundError if model_file does not exists


    See Also
    --------
    set_qual : neuron[0].set_qual(qual)

    Notes
    -----

    Examples
    --------
    quality_predictions, cluster_idx = \
    autoqual(neuron_list, model_file)

    '''

    logger.info('Plotting figures for checking quality')
    lmake_xgbdata = 0
    ldebug = 0

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # check file exist
    if not (op.exists(model_file) and op.isfile(model_file)):
        raise FileNotFoundError("File {} not found".format(model_file))

    # Initialize the features and qual
    # 8 isicont + + 1 peak latency + 1 mean amplitude  +
    # 1 Total Fr + 1 presence ratio +
    # 1  Energy + 1 peaks
    # 75 wf
    #                  1  2  3  4  5  6   7   8   9   10  11  12  13  14
    #                     15 : 15
    # isi contamination
    #                  1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90 and
    #                  100 ms
    #                     16 : 50
    # contamination_lines 1,       2,       3,       4,   and 5
    #                     ms mean std median min max kur skw
    # Peak latency        51 : 52
    # Wf amplitude        53 : 53
    # Total FR            54 : 54
    # Presence ratio      55 : 64
    #                     101, 201, 301, 401, 501, 601, 701, 801, 901 and 1001
    # amplitude bin stats 65:71  mean std median min max kur skw
    # amplitude stats     72:78 mean std median min max kur skw
    # fr stats            79:85 mean std median min max kur skw
    # wf stats            86:92 mean std median min max kur skw
    # wf E                93 : 93
    # cell type           94 : 94
    # mean amp            95 : 95
    # nWF                 96 : 169
    # WF                  170 : 244
    nfet = (15 +
            35 +
            1 +
            1 +
            1 +
            10 +
            #
            1 + 1 + 1 + 1 + 1 + 1 + 1 +
            1 + 1 + 1 + 1 + 1 + 1 + 1 +
            1 + 1 + 1 + 1 + 1 + 1 + 1 +
            1 + 1 + 1 + 1 + 1 + 1 + 1 +
            1 +
            1 +
            1 +
            75 +
            75)
    neuron_features = np.zeros((len(neuron_list), nfet))
    neuron_qual = np.zeros((len(neuron_list)), dtype='int8')
    neuron_indices = np.zeros((len(neuron_list)), dtype='int16')

    print("sh neuron_features ", neuron_features.shape)
    print("sh neuron_qual ", neuron_qual.shape)

    for idx, i in enumerate(neuron_list):
        # assign quality
        if ltrain:
            neuron_qual[idx] = i.quality - 1
        if lmake_xgbdata:
            neuron_qual[idx] = i.quality - 1

        # print("i.quality ", i.quality)
        # print("i.quality ", type(i.quality))
        neuron_indices[idx] = i.clust_idx
        fet_idx = 0

        # ISI contamination
        tmp_fet = None
        tmp_isi = i.isi_contamination(cont_thresh_list=[0.001, 0.002, 0.003,
                                                        0.004, 0.005, 0.010,
                                                        0.020, 0.030, 0.040,
                                                        0.050, 0.060, 0.070,
                                                        0.080, 0.090, 0.100])
        # print("tmp_isi ", tmp_isi)
        # print("sh tmp_isi ", len(tmp_isi))
        tmp_fet = np.asarray(tmp_isi)
        # print("sh tmp_fet ", tmp_fet.shape)
        # print("fet_idx1 ", fet_idx)
        neuron_features[idx, fet_idx:tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx2 ", fet_idx)
        # print("idx isi cont ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("tmp_isi ", tmp_isi)
        tmp_isi = None

        # ISI contamination over time
        # tic = time.time()
        contamination_lines = \
            i.isi_contamination_over_time(cont_thresh_list=[0.001,
                                                            0.002,
                                                            0.003,
                                                            0.004,
                                                            0.005],
                                          binsz=300)

        # toc = time.time()
        # print("Time taken auto {}".format(toc-tic))
        for contamin_idx in range(5):
            tmp_fet = None
            if ldebug:
                print("contamination_lines[", contamin_idx, "] ",
                      contamination_lines[contamin_idx])
            tmp_fet = None
            # print("contamination_lines[",contamin_idx,"] ",
            # contamination_lines[contamin_idx])
            ncontl = np.asarray(contamination_lines[contamin_idx])
            # ncontl = ncontl/np.linalg.norm(ncontl)
            ncontl = ((ncontl - np.min(ncontl)) /
                      (np.max(ncontl) - np.min(ncontl)))
            tmp_fet = np.array([np.mean(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([np.std(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([np.median(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([np.min(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([np.max(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([sc.stats.kurtosis(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            tmp_fet = None
            tmp_fet = np.array([sc.stats.skew(ncontl)])
            neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
            # print("tmp_fet ", tmp_fet)
            fet_idx = fet_idx + tmp_fet.shape[0]
            # print("idx amp ", idx)
            # print("fet_idx ", fet_idx)
            # print(neuron_features[idx, :])
            ncontl = None
        if ldebug:
            print("tmp_isi ", tmp_isi)

        # print("idx isi time ", idx)
        # print("fet_idx ", fet_idx)
        contamination_lines = None
        # print(neuron_features[idx, :])

        # Peak latency
        tmp_fet = None
        tmp_fet = np.array([i.peaklatency])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        if ldebug:
            print("peaklatency ", tmp_fet)

        # Wf amplitude
        tmp_fet = None
        tmp_fet = np.array([i.mean_amplitude])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx wf amp ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("wf amplitude ", tmp_fet)

        # Total Fr
        total_fr =\
            (len(i.spike_time) /
                (i.end_time - i.start_time))
        tmp_fet = None
        tmp_fet = np.array([total_fr])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        if ldebug:
            print("total fr ", tmp_fet)

        # Presence ratio
        # def presence_ratio(self, nbins=101, start=False, end=False,
        #                    lonoff=1):
        # 1001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=1001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 1001 ", tmp_fet)
        # 2001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=2001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 2001 ", tmp_fet)
        # 3001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=3001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 3001 ", tmp_fet)
        # 4001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=4001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 4001 ", tmp_fet)
        # 5001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=5001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 5001 ", tmp_fet)
        # 6001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=6001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 6001 ", tmp_fet)
        # 7001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=7001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 7001 ", tmp_fet)
        # 8001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=8001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 8001 ", tmp_fet)
        # 9001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=9001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 9001 ", tmp_fet)
        # 10001
        tmp_fet = None
        tmp_fet = np.array([i.presence_ratio(nbins=10001)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        # print("idx fr ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("presence ratio 10001 ", tmp_fet)

        # Amplitude mean, std, median, min, max, kurtosis, skew
        tmp_fet = None
        time_s = i.spike_time_sec_onoff
        time_s_amp_idx = np.where(np.logical_and(time_s >= i.start_time,
                                                 time_s <= i.end_time))[0]
        time_s = time_s[time_s_amp_idx]
        namp_s = i.spike_amplitude[time_s_amp_idx]
        start_times = np.arange(0, time_s[-1], 300)
        end_times = np.append(start_times[1:], start_times[-1]+300)
        namp_s_bin_values = []
        for tmp_starttime in range(len(start_times)):
            time_s_amp_idx_bin_tmp =\
                np.where(np.logical_and(time_s >= start_times[tmp_starttime],
                                        time_s <= end_times[tmp_starttime]))[0]
            namp_s_bin_values.append(np.std(namp_s[time_s_amp_idx_bin_tmp]))

        if ldebug:
            print("namp_s_bin_values ", namp_s_bin_values)
        namp_s_bin = np.asarray(namp_s_bin_values)
        # namp_s_bin = namp_s_bin/np.linalg.norm(namp_s_bin)
        namp_s_bin = ((namp_s_bin - np.min(namp_s_bin)) /
                      (np.max(namp_s_bin) - np.min(namp_s_bin)))
        if ldebug:
            print("namp_s_bin ", namp_s_bin)
        tmp_fet = np.array([np.mean(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.std(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.median(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        if ldebug:
            print("namp_s_bin ", np.mean(namp_s_bin), np.median(namp_s_bin))
        tmp_fet = None
        tmp_fet = np.array([np.min(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.max(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.kurtosis(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.skew(namp_s_bin)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("idx amp ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        namp_s_bin = None
        namp_s_bin_values = None
        time_s_amp_idx_bin_tmp = None
        time_s_amp_idx = None
        tmp_starttime = None
        time_s = None
        namp_s = None

        namp = np.asarray(i.spike_amplitude)
        if ldebug:
            print("namp ", namp)
        # namp = namp/np.linalg.norm(namp)
        namp = (namp - np.min(namp)) / (np.max(namp) - np.min(namp))
        if ldebug:
            print("namp ", namp)
        tmp_fet = np.array([np.mean(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.std(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.median(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.min(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.max(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.kurtosis(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.skew(namp)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("idx amp ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("spike amplitude ", np.mean(namp), " ", np.median(namp))
        namp = None

        # Firing rate mean, std, median, min, max, kurtosis, skew
        hzcount, xbins = \
            i.plotFR(binsz=300, lplot=0)
        tmp_fet = None
        # print("hzcount ", hzcount)
        # print("xbins ", xbins)
        nfrate = np.asarray(hzcount)
        # nfrate = nfrate/np.linalg.norm(nfrate)
        nfrate = (nfrate - np.min(nfrate)) / (np.max(nfrate) - np.min(nfrate))
        tmp_fet = np.array([np.mean(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.std(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.median(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.min(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.max(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.kurtosis(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.skew(nfrate)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("idx amp ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("spike fr ", np.mean(nfrate), " ", np.median(nfrate))
        nfrate = None

        # Normalize WF
        tmp_fet_wf = None
        tmp_fet_wf = np.asarray(i.waveform)
        # tmp_fet_wf = tmp_fet_wf / np.linalg.norm(tmp_fet_wf)
        tmp_fet_wf = ((i.waveform - np.min(i.waveform)) /
                      (np.max(i.waveform) - np.min(i.waveform)))
        # print("tmp_fet_wf ", tmp_fet_wf)
        # print("sh tmp_fet_wf ", tmp_fet_wf.shape)
        tmp_fet = None
        tmp_fet = np.array([np.mean(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.std(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.median(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.min(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([np.max(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.kurtosis(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        tmp_fet = None
        tmp_fet = np.array([sc.stats.skew(tmp_fet_wf)])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("idx amp ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("spike wf ", np.mean(tmp_fet_wf), " ", np.median(tmp_fet_wf))

        # Calculate energy
        tmp_fet = None
        tmp_fet = np.array([np.sum(tmp_fet_wf**2, axis=0)])
        # print("tmp_fet ", tmp_fet)
        # print("sh tmp_fet ", tmp_fet.shape)
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        # print("fet_idx3 ", fet_idx)
        if ldebug:
            print("energy ", tmp_fet)

        # WF type
        tmp_fet = None
        if i.cell_type == 'RSU':
            tmp_fet = np.array([1])
        elif i.cell_type == 'FS':
            tmp_fet = np.array([2])
        else:
            tmp_fet = np.array([-1])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # cell_type = None
        # print("idx wf type ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("cell type ", tmp_fet)

        # WF amplitude
        tmp_fet = None
        tmp_fet = np.array([i.mean_amplitude])
        neuron_features[idx, fet_idx:fet_idx+tmp_fet.shape[0]] = tmp_fet
        # print("tmp_fet ", tmp_fet)
        fet_idx = fet_idx + tmp_fet.shape[0]
        # mean_amplitude = None
        # print("idx wf type ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        if ldebug:
            print("wf mean amplitude ", tmp_fet)

        # wf make sure if they are not 75 we have to make them 75 todo
        tmp_fet = None
        # tmp_fet = tmp_fet_wf * 1.0
        tmp_fet = ((i.waveform - np.min(i.waveform)) /
                   (np.max(i.waveform) - np.min(i.waveform)))
        neuron_features[idx, fet_idx:fet_idx + tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        if ldebug:
            print("fet_idx ", fet_idx)
            print("norm wf ", tmp_fet)
        # sys.exit()
        tmp_fet_wf = None
        # print("fet_idx ", fet_idx)
        # print("idx wf ", idx)
        # print("fet_idx ", fet_idx)
        # print(neuron_features[idx, :])
        tmp_fet = None
        tmp_fet = np.asarray(i.waveform)
        neuron_features[idx, fet_idx:fet_idx + tmp_fet.shape[0]] = tmp_fet
        fet_idx = fet_idx + tmp_fet.shape[0]
        if ldebug:
            print("fet_idx ", fet_idx)
        tmp_fet = None
    # return neuron_qual, neuron_features

    if ltrain:
        (neuron_features_train, neuron_features_test,
            neuron_qual_train, neuron_qual_test,
            neuron_indices_train, neuron_indices_test) = \
            train_test_split(neuron_features, neuron_qual,
                             neuron_indices,
                             test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(neuron_features_train, label=neuron_qual_train)
        dtest = xgb.DMatrix(neuron_features_test, label=neuron_qual_test)
    else:
        (neuron_features_test,
            neuron_qual_test,
            neuron_indices_test) = \
            (neuron_features, neuron_qual,
             neuron_indices)

        if lmake_xgbdata:
            return neuron_qual_test, neuron_features_test
        # print("neuron_qual ", neuron_qual)
        # print("neuron_qual_test ", neuron_qual_test)
        dtest = xgb.DMatrix(neuron_features_test, label=neuron_qual_test)

    if ltrain:
        params = {
                  'eta': 0.3,
                  'max_depth': 16,
                  'min_child_weight': 2,
                  'gamma': 0,
                  'objective': 'multi:softmax',
                  'num_class': 4
                 }
        # print("params ", params)
        mxgb_model = xgb.XGBClassifier(objective='multi:softmax')
        mxgb_model = xgb.train(params, dtrain, num_boost_round=1000,
                               xgb_model=model_file)
        # early_stopping_rounds=50,
        # eval_metric=["auc", "error", "logloss"]
        # print("dir mxgb_model ", dir(mxgb_model))
        preds = mxgb_model.predict(dtest)
        accuracy = accuracy_score(neuron_qual_test, preds)
        logger.info('Accuracy of predictions is {}'
                    .format(accuracy))
        if accuracy > 0.90:
            mxgb_model.save_model(model_file)
        else:
            # print('less accurate')
            logger.info('Not saving model as accuracy is lower than 90%')

        preds = preds + 1
        # return accuracy, neuron_features, neuron_qual
        return preds, neuron_indices_test
    else:
        mxgb_model = xgb.Booster()
        mxgb_model.load_model(model_file)
        preds_prob = mxgb_model.predict(dtest)
        preds = np.argmax(preds_prob, axis=1)
        # print("neuron qual ", neuron_qual + 1)
        # print("preds ", preds + 1)
        # accuracy = accuracy_score(neuron_qual_test, preds)
        # logger.info('Accuracy of predictions is {}'
        #             .format(accuracy))
        # preds = preds + 1

        # assign quality
        for idx, i in enumerate(neuron_list):
            i.set_qual(preds[idx] + 1)
            if hasattr(i, 'qual_prob'):
                # i.qual_prob = np.round(preds_prob[idx, preds], 4)
                # print("sh preds_prob", preds_prob.shape, " ",
                #       preds.shape)
                # print("preds[idx] + 1 ", preds[idx] + 1)
                # print("preds_prob[idx, preds] ", preds_prob[idx, preds[idx]])
                # i.qual_prob = preds_prob[preds, idx]
                i.qual_prob = np.round(preds_prob[idx] * 100, 2)

        return preds, neuron_indices_test


def check_sametetrode_neurons(channel1, channel2,
                              ch_grp_size=4,
                              lverbose=1):

    '''
    check_sametetrode_neurons(channel1, channel2,
                              ch_grp_size=4,
                              lverbose=1)

    channel1: first channel
    channel2: second channel
    ch_grp_size : default (4) for tetrodes
    lverbose: default(1) to print tetrodes check

    return True or False


    lsamechannel = \
        check_sametetrode_neurons(neurons[0].peak_channel,
                                  neurons[1].peak_channel,
                                  ch_grp_size=4,
                                  lverbose=1)
    '''

    ch_grp_size = 4
    tet = np.arange(channel1 - (channel1 % ch_grp_size),
                    channel1 + (ch_grp_size -
                    (channel1 % ch_grp_size)))

    tet2 = np.arange(channel2 - (channel2 % ch_grp_size),
                     channel2 + (ch_grp_size -
                     (channel2 % ch_grp_size)))
    if lverbose:
        print(channel1,  " ", channel2,  "", tet, "", tet2, "",
              np.array_equal(tet, tet2))
    return np.array_equal(tet, tet2)


# def n_update_behavior(neuron_list, behavior=None):
#
#     '''
#     Update behavior in neuron_list
#
#     n_update_behavior(neuron_list, behavior=behavior)
#
#     Parameters
#     ----------
#     neuron_list : List of neurons
#     behavior : behavior numpy array
#                (behavior.ndim == 2 or behavior.shape[0] ==2)
#                behavior must be numpy array with ndim==2
#                otherwise raise ValueError
#                Array behavior's first dimension is time and
#                second is sleep states
#
#     Returns
#     -------
#     neuron_list : neuron lists with behavior
#
#     Raises
#     ------
#     ValueError if neuron list is empty
#     ValueError if behavior array is empty
#     ValueError if behavior array is not numpy.ndarray
#     ValueError if behavior array shape[0] not 2
#
#     See Also
#     --------
#
#     Notes
#     -----
#     Please remember to save neuron_list as file for future use
#
#     Examples
#     --------
#     n_update_behavior(neuron_list, behavior=behavior)
#
#     '''
#
#     logger.info('Updating neuron list with behavior')
#     # check neuron_list is not empty
#     if (len(neuron_list) == 0):
#         raise ValueError('Neuron list is empty')
#
#     if behavior is None:
#         raise ValueError('behavior is empty')
#
#     if not type(behavior) is np.ndarray:
#         raise ValueError('behavior is not numpy array')
#
#     if not behavior.shape[0] == 2:
#         raise ValueError('behavior.shape[0] is not 2')
#
#     # Update neurons with behavior
#     for ind, neuron in enumerate(neuron_list):
#         neuron.behavior = neuron.update_behavior(behavior)
#
#     return neuron_list


def n_filt_quality(neuron_list, maxqual=None):

    '''
    Filter neuron_list using maxqual list

    n_filt_quality(neuron_list, maxqual=[1, 2, 3])

    Parameters
    ----------
    neuron_list : List of neurons
    maxqual : default [1, 2, 3, 4], filter by quality,
              neuron.quality in maxqual

    Returns
    -------
    neuron_list : filtered by quality in maxqual, numpy array

    Raises
    ------
    ValueError if neuron list is empty

    See Also
    --------

    Notes
    -----

    Examples
    --------
    n_filt_quality(neuron_list, maxqual=[1, 3])

    '''

    logger.info('Filter neuron_list using maxqual list')
    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')
    if type(neuron_list) == list:
        neuron_list = np.asarray(neuron_list)

    if maxqual is None:
        maxqual = [1, 2, 3, 4]

    # Filter neurons by quality
    filt_list = []
    for ind, neuron in enumerate(neuron_list):
        if int(neuron.quality) in maxqual:
            filt_list.append(ind)

    return neuron_list[filt_list]


def n_plot_neuron_wfs(neuron_list, maxqual=None,
                      pltname=None,
                      saveloc=None):

    '''
    Plotting all waverforms from neuron_list

    n_plot_neuron_wfs(neuron_list, maxqual=[1, 2, 3, 4],
                      pltname=None, saveloc=None)

    Parameters
    ----------
    neuron_list : List of neurons
    maxqual : default [1, 2, 3, 4], filter by quality,
              neuron.quality in maxqual
    pltname : plot name, if None it will be neuron_waveforms.png
    saveloc : if None, show plot, else savepath given
              save figure

    Returns
    -------

    Raises
    ------
    ValueError if neuron list is empty
    FileExistsError if saveloc not found

    See Also
    --------

    Notes
    -----

    Examples
    --------
    n_plot_neuron_wfs(neuron_list, maxqual=[1, 2, 3, 4],
                      pltname=None, saveloc=None)

    '''

    logger.info('Plotting all waveforms from neuron_list')
    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    if maxqual is None:
        maxqual = [1, 2, 3, 4]

    # check saveloc
    if saveloc is not None:
        if not op.exists(saveloc):
            raise FileExistsError("Folder {} not found"
                                  .format(saveloc))

    # Plot all neurons by quality
    len_neurons = 0
    for ind, neuron in enumerate(neuron_list):
        # if int(neuron.quality) <= int(maxqual):
        if int(neuron.quality) in maxqual:
            len_neurons += 1
    # print("len_neurons ", len_neurons)

    if pltname is None:
        pltname = 'neuron_waveforms.png'

    ind = 0
    # fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    plt.figure(figsize=(14, 14))
    for neuron in neuron_list:
        # if neuron.quality <= maxqual:
        if neuron.quality in maxqual:
            ind += 1
            # print("ind ", ind,
            #       " np.ceil(np.sqrt(len_neurons)) ",
            #       np.ceil(np.sqrt(len_neurons)))
            plt.subplot(int(np.ceil(np.sqrt(len_neurons))),
                        int(np.ceil(np.sqrt(len_neurons))), ind)
            plt.plot(neuron.waveform,
                     label=str("ch " +
                               str(neuron.peak_channel) +
                               " id " +
                               str(neuron.clust_idx) +
                               "  qual " +
                               str(neuron.quality)))
            if ind == 0:
                plt.title(str(pltname))
            plt.legend(frameon=False, loc='best', prop={'size': 8})
            # plt.legend(frameon=False, loc='lower right', prop={'size': 9})
            plt.xticks([])
            # plt.yticks([])
            plt.yticks(fontsize=7)
            plt.xlabel("")
            plt.box(False)

    if saveloc is None:
        plt.show()
    else:
        # plt.savefig(op.join(saveloc, pltname), dpi=640)
        plt.savefig(op.join(saveloc, pltname))


def n_getspikes(neuron_list, start=False, end=False, lonoff=1):

    '''
    Extracts spiketimes to a list from neuron_list
    Unless otherwise specified start, end are in seconds

    n_getspikes(neuron_list, start=False, end=False)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    lonoff : Apply on off times (default on, 1)

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

    if start < neuron_list[0].start_time:
        raise ValueError('start is less than neuron_list[0].start_time')
    if end > neuron_list[0].end_time:
        raise ValueError('end is greater than neuron_list[0].end_time')

    # Create empty list
    spiketimes_allcells = []

    # Loop through and get spike times
    for idx, neuron_l in enumerate(neuron_list):
        logger.debug('Getting spiketimes for cell %s', str(idx))

        # get spiketimes for each cell and append
        if lonoff:
            spiketimes = neuron_l.spike_time_sec_onoff
        else:
            spiketimes = neuron_l.spike_time_sec
        idx_l = \
            np.where(np.logical_and(spiketimes >= start, spiketimes <= end))[0]
        spiketimes = spiketimes[idx_l]
        spiketimes_allcells.append(spiketimes)

    return spiketimes_allcells


def n_spiketimes_to_spikewords(neuron_list, binsz=0.02,
                               start=False, end=False,
                               binarize=0, lonoff=1):
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
    lonoff : Apply on off times (default on, 1)

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
    spiketimes = n_getspikes(neuron_list, start=start, end=end,
                             lonoff=lonoff)

    # convert time to milli seconds
    start = start * conv_mills
    end = end * conv_mills
    binsz = binsz * conv_mills

    # startime in bins
    binrange = np.arange(start, (end + binsz), binsz)
    n_cells = len(spiketimes)

    # initialize array
    spikewords_array = np.zeros([n_cells, binrange.shape[0]-1])

    # loop over cells and find counts/binarize
    for i in range(n_cells):

        # spiketimes in seconds to ms
        spiketimes_cell = np.asarray(spiketimes[i]) * conv_mills
        counts, bins = np.histogram(spiketimes_cell, bins=binrange)

        # binarize the counts
        if binarize == 1:
            counts[counts > 0] = 1
        spikewords_array[i, :] = counts

    if binarize == 1:
        if (spikewords_array < 128).all():
            return (spikewords_array.astype(np.int8))
        else:
            return (spikewords_array.astype(np.int32))
    elif binarize == 0:
        if (spikewords_array < 2147483647).all():
            return (spikewords_array.astype(np.int32))
        else:
            return (spikewords_array.astype(np.int64))


def n_branching_ratio(neuron_list, ava_binsz=0.004,
                      kmax=500,
                      start=0, end=2,
                      binarize=1,
                      plotname=None):
    '''
    This function is a wrapper for branching ratio using
    "complex" and "exp_offset" fit

    Articles
    Inferring collective dynamical states from widely unobserved systems
    Jens Wilting & Viola Priesemann
    and
    Nature Communications volume 9, Article number: 2325 (2018)
    MR. Estimator, a toolbox to determine intrinsic timescales from subsampled
    spiking activity
    F. P. Spitzner, J. Dehning, J. Wilting, A. Hagemann, J. P. Neto,
    J. Zierenberg, V. Priesemann
    https://arxiv.org/abs/2007.03367

    br1, br2, acc1, acc2=
    n_branching_ratio(neuron_list, ava_binsz=0.002
                      kmax=500,
                      start=False, end=False,
                      binarize=1,
                      plotname=None):

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    ava_binsz : Bin size (default 0.002 (4 ms))
    start : default, 0 from starting time
    end : default, 2 from starting time to 2 hours
    binarize : Get counts (default 0) in bins,
    if binarize is 1,binarize to 0 and 1
    plotname : location of figure to save

    Returns
    -------
    br1: Branching ratio with complex
    br2: Branching ratio with exp_offset
    acc1 : pearsons correlation a test way to not check plot, complex
    acc2 : pearsons correlation a test way to not check plot, exp_offset

    Raises
    ------
    ValueError if neuron list is empty

    See Also
    --------

    Notes
    -----

    Examples
    --------
    br1, br2, acc1, acc2 =
    n_branching_ratio(neuron_list, ava_binsz=0.004
                      kmax=500,
                      start=False, end=False,
                      binarize=1,
                      plotname=None):

    '''

    try:
        import mrestimator as mre
    except ImportError:
        raise ImportError('''Run command :
                 git clone https://github.com/Priesemann-Group/mrestimator.git
                 cd mrestimator
                 pip install .''')
    from sys import platform
    if platform == "darwin":
        # matplotlib.use('TkAgg')
        plt.switch_backend('TkAgg')
    else:
        # matplotlib.use('Agg')
        plt.switch_backend('Agg')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # lplot = 1
    # ava_binsz = 0.004
    # start = 0
    # end = 2
    # kmax = 500

    # filter neural data
    # newlist = [x for x in n if x.quality is in [filt]]
    # print("len newlist ", len(newlist))

    data = n_spiketimes_to_spikewords(neuron_list,
                                      ava_binsz,
                                      3600*start,
                                      3600*end,
                                      binarize)

    # dt = ava_binsz * 1000
    dt = 1
    A_t = np.sum(data, 0)
    src = mre.input_handler(A_t)
    rks = mre.coefficients(src, steps=(1, kmax),
                           dt=dt, dtunit='ms',
                           method='trialseparated')

    fit1 = mre.fit(rks, fitfunc="complex")
    fit2 = mre.fit(rks, fitfunc='exp_offset')
    # print("method ", method, " br ", fit1.mre)

    if plotname is not None:
        fig, ax1 = plt.subplots()
        ax1.plot(rks.steps, rks.coefficients, '.k', alpha=0.2, label=r'Data')
        ax1.plot(rks.steps, mre.f_complex(rks.steps*dt, *fit1.popt),
                 label='complex m={:.5f}'.format(fit1.mre))
        ax1.plot(rks.steps, mre.f_exponential_offset(rks.steps*dt, *fit2.popt),
                 label='exp_offset m={:.5f}'.format(fit2.mre))
        ax1.set_xlabel(r'Time lag $\delta t$')
        ax1.set_ylabel(r'Autocorrelation $r_{\delta t}$')
        ax1.legend()
        if plotname is not None:
            plt.savefig(plotname)
            plt.close()

    fit_acc1 = sc.stats.pearsonr(rks.coefficients,
                                 mre.f_complex(rks.steps*dt,
                                               *fit1.popt))[0]
    fit_acc2 = sc.stats.pearsonr(rks.coefficients,
                                 mre.f_exponential_offset(rks.steps*dt,
                                                          *fit2.popt))[0]
    # print(sc.stats.pearsonr(rks.coefficients, rks.coefficients))
    # print(sc.stats.pearsonr(rks.coefficients, np.random.random(500)))
    return fit1.mre, fit2.mre, fit_acc1, fit_acc2


def n_zero_crosscorr(neuron_list, ltetrode=1):
    '''

    Calculate zero lag cross correlation in descending order
    and return neuron indices

    n_zero_crosscorr(neuron_list, ltetrode=1)

    returns
     zcl_in, zcl_vn

     zcl_in : neuron pairs
     zcl_vn : zero lag cross cor values in descending order
    '''

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    zero_lag_list_index = None
    zero_lag_list_index = []
    zero_lag_list_values = None
    zero_lag_list_values = []
    waveform_mse_list = None
    waveform_mse_list = []

    for i in range(len(neuron_list)):
        for j in range(i+1, len(neuron_list)):
            # select spike timestamps within on/off times for ith neuron
            stimes1 = neuron_list[i].spike_time_sec_onoff
            stimes2 = neuron_list[j].spike_time_sec_onoff
            _, mse, _ = wf_comparison(neuron_list[i].waveforms,
                                      neuron_list[j].waveforms)
            corr = sc.stats.zscore(fftconvolve(stimes1, stimes2))
            zero_lag_corr = corr[(len(corr) - 1)//2]
            zero_lag_list_index.append([i, j])
            zero_lag_list_values.append(zero_lag_corr)
            waveform_mse_list.append(mse)

    # get descending order and apply to index and values
    order = np.argsort(zero_lag_list_values)[::-1]
    zcl_in = np.asarray(zero_lag_list_index)
    zcl_vn = np.asarray(zero_lag_list_values)
    zcl_mse = np.asarray(waveform_mse_list)
    zcl_in = zcl_in[order]
    zcl_vn = zcl_vn[order]
    zcl_mse = zcl_mse[order]

    return zcl_in, zcl_vn, zcl_mse


def n_save_modified_neuron_list(neuron_list, file_name):

    '''
    Save modified neuron_list

    n_save_modified_neuron_list(neuron_list, file_name)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    file_name : Filename with path, '/home/kbn/neuron_withqual.npy'

    Returns
    -------
    spiketimes_allcells : List of all spiketimes

    Raises
    ------
    ValueError if neuron list is empty
    FileExistsError if file_name exists

    See Also
    --------

    Notes
    -----

    Examples
    --------
    n_save_modified_neuron_list(neuron_list, '/home/kbn/neuron_withqual.npy')

    '''

    logger.info('Saving modified neuron_list')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # check file exist
    if op.exists(file_name) and op.isfile(file_name):
        raise FileExistsError("File {} exists".format(file_name))

    # Save
    np.save(file_name, neuron_list)
    logger.info('Saved modified neuron_list, %s.', file_name)


def n_check_date_validity(neurons_group0_files_list,
                          surgeryday_time_string,
                          sacday_time_string,
                          birthday_time_string=None):
    '''
    def n_check_date_validity(neurons_group0_files_list,
                              sujday_time_string,
                              sacday_time_string,
                              birthday_time_string=None)

    neurons_group0_files_list : list of neurons_group0.npy files
    surgeryday_time_string : surgery date in the format ('2021-04-15_07-30-00')
                same as ecube file time format
    sacday_time_string : sac date in the format ('2021-12-10_07-30-00')
                same as ecube file time format
    birthday_time_string : birthday default(None)
                get birthday from neurons_grou0 file
                or give date in the format ('2021-01-10_07-30-00')
                same as ecube file time format

    returns None
        crashes if the checks fails


    import numpy as np
    import musclebeachtools as mbt
    import glob

    main_dir = '/media/HlabShare/Clustering_Data/CAF00080/'
    neurons_group0_files_list = \
        sorted(glob.glob(main_dir + '/*/*/*/*/H*neurons_group0*'))
    n_check_date_validity(neurons_group0_files_list,
                          surgeryday_time_string='2021-02-04_07-30-00',
                          sacday_time_string='2021-03-08_07-30-00',
                          birthday_time_string='2020-04-19_07-30-00')

    '''

    for neurons_file in neurons_group0_files_list:
        # check file exist
        if not (op.exists(neurons_file) and op.isfile(neurons_file)):
            raise FileNotFoundError("File {} not found".format(neurons_file))

        neurons = np.load(neurons_file, allow_pickle=True)
        # check neurons is not empty
        if (len(neurons) == 0):
            raise ValueError(f'Neuron list is empty {neurons_file}')

        if birthday_time_string is None:
            birthday = neurons[0].birthday
        else:
            birthday = \
                datetime.strptime(birthday_time_string.replace("_", " "),
                                  "%Y-%m-%d %H-%M-%S")

        sujday = \
            datetime.strptime(surgeryday_time_string.replace("_", " "),
                              "%Y-%m-%d %H-%M-%S")

        sacday = \
            datetime.strptime(sacday_time_string.replace("_", " "),
                              "%Y-%m-%d %H-%M-%S")

        e_time_string = neurons[0].rstart_time
        clust_date = datetime.strptime(e_time_string.replace("_", " "),
                                       "%Y-%m-%d %H-%M-%S")

        # Check clustering not happened before birthday
        if (clust_date - birthday).total_seconds() < 0:
            print(f'birthday is {birthday}')
            print(f'surgery date is {sujday}')
            print(f'Clustering date is {clust_date}')
            print(f'sac date is {sacday}')
            raise ValueError(f'Clustering data before surjery {neurons_file}')

        # Check clustering not happened before surjery
        if (clust_date - sujday).total_seconds() < 0:
            print(f'birthday is {birthday}')
            print(f'surgery date is {sujday}')
            print(f'Clustering date is {clust_date}')
            print(f'sac date is {sacday}')
            raise ValueError(f'Clustering data before surjery {neurons_file}')

        # Check clustering not happened after sacday
        if (sacday - clust_date).total_seconds() < 0:
            print(f'birthday is {birthday}')
            print(f'surgery date is {sujday}')
            print(f'Clustering date is {clust_date}')
            print(f'sac date is {sacday}')
            raise ValueError(f'Clustering data after sac date {neurons_file}')


def load_spike_amplitudes(neuron_list, file_name):

    '''
    Get spike amplitudes from numpy list

    load_spike_amplitudes(neuron_list, file_name)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    file_name : Filename with path, '/home/kbn/neuron_amplitudes.npy',
                output from spike_interface

    Returns
    -------
    neuron_list_with_amplitudes : neuron list with amplitudes in
                                  n[ ].spike_amplitude field

    Raises
    ------
    ValueError if neuron list is empty
    FileNotFoundError if file_name not found
    ValueError if each clusters spike_time and spike_amplitude
               has different lengths

    See Also
    --------

    Notes
    -----

    Examples
    --------
    neuron_list_with_amplitudes = \
            load_spike_amplitudes(neuron_list,
                                  '/home/kbn/neuron_amplitudes.npy')

    '''
    logger.info('Updating spike_amplitude')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # check file exist
    if not (op.exists(file_name) and op.isfile(file_name)):
        raise FileNotFoundError("File {} not found".format(file_name))

    # Load file, loop and update
    sp_amp = load_np(file_name, lpickle=True)
    for neuron in neuron_list:
        len_sp_amp = len(sp_amp[neuron.clust_idx])
        len_sp_t = len(neuron.spike_time)
        if (len_sp_amp != len_sp_t):
            raise \
                ValueError('Clust {} length of spike_time {} != amplitudes {}'
                           .format(neuron.clust_idx,
                                   len_sp_amp,
                                   len_sp_t))
        neuron.spike_amplitude = sp_amp[neuron.clust_idx]

    return neuron_list


def load_spike_waveform_tet(neuron_list, file_name):

    '''
    Get spike waveforms from numpy list and update waveform_tetrodes

    load_spike_waveform_tet(neuron_list, file_name)

    Parameters
    ----------
    neuron_list : List of neurons
    file_name : Filename with path,
                '/home/kbn/neuron_waveforms_group0.npy',
                output from spike_interface

    Returns
    -------
    neuron_list_with_waveform : neuron list with waveform in
                                  n[ ].waveform field

    Raises
    ------
    ValueError if neuron list is empty
    FileNotFoundError if file_name not found
    ValueError if filename does not contain _waveforms_group
    ValueError if length of neuron list not same as length of waveform list

    See Also
    --------

    Notes
    -----

    Examples
    --------
    neuron_list_with_waveform = \
            load_spike_waveform_tet(neuron_list,
                                  '/home/kbn/neuron_waveforms_group0.npy')
    '''

    logger.info('Updating neurons[].waveform_tetrodes')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # check file exist
    if not (op.exists(file_name) and op.isfile(file_name)):
        raise FileNotFoundError("File {} not found".format(file_name))

    # Check _waveforms_group
    if "_waveforms_group" in file_name:
        logger.debug("_waveforms_group")
    else:
        raise \
            ValueError('Filename error: not _waveforms_group')

    # Load file, loop and update
    sp_wft = load_np(file_name, lpickle=True)

    # check length of neuron list and waveform list same
    if (len(sp_wft) != len(neuron_list)):
        raise \
            ValueError('Length of neuron list {} != length of waveform list {}'
                       .format(len(sp_wft),
                               len(neuron_list)))

    for neuron in neuron_list:
        if "_waveforms_group" in file_name:
            logger.debug('Neuron %s', neuron.clust_idx)
            neuron.waveform_tetrodes = \
                np.mean(np.asarray(sp_wft[neuron.clust_idx]).T, axis=2)

    if "_waveforms_group" in file_name:
        logger.info('Updated neurons[].waveform_tetrodes')

    return neuron_list


def load_spike_waveforms_be(neuron_list, file_name):

    '''
    Get spike waveforms_be from numpy list

    load_spike_waveforms_be(neuron_list, file_name)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    file_name : Filename with path,
                '/home/kbn/neuron_b_waveforms_group0.npy',
                or
                '/home/kbn/neuron_e_waveforms_group0.npy',
                output from spike_interface

    Returns
    -------
    neuron_list_with_waveforms_be : neuron list with waveform_bes in
                                  n[ ].spike_waveform_be field

    Raises
    ------
    ValueError if neuron list is empty
    FileNotFoundError if file_name not found
    ValueError if filename does not contain _b_waveforms_ or _e_waveforms_
    ValueError if length of neuron list not same as length of waveform list

    See Also
    --------

    Notes
    -----

    Examples
    --------
    neuron_list_with_waveform_bes = \
            load_spike_waveform_bes(neuron_list,
                                  '/home/kbn/neuron_b_waveforms_group0.npy')

    '''

    logger.info('Updating neurons[].wf_b/e')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    # check file exist
    if not (op.exists(file_name) and op.isfile(file_name)):
        raise FileNotFoundError("File {} not found".format(file_name))

    # Check _b_ or _e_
    if "_b_waveforms_" in file_name:
        logger.debug("_b_waveforms_")
    elif "_e_waveforms_" in file_name:
        logger.debug("_e_waveforms_")
    else:
        raise \
            ValueError('Filename error: not _b_waveforms_ or _e_waveforms_')

    # Load file, loop and update
    sp_wf = load_np(file_name, lpickle=True)

    # check length of neuron list and waveform list same
    if (len(sp_wf) != len(neuron_list)):
        raise \
            ValueError('Length of neuron list {} != length of waveform list {}'
                       .format(len(sp_wf),
                               len(neuron_list)))

    for neuron in neuron_list:
        if "_b_waveforms_" in file_name:
            neuron.wf_b = np.asarray(sp_wf[neuron.clust_idx]).T
        elif "_e_waveforms_" in file_name:
            neuron.wf_e = np.asarray(sp_wf[neuron.clust_idx]).T

    if "_b_waveforms_" in file_name:
        logger.info('Updated neurons[].wf_b')
    if "_e_waveforms_" in file_name:
        logger.info('Updated neurons[].wf_e')

    return neuron_list


def cell_isi_hist(time_s, start, end, isi_thresh=0.1,
                  nbins=101, lplot=1):

    '''
    Return a histogram of the interspike interval (ISI) distribution.
    This is typically used to evaluate whether a spike train exhibits
    a refractory period and is thus consistent with a
    single unit or a multi-unit recording.
    This function will plot the bar histogram of that distribution
    and calculate the percentage of ISIs that fall under 2 msec.

    cell_isi_hist(time_s, start=False, end=False, isi_thresh=0.1,
                  nbins=101, lplot=1)

    Parameters
    ----------
    time_s : time in seconds
    start : Start time
    end : End time
    isi_thresh : isi threshold (default 0.1)
    nbins : Number of bins (default 101)
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
    cell_isi_hist(time_s, start, end, isi_thresh=0.1, nbins=101, lplot=1)

    '''

    # For a view of how much this cell is like a "real" neuron,
    # calculate the ISI distribution between 0 and 100 msec.

    logger.info('Calculating ISI')

    logger.debug('start and end is %s and %s', start, end)

    # Calulate isi
    idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
    ISI = np.diff(time_s[idx_l])

    # plot histogram and calculate contamination
    edges = np.linspace(0, isi_thresh, nbins)
    hist_isi = np.histogram(ISI, edges)

    # Calculate contamination percentage
    contamination = 100*(sum(hist_isi[0][0:int((0.1/isi_thresh) *
                         (nbins-1)/50)])/sum(hist_isi[0]))
    contamination = round(contamination, 2)
    cont_text = 'Contamination is {} percent.' .format(contamination)
    logger.info('Contamination is {} percent.' .format(contamination))

    if lplot:
        plt.ion()
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
    return ISI, edges, hist_isi


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


def one_channel_projected_spikes(ax, chan_dat, all_cells, clust_num, chan_num,
                                 loc, window_len, fs=25000):
    '''
    plots one chan with spikes of that cell highlighted on it and the other
    cells on that channel plotted lighter

    chan_dat: channel data already loaded
    all_cells: all the clusters already loaded
    clust_num: main cluster in question
    chan_num: what channel this is
    loc: where in the data to start looking in sample points
    window_len: num sample points to look at
    fs: sample rate
    '''

    data = ntk.butter_bandpass(chan_dat, 500, 10000, fs=25000)
    st = loc
    if st < 0:
        st = 0
    ax.plot(np.arange(st, loc+window_len), data, color='darkgrey')

    cells_on_chan = \
        [cell for cell in all_cells if cell.peak_channel == chan_num]
    dcolors = sns.color_palette(palette="deep", n_colors=len(cells_on_chan))

    for i, cell in enumerate(cells_on_chan):
        print("clust_num ", clust_num)
        print("cell ", cell)
        print("cell.amp ", cell.mean_amplitude)
        # if 1:
        if cell.clust_idx == clust_num:
            print("cell ", cell, " clust_num ", clust_num)
            spikes = cell.spike_time
            spikes = spikes[spikes < loc+window_len]
            spikes = spikes[spikes > loc]
            spikes = spikes - loc

            if cell.clust_idx == clust_num:
                alpha = 1
            else:
                alpha = 0.4

            for spike in spikes:
                ax.plot(np.arange((spike+loc)-50, (spike+loc)+50),
                        data[spike-50:spike+50], color=dcolors[i], alpha=alpha)


def find_5_min(start_string, raw_dir, sample_point, fs=25000):
    whole_dir = sorted(glob.glob(raw_dir+'/*'))
    first_file = [x for x in whole_dir if start_string in x]
    idx = whole_dir.index(first_file[0])

    sps_per_file = 300*fs
    num_files_away = int(sample_point/sps_per_file)
    print(num_files_away)

    st_base = start_string[:14]
    minu = int(start_string[14:16])
    new_min = minu+(num_files_away*5)
    new_time_str = st_base+str(new_min)
    print(new_time_str)

    new_file = whole_dir[idx+num_files_away]

    return new_file


def amp_spike_projection(cells, clust_num, num_chans, probe_num, raw_dir,
                         chan_map, window_len=25000, fs=25000):

    '''
    def amp_spike_projection(cells, clust_num, num_chans, probe_num, raw_dir,
                             chan_map, window_len=25000, fs=25000)
    '''

    # check cells
    if (len(cells) == 0):
        raise ValueError('Neuron list is empty')

    # check clust_num not negative
    if (clust_num < 0):
        raise ValueError('clust_num is less than zero')

    # check num_chans not below 64
    if (num_chans < 64):
        raise ValueError('num_chans is less than 64')

    # check probe_num not zero
    if (probe_num < 1):
        raise ValueError('probe_num is less than 1')

    # check raw_dir exists
    if not (os.path.exists(raw_dir) and os.path.isdir(raw_dir)):
        raise NotADirectoryError("Directory {} does not exists".
                                 format(raw_dir))

    point = [0]

    def onclick(event):
        point[0] = np.int(event.xdata)
        print("You have selected ", point[0])

    plt.ion()
    cell = [cell for cell in cells if cell.clust_idx == clust_num][0]

    peak_channel = cell.peak_channel
    print("peak_channel ", peak_channel)

    start_string = cell.rstart_time

    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    gs = fig.add_gridspec(2, 3)

    # plot wf
    wf_ax = fig.add_subplot(gs[:-1, 0])
    amp_ax = fig.add_subplot(gs[0, 1:])
    raw_ax = fig.add_subplot(gs[-1, :])

    line_ax = amp_ax.twinx()

    wf_sh = cell.waveform_tetrodes.shape[0]
    wf_ax.plot(np.linspace(0, (wf_sh * 1000.0) / cell.fs,
                           wf_sh),
               cell.waveform_tetrodes,
               color='#6a88f7')
    wf_ax.plot(np.linspace(0, (wf_sh * 1000.0) / cell.fs,
                           wf_sh),
               cell.waveform,
               'g*')

    # plot amp
    th_zs = 2
    z = np.abs(sc.stats.zscore(cell.spike_amplitude))
    amp_ax.plot((cell.spike_time / cell.fs),
                cell.spike_amplitude, 'bo',
                markersize=1.9, alpha=0.2)
    amp_ax.plot((cell.spike_time[np.where(z > th_zs)] / cell.fs),
                cell.spike_amplitude[np.where(z > th_zs)],
                'ro',
                markersize=1.9, alpha=0.2)
    amp_ax.set_xlabel('Time (s)')
    amp_ax.set_ylabel('Amplitudes', labelpad=-3)
    amp_ax.set_xlim(left=(cell.start_time),
                    right=(cell.end_time))
    amp_ax.set_ylim(bottom=0,
                    top=(min((np.mean(cell.spike_amplitude) +
                             (3*np.std(cell.spike_amplitude))),
                         np.max(cell.spike_amplitude))))
    amp_stats_str = '\n'.join((
        r'$Min: %d, Max: %d$' % (np.min(cell.spike_amplitude),
                                 np.max(cell.spike_amplitude), ),
        r'$Mean:%d, Med:%d, Std:%d$'
        % (np.mean(cell.spike_amplitude),
            np.median(cell.spike_amplitude),
            np.std(cell.spike_amplitude), )))
    props = dict(boxstyle='round', facecolor='wheat',
                 alpha=0.5)
    amp_ax.text(0.73, 0.27, amp_stats_str,
                transform=amp_ax.transAxes,
                fontsize=8,
                verticalalignment='top', bbox=props)

    done = False

    fig.canvas.mpl_connect('button_press_event', onclick)

    while (not done):
        # plt.show()
        print("Press amplitude plot to plot corresponding raw data")
        try:
            plt.waitforbuttonpress()
        except Exception as e:
            print("Error ", e)
            raise RuntimeError('Plot closed')
        line_ax.clear()
        raw_ax.clear()
        print("samp point: ", point[0])

        file_to_load = find_5_min(start_string, raw_dir, point[0]*fs, fs=25000)
        print("file_to_load ", file_to_load)

        ts = ((point[0] % 300) * fs)
        if ts < 0:
            ts = 0

        te = ts + window_len
        if te > (300 * fs):
            te = -1
        t, dat = ntk.load_raw_gain_chmap_1probe(file_to_load, num_chans,
                                                chan_map,
                                                nprobes=int(num_chans/64),
                                                lraw=1, ts=ts, te=te,
                                                probenum=probe_num-1,
                                                probechans=64)

        ch_dat = dat[peak_channel, :]

        one_channel_projected_spikes(raw_ax, ch_dat, cells, clust_num,
                                     peak_channel, point[0]*fs,
                                     window_len, fs=25000)
        line_ax.plot([point[0], point[0]], [0, line_ax.get_ylim()[1]],
                     color="black")
        plt.show()
        plt.pause(1)


if __name__ == "__main__":

    datadir = "/hlabhome/kiranbn/neuronclass/t_lit_EAB50final/"
    n1 = ksout(datadir, filenum=0, prbnum=4, filt=[1, 3])
