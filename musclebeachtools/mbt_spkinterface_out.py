import glob
import os
import os.path as op
import pickle
import numpy as np
import neuraltoolkit as ntk
from musclebeachtools import mbt_neurons as mb


def siout(sorted_data, noflylist, rec_time,
          file_datetime_list, ecube_time_list,
          amps=None,
          wf_b=None, wf_e=None,
          filt=None,
          t_ch_size=None,
          model_file='/media/HlabShare/models/xgboost_autoqual_prob',
          sex=None, birthday=None, species=None,
          animal_name=None,
          region_loc=None,
          genotype=None,
          expt_cond=None,
          lskipautoqual=None):
    '''
    function to load neuron objects from the spike interface output

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

    # print("Finding unit ids")
    unique_clusters = sorted_data.get_unit_ids()

    # Sampling rate
    # print('Finding sampling rate')
    # mb.Neuron.fs = sorted_data.get_sampling_frequency()
    fs = sorted_data.get_sampling_frequency()
    print("Sampling frequency ", fs)
    n = []

    # Start and end time
    # print('Finding start and end time')
    # start_time = raw_data_start
    # end_time = raw_data_end
    # # Convert to seconds
    # end_time = (np.double(np.int64(end_time) - np.int64(start_time))/1e9)
    # # reset start to zero
    # start_time = np.double(0.0)
    # print((end_time - start_time))
    # assert (end_time - start_time) > 1.0, \
    #     'Please check start and end time is more than few seconds apart'
    # print('Start and end times are %f and %f', start_time, end_time)

    # mb.Neuron.start_time = 0.0
    start_time = 0.0

    # KIRAN this shouldn't be hard coded? max of spt below
    # mb.Neuron.end_time = 300.0
    end_time = rec_time / fs
    print("Start time ", start_time, " end time ", end_time)

    # Loop through unique clusters and make neuron list
    for unit_idx, unit in enumerate(unique_clusters):
        ch_group = sorted_data.get_unit_property(unit, "group")
        # print("Total i ", i, " unit ", unit)
        if unit_idx not in noflylist:
            # print("qual ", cluster_quals[i])
            if len(filt) == 0:
                # this is the unit number for indexing spike times
                # and unit properties
                sp_c = [unit_idx]
                # these are spike times
                sp_t = sorted_data.get_unit_spike_train(unit)
                qual = 0
                # mean WF @ Fs of recording
                mwf_list = sorted_data.get_unit_property(unit, "template").T
                # print("mwf_list ", mwf_list)
                # print("len mwf_list ", len(mwf_list))
                # mwfs = np.arange(0, 100)
                # KIRAN please spline this
                # mwfs = sorted_data.get_unit_property(unit, "template").T
                tmp_max_channel = sorted_data.get_unit_property(unit,
                                                                'max_channel')
                # print("t_ch_size ", t_ch_size)
                max_channel = \
                    [sorted_data.get_unit_property(unit, 'max_channel') +
                        (t_ch_size * ch_group)]
                #  try:
                # print("Skipped i ", i, " unit ", unit)
                print("unit_idx ", unit_idx, " unit ", unit,
                      " max_channel : ", tmp_max_channel,
                      " ", max_channel)
                mwf = [row[tmp_max_channel] for row in mwf_list]
                t = np.arange(0, len(mwf))
                _, mwfs = ntk.data_intpl(t, mwf, 3, intpl_kind='cubic')

                # def __init__(self, sp_c, sp_t, qual, mwf, mwfs, max_channel,
                #              fs=25000, start_time=0, end_time=12 * 60 * 60,
                #              mwft=None,
                #              sex=None, birthday=None, species=None):
                if ((len(file_datetime_list) == 2) and
                        (len(ecube_time_list) == 2)):
                    if (amps is not None):
                        # print("amps not None")
                        # print("amps not None")
                        # print("mean amps ", unit_idx)
                        tmp_mean_amps = np.mean(np.asarray(amps[unit_idx]))
                        if tmp_mean_amps >= 20:
                            print("mean amps ", unit_idx, " ", tmp_mean_amps,
                                  flush=True)
                            if ((wf_b is not None) and (wf_e is not None)):
                                n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                                         mwfs, max_channel,
                                         fs=fs,
                                         start_time=start_time,
                                         end_time=end_time,
                                         mwft=mwf_list,
                                         rstart_time=str(file_datetime_list
                                                         [0]),
                                         rend_time=str(file_datetime_list[1]),
                                         estart_time=np.int64(ecube_time_list
                                                              [0]),
                                         eend_time=np.int64(ecube_time_list
                                                            [1]),
                                         sp_amp=amps[unit_idx],
                                         wf_b=np.asarray(wf_b[unit_idx]).T,
                                         wf_e=np.asarray(wf_e[unit_idx]).T,
                                         sex=sex, birthday=birthday,
                                         species=species,
                                         animal_name=animal_name,
                                         region_loc=region_loc,
                                         genotype=genotype,
                                         expt_cond=expt_cond))
                            #
                            elif (wf_b is None):
                                n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                                         mwfs, max_channel,
                                         fs=fs,
                                         start_time=start_time,
                                         end_time=end_time,
                                         mwft=mwf_list,
                                         rstart_time=str(file_datetime_list
                                                         [0]),
                                         rend_time=str(file_datetime_list[1]),
                                         estart_time=np.int64(ecube_time_list
                                                              [0]),
                                         eend_time=np.int64(ecube_time_list
                                                            [1]),
                                         sp_amp=amps[unit_idx],
                                         sex=sex, birthday=birthday,
                                         species=species,
                                         animal_name=animal_name,
                                         region_loc=region_loc,
                                         genotype=genotype,
                                         expt_cond=expt_cond))

                        else:
                            print("Not added mean amps ", unit_idx, " ",
                                  tmp_mean_amps, flush=True)

                    elif (amps is None):
                        n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                                 mwfs, max_channel,
                                 fs=fs,
                                 start_time=start_time, end_time=end_time,
                                 mwft=mwf_list,
                                 rstart_time=str(file_datetime_list[0]),
                                 rend_time=str(file_datetime_list[1]),
                                 estart_time=np.int64(ecube_time_list[0]),
                                 eend_time=np.int64(ecube_time_list[1]),
                                 sex=sex, birthday=birthday, species=species,
                                 animal_name=animal_name,
                                 region_loc=region_loc,
                                 genotype=genotype,
                                 expt_cond=expt_cond))
                elif ((len(file_datetime_list) == 2) and
                        (len(ecube_time_list) == 0)):
                    n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                             mwfs, max_channel,
                             fs=fs,
                             start_time=start_time, end_time=end_time,
                             mwft=mwf_list,
                             rstart_time=str(file_datetime_list[0]),
                             rend_time=str(file_datetime_list[1]),
                             sex=sex, birthday=birthday, species=species,
                             animal_name=animal_name,
                             region_loc=region_loc,
                             genotype=genotype,
                             expt_cond=expt_cond))
                else:
                    n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                             mwfs, max_channel,
                             fs=fs,
                             start_time=start_time, end_time=end_time,
                             mwft=mwf_list,
                             sex=sex, birthday=birthday, species=species,
                             animal_name=animal_name,
                             region_loc=region_loc))
                #  except:
                #  pdb.set_trace()
            elif len(filt) > 0:
                print("sorry, we don't have qualities set yet, "
                      "run again with no filter")

    print(f'Found {len(n)} neurons\n')
    if lskipautoqual is None:
        if op.exists(model_file) and op.isfile(model_file):
            print("model_file was used ", model_file, flush=True)
            mb.autoqual(n, model_file)
        else:
            print("Model_file {} does not exists".format(model_file),
                  flush=True)
            print("neurons[0].quality not calculated")

    return n


def mbt_spkinterface_out(
     clust_out_dir,
     model_file='/media/HlabShare/models/xgboost_autoqual_prob',
     sex=None, birthday=None, species=None,
     animal_name=None,
     region_loc=None,
     genotype=None,
     expt_cond=None):

    '''
    Function loads spikeinterface output to neuron

    mbt_spkinterface_out('spikeinterface_output_directory',
                         model_file,
                         sex=None, birthday=None, species=None,
                         animal_name=None,
                         region_loc=None,
                         genotype=None,
                         expt_cond=None)

    Parameters
    ----------
    spikeinterface_output_directory : spikeinterface output directory
    model_file : path of model file
    sex: 'm' or 'f'
    birthday: datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
    species='r' or 'm', rat or mice
    animal_name : UUU12345
    region_loc : string , ca1, v1, m1
    genotype : "wt",
    expt_cond : "experimental condition",

    Returns
    -------
    cells : clusters found in spikeinterface output

    Raises
    ------
    NotADirectoryError
    See Also
    --------

    Notes
    -----

    Examples
    --------
    mbt_spkinterface_out('/home/kbn/co/',
                         model_file,
                         sex=None, birthday=None, species=None,
                         animal_name=None,
                         region_loc=None)

    '''

    # constants / variables
    filt = None
    noflylist = []

    # check file exist
    if op.exists(clust_out_dir) and op.isdir(clust_out_dir):
        print("clust_out_dir ", clust_out_dir)
    else:
        raise NotADirectoryError("Directory {} not found"
                                 .format(clust_out_dir))

    os.chdir(clust_out_dir)

    try:
        sort_pickle_file = op.join(clust_out_dir,
                                   'spi_dict_final.pickle')
        print("picklefile_selected ", sort_pickle_file)
        pickle_in = open(sort_pickle_file, "rb")
        sorted_data = pickle.load(pickle_in)
        pickle_in.close()
    except Exception as e:
        print("Error : ", e)
        raise FileNotFoundError('Error loading spi_dict_final.pickle')

    rl = glob.glob('*rec_length0.npy')[0]
    rec_time = mb.load_np(rl, 1)

    rl = glob.glob('*_file_datetime_list.npy')[0]
    file_datetime_list = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*ecube_time_list.npy')[0]
    ecube_time_list = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_amplitudes0.npy')[0]
    amps = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_b_waveforms_group0.npy')[0]
    wf_b = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_e_waveforms_group0.npy')[0]
    wf_e = mb.load_np(rl, lpickle=True)

    try:
        rl = glob.glob('*_t_ch_size0.npy')[0]
        t_ch_size = np.int(mb.load_np(rl, lpickle=True))
    except Exception as e:
        print("Error: ", e)
        t_ch_size = 4
        print("Setting t_ch_size to 4")

    cells = siout(sorted_data, noflylist, rec_time,
                  file_datetime_list, ecube_time_list,
                  amps=amps,
                  wf_b=wf_b, wf_e=wf_e,
                  filt=filt,
                  t_ch_size=t_ch_size,
                  model_file=model_file,
                  sex=sex, birthday=birthday, species=species,
                  animal_name=animal_name,
                  region_loc=region_loc,
                  genotype=genotype,
                  expt_cond=expt_cond)
    return cells
