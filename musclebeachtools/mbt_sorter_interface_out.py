import glob
import os
import os.path as op
import pickle
import numpy as np
import neuraltoolkit as ntk
import musclebeachtools as mb


def ms5out(sorted_data, noflylist, rec_time,
           file_datetime_list, ecube_time_list,
           wfs=None,
           peak_channels=None,
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
           lskipautoqual=None,
           min_amps=10,
           lverbose=0):
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
    if lverbose:
        print("dir sorted_data ", dir(sorted_data), flush=True)
        print(f'amps {amps}')
    amps = amps[0]
    amps_keys = list(amps.keys())
    if filt is None:
        filt = []

    # print("Finding unit ids")
    unique_clusters = sorted_data.get_unit_ids()

    # Sampling rate
    # print('Finding sampling rate')
    # mb.Neuron.fs = sorted_data.get_sampling_frequency()
    fs = sorted_data.get_sampling_frequency()
    if lverbose:
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

    end_time = rec_time / fs
    if lverbose:
        print("Start time ", start_time, " end time ", end_time)

    # Loop through unique clusters and make neuron list
    for unit_idx, unit in enumerate(unique_clusters):
        # ch_group = sorted_data.get_unit_property(unit, "group")
        # print("Total i ", i, " unit ", unit)
        if unit_idx not in noflylist:
            # print("qual ", cluster_quals[i])
            if len(filt) == 0:
                # this is the unit number for indexing spike times
                # and unit properties
                sp_c = [unit]
                # these are spike times
                sp_t = sorted_data.get_unit_spike_train(unit)
                qual = 0
                # mean WF @ Fs of recording
                # nblocked mwf_list = \
                # nblocked  sorted_data.get_unit_property(unit, "template").T
                if lverbose:
                    print(f'unit_idx {unit_idx}', flush=True)
                    print(f't_ch_size {t_ch_size}', flush=True)
                    print(f'peak_channels {peak_channels[unit_idx]}')
                    print(f'unit {unit}')
                    print(f'sh wfs[unit_idx][:, :, :] {wfs[unit_idx].shape}')
                mwf_list =\
                    np.mean(wfs[unit_idx][:, :, :],
                            axis=0)

                max_channel = peak_channels[unit_idx]
                #  try:
                # print("Skipped i ", i, " unit ", unit)
                # nblocked       print("unit_idx ", unit_idx, " unit ", unit,
                # nblocked             " max_channel : ", tmp_max_channel,
                # nblocked             " ", max_channel)
                if lverbose:
                    print("unit_idx ", unit_idx, " unit ", unit,
                          " max_channel : ", max_channel)
                # mwf = [row[tmp_max_channel] for row in mwf_list]
                # mwf =\
                #     np.mean(wfs[unit_idx][:, :, peak_channels[unit_idx]],
                #             axis=0)
                tmp_ch_num = \
                    ntk.get_tetrodechannelnum_from_channel(max_channel,
                                                           t_ch_size)
                if lverbose:
                    print(
                        f'max_channel {max_channel} t_ch_size {t_ch_size}'
                        f'ntk.get_tetrodechannelnum_from_channel {tmp_ch_num}'
                    )
                mwf = np.mean(
                    wfs[unit_idx][:, :, tmp_ch_num],
                    axis=0
                )
                del tmp_ch_num

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

                        # tmp_mean_amps = np.mean(np.asarray(amps[unit_idx]))
                        tmp_mean_amps = \
                            np.mean(np.abs(amps[amps_keys[unit_idx]]))
                        if tmp_mean_amps >= min_amps:
                            if lverbose:
                                print(f'unit_idx {unit_idx} unit {unit}')
                                print(f'amps_keys {amps_keys}')
                                print(f'len wf_b {len(wf_b)}')
                                print(f'len wf_e {len(wf_e)}')
                                print("mean amps ", unit_idx,
                                      " ", tmp_mean_amps,
                                      flush=True)
                            if ((wf_b is not None) and (wf_e is not None)):
                                n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                                         mwfs, [max_channel],
                                         fs=fs,
                                         start_time=start_time,
                                         end_time=end_time,
                                         mwft=mwf_list,
                                         rstart_time=str(file_datetime_list
                                                         [0]),
                                         rend_time=str(file_datetime_list[1]),
                                         estart_time=int(ecube_time_list
                                                         [0]),
                                         eend_time=int(ecube_time_list
                                                       [1]),
                                         sp_amp=np.abs(
                                                amps[amps_keys[unit_idx]]
                                                ),
                                         wf_b=np.asarray(wf_b[unit_idx]).T,
                                         wf_e=np.asarray(wf_e[unit_idx]).T,
                                         sex=sex, birthday=birthday,
                                         species=species,
                                         animal_name=animal_name,
                                         region_loc=region_loc,
                                         genotype=genotype,
                                         expt_cond=expt_cond))
                            #
                            # nblocked elif (wf_b is None):
                            # nblocked     n.append(mb.Neuron(sp_c, sp_t, qual,
                            # nblocked       mwf,
                            # nblocked       mwfs, max_channel,
                            # nblocked       fs=fs,
                            # nblocked       start_time=start_time,
                            # nblocked       end_time=end_time,
                            # nblocked       mwft=mwf_list,
                            # nblocked       rstart_time=str(file_datetime_list
                            # nblocked                       [0]),
                            # nblocked    rend_time=str(file_datetime_list[1]),
                            # nblocked       estart_time=int(ecube_time_list
                            # nblocked                       [0]),
                            # nblocked       eend_time=int(ecube_time_list
                            # nblocked                     [1]),
                            # nblocked       sp_amp=amps[unit_idx],
                            # nblocked       sex=sex, birthday=birthday,
                            # nblocked       species=species,
                            # nblocked       animal_name=animal_name,
                            # nblocked       region_loc=region_loc,
                            # nblocked       genotype=genotype,
                            # nblocked       expt_cond=expt_cond))

                        else:
                            print("Not added mean amps ", unit_idx, " ",
                                  tmp_mean_amps, flush=True)

                    # nblocked elif (amps is None):
                    # nblocked     n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                    # nblocked          mwfs, [max_channel],
                    # nblocked          fs=fs,
                    # nblocked          start_time=start_time,
                    # nblocked          end_time=end_time,
                    # nblocked          mwft=mwf_list,
                    # nblocked          rstart_time=str(file_datetime_list[0]),
                    # nblocked          rend_time=str(file_datetime_list[1]),
                    # nblocked          estart_time=int(ecube_time_list[0]),
                    # nblocked          eend_time=int(ecube_time_list[1]),
                    # nblocked          sex=sex, birthday=birthday,
                    # nblocked          species=species,
                    # nblocked          animal_name=animal_name,
                    # nblocked          region_loc=region_loc,
                    # nblocked          genotype=genotype,
                    # nblocked          expt_cond=expt_cond))
                elif ((len(file_datetime_list) == 2) and
                        (len(ecube_time_list) == 0)):
                    n.append(mb.Neuron(sp_c, sp_t, qual, mwf,
                             mwfs, [max_channel],
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
                             mwfs, [max_channel],
                             fs=fs,
                             start_time=start_time, end_time=end_time,
                             mwft=mwf_list,
                             sex=sex, birthday=birthday, species=species,
                             animal_name=animal_name,
                             region_loc=region_loc))
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


def mbt_sorter_interface_out(
     clust_out_dir,
     model_file='/media/HlabShare/models/xgboost_autoqual_prob',
     sex=None, birthday=None, species=None,
     animal_name=None,
     region_loc=None,
     genotype=None,
     expt_cond=None):

    '''
    Function loads spikeinterface output to neuron

    mbt_sorter_interface_out('spikeinterface_output_directory',
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
    mbt_sorter_interface_out('/home/kbn/co/',
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

    rl = glob.glob('*_amplitudes_group0.npy')[0]
    amps = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_waveforms_list_group0.npy')[0]
    wfs = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_peak_channel_group0.npy')[0]
    peak_channels = mb.load_np(rl, lpickle=True)
    peak_channels = list(peak_channels[()].values())

    rl = glob.glob('*_b_waveforms_group0.npy')[0]
    wf_b = mb.load_np(rl, lpickle=True)

    rl = glob.glob('*_e_waveforms_group0.npy')[0]
    wf_e = mb.load_np(rl, lpickle=True)

    try:
        rl = glob.glob('*_t_ch_size0.npy')[0]
        t_ch_size = int(mb.load_np(rl, lpickle=True))
    except Exception as e:
        print("Error: ", e)
        t_ch_size = 4
        print("Setting t_ch_size to 4")

    # cells = mb.siout(sorted_data, noflylist, rec_time,
    #                  file_datetime_list, ecube_time_list,
    #                  amps=amps,
    #                  wf_b=wf_b, wf_e=wf_e,
    #                  filt=filt,
    #                  t_ch_size=t_ch_size,
    #                  model_file=model_file,
    #                  sex=sex, birthday=birthday, species=species,
    #                  animal_name=animal_name,
    #                  region_loc=region_loc,
    #                  genotype=genotype,
    #                  expt_cond=expt_cond)
    cells = ms5out(sorted_data, noflylist, rec_time,
                   file_datetime_list, ecube_time_list,
                   wfs=wfs,
                   peak_channels=peak_channels,
                   amps=amps,
                   wf_b=wf_b, wf_e=wf_e,
                   filt=filt,
                   t_ch_size=t_ch_size,
                   model_file=model_file,
                   sex=sex, birthday=birthday, species=species,
                   animal_name=animal_name,
                   region_loc=region_loc,
                   genotype=genotype,
                   expt_cond=expt_cond,
                   lskipautoqual=None,
                   min_amps=15)

    # wfs=wfs,
    # peak_channels=peak_channel,

    return cells
