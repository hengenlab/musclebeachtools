import glob
import os
import os.path as op
import pickle
# import numpy as np
# import neuraltoolkit as ntk
import musclebeachtools as mb


def get_group_value(channel, group_size):
    '''
    get_group_value(channel, group_size)

    Calculates the group value of a channel
     by finding the remainder when divided by group_size.

    channel (int): The channel number.
    group_size (int): The number of groups.

    Group size must not be zero or negative
    channel must be non-negative

    Returns
        int: The group value (0 to group_size - 1).
'''

    # raise errors
    if channel < 0:
        raise ValueError("Channel must be non-negative")
    if group_size <= 0:
        raise ValueError("Group size must not be zero or negative")

    return int(channel % group_size)


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

    cells = mb.siout(sorted_data, noflylist, rec_time,
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
    # def siout(sorted_data, noflylist, rec_time,
    #           file_datetime_list, ecube_time_list,
    #           wfs=None,
    #           peak_channels=None,
    #           amps=None,
    #           wf_b=None, wf_e=None,
    #           filt=None,
    #           t_ch_size=None,
    #           model_file='/media/HlabShare/models/xgboost_autoqual_prob',
    #           sex=None, birthday=None, species=None,
    #           animal_name=None,
    #           region_loc=None,
    #           genotype=None,
    #           expt_cond=None,
    #           lskipautoqual=None,
    #           min_amps=10)

    return cells
