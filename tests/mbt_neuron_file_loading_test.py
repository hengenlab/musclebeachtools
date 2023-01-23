import unittest
import os
import os.path as op
import numpy as np
import musclebeachtools as mbt


class Test_mbt_load_neuron(unittest.TestCase):
    print(os.getcwd())
    workdir = \
        '/home/runner/work/musclebeachtools_hlab/musclebeachtools_hlab/tests/'
    if op.exists(workdir):
        os.chdir(workdir)
    expected_output_st = np.loadtxt('spike_time.csv', delimiter=',')
    n = np.load('test_neuron.npy', allow_pickle=True)
    print(dir(mbt))
    output = n[0].spike_time
    print(n)
    print(expected_output_st)
    print(output)
    output_fr, _ = \
        n[0].plotFR(binsz=3600, start=False, end=False, lplot=0, lonoff=1)
    expected_output_fr = np.loadtxt('fr_rate.csv', delimiter=',')
    _, _, output_isi = \
        n[0].isi_hist(start=False, end=False, isi_thresh=0.1, nbins=101,
                      lplot=0, lonoff=1)
    output_isi = output_isi[0]
    expected_output_isi = np.loadtxt('isi_hist.csv', delimiter=',')

    output_wf = n[0].waveform
    expected_output_wf = np.loadtxt('waveform.csv', delimiter=',')
    expected_output_wf = expected_output_wf.tolist()

    output_fs = n[0].fs
    expected_output_fs = np.loadtxt('fs.csv', delimiter=',')

    output_stsec = n[0].spike_time_sec
    expected_output_stsec = np.loadtxt('spike_time_sec.csv', delimiter=',')

    output_gender = n[0].sex
    expected_output_gender = np.loadtxt('animal_gender.csv',
                                        delimiter=',', dtype='str')

    def test_checkspk(self):
        print(self.expected_output_st)
        print(type(self.expected_output_st))
        print(self.output)
        print(type(self.output))
        msg = 'spike times are different'
        self.assertEqual(self.expected_output_st.tolist(),
                         self.output.tolist(), msg)

    def test_checkfr(self):
        print(self.expected_output_fr)
        print(type(self.expected_output_fr))
        print(self.output_fr)
        print(type(self.output_fr))
        msg = 'firing rates are different'
        self.assertEqual(self.expected_output_fr.tolist(),
                         self.output_fr.tolist(), msg)

    def test_checkisi(self):
        print(self.expected_output_isi)
        print(type(self.expected_output_isi))
        print(self.output_isi)
        print(type(self.output_isi))
        msg = 'isi hists are different'
        self.assertEqual(self.expected_output_isi.tolist(),
                         self.output_isi.tolist(), msg)

    def test_checkwf(self):
        print("wfwfwfwf ", self.expected_output_wf)
        print(type(self.expected_output_wf))
        print("wfwfwfwf ", self.output_wf)
        print(type(self.output_wf))
        msg = 'wf hists are different'
        for i, j in zip(self.expected_output_wf, self.output_wf):
            self.assertAlmostEqual(i, j,
                                   places=2,
                                   msg=msg)

    def test_checkfs(self):
        print(self.expected_output_fs)
        print(type(self.expected_output_fs))
        print(self.output_fs)
        print(type(self.output_fs))
        msg = 'fs are different'
        self.assertEqual(self.expected_output_fs,
                         self.output_fs, msg)

    def test_checkspksec(self):
        print(self.expected_output_stsec)
        print(type(self.expected_output_stsec))
        print(self.output_stsec)
        print(type(self.output_stsec))
        msg = 'spike times sec are different'
        self.assertEqual(self.expected_output_stsec.tolist(),
                         self.output_stsec.tolist(), msg)

    def test_gender(self):
        print(self.expected_output_gender)
        print(type(self.expected_output_gender))
        print(self.output_gender)
        print(type(self.output_gender))
        msg = 'gender is different'
        self.assertEqual(self.expected_output_gender,
                         self.output_gender, msg)
