import unittest
import musclebeachtools as mbt


class Test_mbt_funcs(unittest.TestCase):

    def test_check_sametetrode_neurons(self):
        channel1_list = [0, 4, 63]
        channel2_list = [0, 4, 63]
        expected_output = True
        for channel1, channel2 in zip(channel1_list,
                                      channel2_list):
            output_sametetrode_neurons = \
                mbt.check_sametetrode_neurons(channel1, channel2,
                                              ch_grp_size=4,
                                              lverbose=1)
            print(expected_output)
            print(type(expected_output))
            print(output_sametetrode_neurons)
            print(type(output_sametetrode_neurons))
            msg = 'channels are different'
            self.assertEqual(expected_output,
                             output_sametetrode_neurons, msg)

        channel1_list = [0, 4, 59]
        channel2_list = [4, 8, 63]
        expected_output = False
        for channel1, channel2 in zip(channel1_list,
                                      channel2_list):
            output_sametetrode_neurons = \
                mbt.check_sametetrode_neurons(channel1, channel2,
                                              ch_grp_size=4,
                                              lverbose=1)
            print(expected_output)
            print(type(expected_output))
            print(output_sametetrode_neurons)
            print(type(output_sametetrode_neurons))
            msg = 'channels are different'
            self.assertEqual(expected_output,
                             output_sametetrode_neurons, msg)
