"""Output Manager: Stores and writes macroscopic pore-network data 
such as saturation and relative permeability
"""

import cPickle as pickle

import numpy as np


class OutputReader(object):
    @staticmethod
    def convert_input_data_to_np(data):
        for i in data:
            data[i] = np.array(data[i])
        
    def read_output_data(self, filename='macro_data.pkl'):
        input_file = open(filename, 'rb')
        input_data = pickle.load(input_file)
        
        self.convert_input_data_to_np(input_data)
        return input_data