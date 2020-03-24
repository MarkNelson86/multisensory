
# This file takes .mat struct with data and epochs (two seperate columns) and converts it into a readable pandas file

from cats_reader import CatsReader

block_to_column_header = {
    2: ['duration', 'frequency', 'attenuation'],
    3: ['duration'],
    4: ['delay'],
} #dictionary

all_keys = ['cat', 'penetration', 'channel', 'block', 'unit', 'delay', 'duration', 'frequency', 'attenuation']

reader = CatsReader(all_keys, block_to_column_header)
results = reader.read_cats_file('data/cats.mat')
results.to_hdf('results_long.h5', key='cats', mode='w', data_columns=True, complevel=9)

