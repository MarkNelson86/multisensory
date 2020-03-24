
# This file takes .mat struct with data and epochs (two seperate columns) and converts it into a readable pandas file

from cats_reader import CatsReader
import pandas as pd

block_to_column_header = {
    3: ['duration'],
    6: ['duration','flip'],
    7: ['duration', 'spatial_freq', 'angle'],
    8: ['duration', 'flip', 'direction'],
    9: ['duration', 'flip'],
    10: ['duration', 'spatial_freq', 'angle', 'angle2'],
    11: ['duration', 'spatial_freq', 'angle', 'angle2'],
    12: ['duration', 'flip', 'direction'],

} #dictionary

all_keys = ['cat', 'penetration', 'channel', 'block', 'unit', 'duration', 'flip', 'direction', 'spatial_freq',
            'angle', 'angle2']

reader = CatsReader(all_keys, block_to_column_header)
results = reader.read_cats_file('data/visual.mat')
# only keeping responsive channels from the 2 electrodes
df_froggy_rascal = results.loc[([1,2], slice(None), slice(None), [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]), :]
df_spain_france = results.loc[([0,3], slice(None), slice(None), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), :]
df_cleaned = pd.concat([df_froggy_rascal, df_spain_france])
df_cleaned.to_hdf('results_visual.h5', key='cats', mode='w', data_columns=True, complevel=9)

