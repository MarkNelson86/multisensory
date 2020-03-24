from tdt import read_block, epoc_filter
import numpy as np
import pandas as pd
from os import listdir
import glob
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt

# to run: exec(open('TDT_py_lfp.py').read())
# input = '/Users/cat/ownCloud/Cats Projects/TANKS/'
input = '/home/cat/Insync/catherine.boucher2@mail.mcgill.ca/OneDrive Biz/TDT tanks 2019/'
tanks = [f for f in listdir(input) if 'Tanks' in f]
blocks = [filename for filename in glob.iglob(input + '**/', recursive=True) if 'Block-' in filename if
          not 'sort' in filename]
#blocks = blocks[:25]  # testing
channels = [1,2,3,4,5,6,8,10,11,12,14,16] #[0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 15]
#channels = channels[:2]  # testing
STREAM_STORE = 'Wave'
ARTIFACT = np.inf
TRANGE = [-1, 0.5]


def extract_lfp(data, ARTIFACT):
    #from TDT examples. removes artifacts and returns filtered lfp
    art1 = np.array([np.any(x > ARTIFACT)
                     for x in data.streams[STREAM_STORE].filtered], dtype=bool)

    art2 = np.array([np.any(x < -ARTIFACT)
                     for x in data.streams[STREAM_STORE].filtered], dtype=bool)

    good = np.logical_not(art1) & np.logical_not(art2)

    num_artifacts = np.sum(np.logical_not(good))
    if num_artifacts == len(art1):
        raise Exception('all waveforms rejected as artifacts')

    data.streams[STREAM_STORE].filtered = [data.streams[STREAM_STORE].filtered[i] for i in range(len(good)) if good[i]]

    min_length = np.min([len(x) for x in data.streams[STREAM_STORE].filtered])
    data.streams[STREAM_STORE].filtered = [x[:min_length] for x in data.streams[STREAM_STORE].filtered]
    #min_width = np.min([len(x[0]) for x in data.streams[STREAM_STORE].filtered])
    #data.streams[STREAM_STORE].filtered = [x[0][:min_width] for x in data.streams[STREAM_STORE].filtered] # to double check

    signals = np.vstack(data.streams[STREAM_STORE].filtered) #where the code would break from unequal array dimensions
    return signals


def str_blocknames(block, PARAM):
    #converts block numbers to strings
    if (block == 7 and PARAM == 'Nois') or (block == 8 and PARAM == 'TLvl') or (block == 9 and PARAM == 'FDur'):
        bnames = ['freq tuning', 'FRA', 'best tone', 'flash', 'AV', 'freq tuning2', 'freq corr', 'freq corr AV',
                  'freq corr flash']
    else:
        bnames = ['freq tuning', 'FRA', 'best tone', 'flash', 'AV', 'freq tuning2', 'check', 'grating', 'moving dots',
                  'radialcheck',
                  'plaid1', 'plaid2', 'drifting dots', 'freq corr', 'freq corr AV', 'freq corr flash']
    bnames = bnames[block - 1]
    return bnames


def get_data(blocks, channels):
    # extracts data from TDT tank and stores it in a MultiIndex pandas dataframe
    dframe = []
    for i in range(len(blocks)):
        for c in channels:
            print(i)
            print(c)
            if i == 25 or i == 371 or i == 382 or i == 436 or i == 443 or i == 444 or i == 466 or i == 666: # skipping a weird block
                continue
            data = read_block(blocks[i], evtype=['streams', 'epocs'], channel=c)

            nblock = int(data.info.blockname.partition("-")[2])  # gets block #
            nepocs = pd.DataFrame(data.epocs).columns.values.tolist()
            if 'PROT' in nepocs: nepocs.remove('PROT')
            # df = pd.DataFrame([nepocs]).T
            # df.columns = [nblock]
            # dframe.merge(df)

            epoc_list = ['TLvl', 'FDur', 'vdur', 'Nois']  # use epoc_list to automate PARAM1 selection
            PARAM1 = next(iter(set(nepocs).intersection(epoc_list)))
            number_epocs = len(nepocs)

            filtered_data = epoc_filter(data, PARAM1, t=TRANGE)
            all_signals = pd.DataFrame(data=extract_lfp(filtered_data, ARTIFACT))
            tankname = filtered_data.info.tankpath.split()[4].partition("/")[2].partition("_")
            all_signals['catname'] = tankname[0]
            all_signals['tanknumber'] = tankname[2]
            all_signals['channel'] = c
            all_signals['blockname'] = str_blocknames(nblock, PARAM1)
            for e in range(len(nepocs)):
                all_signals[str(nepocs[e])] = data.epocs[nepocs[e]].data
            col_names = [x for x in all_signals.columns.values.tolist() if not isinstance(x, int)]
            # df['column_new_1'], df['column_new_2'], df['column_new_3'] = [np.nan, 'dogs', 3]
            id_signals = all_signals.set_index(col_names)*1e6 #index df and data in microvolts
            dframe.append(id_signals)
    df = pd.concat(dframe, ignore_index=False)
    return df
