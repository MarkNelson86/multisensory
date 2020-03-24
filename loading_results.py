import numpy as np
import h5py
import pandas as pd
import scipy.stats as ss

df = pd.read_hdf('results2.h5')
auditory = df.xs((2),level=('block'))
auditory = auditory.divide(0.001) # turn into spikes/seconds
#auditory_stats = auditory.T.set_index(pd.MultiIndex.from_tuples(tuple(map(lambda column_index: ('pre' if column_index < 1500 else 'post', 1510 < column_index < 1575), auditory.columns)))).T#.groupby(level=(0, 1, 2, 3)).mean()
pre_a_stats = auditory.loc[:,0:1499].mean(axis=1)
post_a_stats = auditory.loc[:,1509:1574].mean(axis=1)
a_stats=pd.DataFrame(dict(pre = pre, post = post))
a_result = a_stats.groupby(level=(0, 1, 2, 3)).apply(ss.ranksums(a_stats[pre], a_stats[post]))


visual = df.xs((3),level=('block'))
visual = visual.divide(0.001)
pre_v_stats = visual.loc[:,0:1499].mean(axis=1)
post_v_stats = visual.loc[:,1509:1574].mean(axis=1)
v_stats=pd.DataFrame(dict(pre = pre, post = post))
v_result = v_stats.groupby(level=(0, 1, 2, 3))