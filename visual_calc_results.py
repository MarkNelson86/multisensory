import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import datetime

def group_to_ranksum(group_df):
    to_return =  pd.Series(
        ss.ranksums(
            group_df.loc[:,'pre'].values,
            group_df.loc[:,'post'].values
        ),
        index=['statistic', 'p_value'],
    )
    return to_return

def analyze_response(df):
	#df = df.divide(0.001)
	pre_stats = df.loc[:, -1.5:-0.499].mean(axis=1)
	code = df[:1].index.codes[2][0]
	block = df[:1].index.levels[2][code]
	post_stats = pd.DataFrame([])
	print('analyzing block: ', block)
	if block == 4:
		post_stats = df.loc[:, 0.25:0.499].mean(axis=1)
	else:
		post_stats = df.loc[:, 0.25:0.999].mean(axis=1)
		return post_stats
	all_stats = pd.DataFrame(dict(pre = pre_stats, post = post_stats))
	all_results = all_stats.groupby(level=(0,1,2,3,4)).apply(group_to_ranksum)# apply creates df's for groups and sends it into supplied function
	all_results['baseline'] = pre_stats.groupby(level=(0,1,2,3,4)).mean()
	return all_results

def analyze_responses(df):
	return df.groupby(level=2).apply(analyze_response).droplevel(0)

def extract_misc_levels_to_response_stats(df):
	# We drop_duplicates to remove any rows with redundant index information
	stats = df.index.to_frame().drop_duplicates()
	return stats

def plot_av(response_stats):
	for index in [3,6,7,8,9,10,11,12]:
		plt.figure(index)
		hist_av = hist.groupby(level=(2)).mean().T
		hist_av.iloc[:,index].plot()
		plt.show()

source_df = pd.DataFrame([])
try:
	print(datetime.datetime.utcnow(), "Reading fixed file if exists...")
	source_df = pd.read_hdf('results_visual.h5')
	print(datetime.datetime.utcnow(), "Successfully read fixed file!")
except FileNotFoundError:
	print(datetime.datetime.utcnow(), "Fixed file doesnt exist. Reading long HDF...")
	df = pd.read_hdf('results_visual.h5')
	print(datetime.datetime.utcnow(), "Cleaning NaNs")
	df.set_index(['cat', 'penetration', 'block', 'channel', 'unit', 'duration', 'flip', 'direction', 'spatial_freq','angle', 'angle2'], inplace=True)
	df.to_hdf('results_visual.h5', key='cats', mode='w', data_columns=True, complevel=9)

# To run analysis and plot
print(datetime.datetime.utcnow(), "Analyzing...")
# df = source_df[:10000]
df = source_df
stat = extract_misc_levels_to_response_stats(df)
response_stats = analyze_responses(df).join(stat)
plot_av(df)
#response_stats = extract_misc_levels_to_response_stats(df)
# adding more columns, based off the response_stats indices, to response_stats from wilcoxon ranksum test (statistic and p value)
#response_stats = analyze_responses(df).join(response_stats)


# to run... exec(open('visual_calc_results.py').read())