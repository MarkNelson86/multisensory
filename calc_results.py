import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from os import makedirs
import dexplot as dxp
from rpy2.robjects.packages import importr
stats = importr('stats')
from statsmodels.stats.multitest import multipletests

R_stats = importr('stats')

# Key is (has
# (has_auditory_response, has_visual_response, is_any_integrative)
CLASSIFICATION_DICT = {
	(True, True, True): 'Bimodal integrative',
	(True, True, False): 'Bimodal non-integrative',
	(True, False, True): 'Auditory subthreshold',
	(True, False, False): 'Auditory unimodal',
	(False, True, True): 'Visual subthreshold',
	(False, True, False): 'Visual unimodal',
	(False, False, True): 'Bimodal subthreshold',
	(False, False, False): 'Unresponsive',
}

UNRESPONSIVE_KEYS = [(1, 3, 4, 0),
	(1, 3, 5, 0),
	(1, 3, 9, 0),
	(1, 5, 3, 0),
	(1, 5, 12, 0),
	(1, 9, 9, 0),
	(1, 10, 6, 0),
	(1, 10, 14, 0),
	(1, 11, 15, 0),
	(1, 12, 2, 0),
	(1, 12, 4, 0),
	(1, 12, 5, 0),
	(1, 12, 6, 0),
	(1, 13, 4, 0),
	(1, 14, 4, 0),
	(1, 14, 6, 0),
	(1, 15, 0, 0),
	(1, 15, 6, 0),
	(1, 22, 0, 0),
	(1, 22, 7, 0),
	(1, 22, 11, 0),
	(1, 22, 13, 0),
	(1, 25, 2, 0),
	(1, 25, 3, 0),
	(1, 25, 6, 0),
	(1, 25, 8, 0),
	(1, 25, 9, 0),
	(1, 25, 10, 1),
	(1, 25, 13, 0),
	(1, 26, 1, 0),
	(1, 26, 7, 0),
	(1, 26, 9, 0),
	(1, 26, 11, 0),
	(1, 27, 1, 0),
	(1, 27, 3, 0),
	(1, 27, 8, 1),
	(1, 28, 15, 0),
	(1, 30, 9, 0),
	(1, 31, 7, 0),
	(1, 31, 8, 1),
	(1, 31, 9, 0),
	(1, 31, 11, 0),
	(1, 31, 13, 0),
	(1, 31, 15, 0),
	(2, 4, 15, 0),
	(2, 4, 15, 1),
	(2, 8, 1, 0),
	(2, 8, 2, 0),
	(2, 8, 3, 0),
	(2, 8, 4, 0),
	(2, 8, 6, 0),
	(2, 8, 7, 0),
	(2, 8, 9, 0),
	(2, 8, 10, 1),
	(2, 8, 11, 0),
	(2, 8, 15, 0),
	(2, 9, 5, 0),
	(2, 9, 7, 0),
	(2, 13, 11, 0),
	(2, 15, 3, 0),
	(2, 15, 4, 0),
	(2, 15, 5, 0),
	(2, 15, 7, 0),
	(2, 15, 9, 0),
	(2, 15, 11, 0),
	(2, 15, 15, 0),
	(2, 16, 2, 0),
	(2, 16, 4, 0),
	(2, 16, 10, 1),
	(2, 16, 15, 0),
	(2, 17, 2, 0),
	(2, 17, 3, 0),
	(2, 17, 4, 0),
	(2, 17, 5, 0),
	(2, 17, 6, 0),
	(2, 17, 7, 0),
	(2, 17, 9, 0),
	(2, 17, 11, 0),
	(2, 17, 15, 0),
	(2, 18, 4, 0),
	(2, 18, 6, 0),
	(2, 18, 9, 0),
	(2, 18, 11, 0),
	(2, 18, 13, 0),
	(2, 18, 15, 0),
	(2, 20, 0, 0),
	(2, 20, 1, 0),
	(2, 20, 2, 0),
	(2, 20, 3, 0),
	(2, 20, 4, 0),
	(2, 20, 6, 0),
	(2, 20, 10, 1),
	(2, 20, 11, 0),
	(2, 23, 4, 0),
	(2, 23, 6, 0),
	(2, 23, 7, 0),
	(2, 23, 9, 0),
	(2, 23, 10, 1),
	(2, 23, 11, 0),
	(2, 23, 13, 0),
	(2, 23, 15, 0),
	(2, 24, 0, 0),
	(2, 24, 5, 0),
	(2, 24, 6, 0),
	(2, 24, 15, 0),
	(2, 26, 2, 0),
	(2, 27, 6, 0),
	(2, 27, 10, 0),
	(2, 28, 2, 0),
	(2, 28, 4, 0),
	(2, 28, 6, 0),
	(2, 28, 11, 0),
	(2, 28, 15, 0),
	(2, 31, 0, 0),
	(2, 31, 6, 0),
	(2, 31, 7, 0),
	(2, 31, 9, 0),
	(2, 31, 15, 0),
	(2, 32, 7, 0),
	(2, 33, 2, 0),
	(2, 33, 3, 0),
	(2, 33, 11, 0),
	(2, 34, 0, 0),
	(2, 34, 1, 0),
	(2, 34, 10, 0),
	(2, 34, 11, 0),
	(2, 34, 13, 0),
	(2, 34, 14, 0),
	(2, 35, 1, 0),
	(2, 35, 2, 0),
	(2, 35, 9, 0),
	(2, 36, 4, 0),
	(2, 37, 0, 0),
	(2, 37, 2, 0),
	(2, 37, 3, 0),
	(2, 37, 7, 0),
	(2, 37, 11, 0),
	(2, 38, 3, 1),
	(2, 38, 7, 0),
	(2, 39, 4, 0),
	(2, 39, 6, 0),
	(2, 39, 9, 0),
	(2, 44, 1, 0),
	(2, 44, 4, 0),
	(2, 44, 6, 0),
	(2, 44, 9, 0),
	(2, 44, 15, 0),
]
	

Z_SCORE_01 = -2.575829
Z_SCORE_001 = -3.290527
Z_SCORE_05 = -1.96
P_VALUE = 0.05
Z_SCORE = Z_SCORE_05
P_VALUE_INTEGRATIVE = 0.05
Z_SCORE_INTEGRATIVE = -Z_SCORE_05

def group_to_ranksum(group_df):
    to_return =  pd.Series(
        ss.ranksums(
            group_df.loc[:,'pre'].values,
            group_df.loc[:,'post'].values
        ),
        index=['statistic', 'p_value'],
    )
    return to_return

def pre_post_to_ranksum(av_post, auditory):
	 return pd.DataFrame(
        ss.ranksums(
        	auditory.values,
        	av_post.values,
        ),
        index=['statistic', 'p_value'],
    ).T.reset_index(drop=True)

def delay_fix(df):
	code = df[:1].index.codes[5][0]
	delay = df[:1].index.levels[5][code]
	maxtime = delay + 2200
	return df.iloc[:, int(delay):int(maxtime)].T.set_index(pd.Index(np.arange(-1.5, .7, 0.001))).T

def analyze_response(df):
	if df.empty:
		return df
	#df = df.divide(0.001)
	pre_stats = df.loc[:, -1.5:-0.499].mean(axis=1)
	code = df[:1].index.codes[2][0]
	block = df[:1].index.levels[2][code]
	post_stats = pd.DataFrame([])
	print('analyzing block: ', block)
	if block == 2:
		post_stats = df.loc[:, 0.009:0.074].mean(axis=1)
	elif block == 3:
		post_stats = df.loc[:, 0.25:0.499].mean(axis=1)
	else:
		no_stats_results = pd.DataFrame(index=df.index.drop_duplicates())
		no_stats_results['statistic'] = np.nan
		no_stats_results['p_value'] = np.nan
		# Potential for bug check if groupby is merging in correctly
		no_stats_results['baseline'] = pre_stats.groupby(level=(0,1,2,3,4,5)).mean()
		return no_stats_results
	all_stats = pd.DataFrame(dict(pre = pre_stats, post = post_stats))
	all_results = all_stats.groupby(level=(0,1,2,3,4,5)).apply(group_to_ranksum)# apply creates df's for groups and sends it into supplied function
	all_results['baseline'] = pre_stats.groupby(level=(0,1,2,3,4,5)).mean()
	return all_results

def analyze_responses(df):
	return df.groupby(level=2).apply(analyze_response).droplevel(0)

DEFAULT_RETURN = (np.nan, np.nan, np.nan)
UNIT_FIELD_NAMES = ['category', 'enhancement_index', 'int_delay_p_value', 'int_delay_statistic']

def create_error_frame(msg):
	return pd.DataFrame((
			msg,
			*DEFAULT_RETURN,
		),
		index=UNIT_FIELD_NAMES
	).T.reset_index(drop=True)

def shift_av(delay_df):
	code = delay_df[:1].index.codes[5][0]
	delay = delay_df[:1].index.levels[5][code] / 1000
	start = 0.25 - delay
	finish = 0.499 - delay
	return delay_df.loc[:, start:finish].T.reset_index(drop=True).T

def analyze_unit(unit_df, response_stats, psth):
	response_stats_for_unit = response_stats[response_stats.index.isin(unit_df.index)]
	all_auditory_response_stats_for_unit = response_stats_for_unit.xs(2, level=2)
	all_visual_response_stats_for_unit = response_stats_for_unit.xs(3, level=2)
	if response_stats_for_unit.index.get_level_values(2).contains(4):
		all_audiovisual_response_stats_for_unit = response_stats_for_unit.xs((4), level=(2))
	#else:
	#	return create_error_frame('Error: Missing av response stats')

	#if len(all_auditory_response_stats_for_unit.index) > 1:
	#	return create_error_frame('Error: Duplicated auditory stats found')

	#if len(all_visual_response_stats_for_unit.index) > 1:
	#	return create_error_frame('Error: Duplicated visual stats found')

	if unit_df.index.get_level_values(2).contains(2):
		auditory_response_window_df = unit_df.xs(2, level=2, drop_level=False).loc[:,0.009:0.074].mean(axis=1)
	#else:
	#	return create_error_frame('Error: Missing auditory block')

	if unit_df.index.get_level_values(2).contains(2):
		visual_response_window_df = unit_df.xs(3, level=2, drop_level=False).loc[:,0.25:0.499].mean(axis=1)
	#else:
	#	return create_error_frame('Error: Missing auditory block')

	if unit_df.index.get_level_values(2).contains(4):
		av_response_window_df = unit_df.xs(4, level=2, drop_level=False).loc[:,0.009:0.074].mean(axis=1)
	#else:
	#	return create_error_frame('Error: Missing av block')

	auditory_response_stats_for_unit = all_auditory_response_stats_for_unit.iloc[0]
	visual_response_stats_for_unit = all_visual_response_stats_for_unit.iloc[0]
	auditory_peak = auditory_response_stats_for_unit['peak_response']

	has_auditory_response = auditory_response_stats_for_unit['statistic'] < Z_SCORE
	has_visual_response = visual_response_stats_for_unit['statistic'] < Z_SCORE

	visual_peak = visual_response_stats_for_unit['peak_response']
	audiovisual_peak = all_audiovisual_response_stats_for_unit['peak_response']

	if auditory_peak >= visual_peak:	
		target_response_window_df = auditory_response_window_df
		enhancement_index = (((audiovisual_peak - auditory_peak) / (auditory_peak + audiovisual_peak) * 100).droplevel((0, 1, 2, 3))).rename('enhancement_index')
		additivity_index = (((audiovisual_peak - (auditory_peak + visual_peak)) / (auditory_peak + (audiovisual_peak + visual_peak)) * 100).droplevel((0, 1, 2, 3))).rename('additivity_index')
	else:
		target_response_window_df = visual_response_window_df
		av_response_window_df = unit_df.xs(4, level=2, drop_level=False).groupby(level=5).apply(shift_av).mean(axis=1)
		enhancement_index = (((audiovisual_peak - visual_peak) / (visual_peak + audiovisual_peak) * 100).droplevel((0, 1, 2, 3))).rename('enhancement_index')
		additivity_index = (((audiovisual_peak - (visual_peak + auditory_peak)) / (visual_peak + (audiovisual_peak + auditory_peak)) * 100).droplevel((0, 1, 2, 3))).rename('additivity_index')

	av_ranksum = av_response_window_df.groupby(level=5).apply(pre_post_to_ranksum, target_response_window_df).droplevel(1)
	av_ranksum['adjusted_p_value'] = multipletests(av_ranksum['p_value'], method='holm')[1] #multiple comparisons correction
	is_integrative = (av_ranksum['adjusted_p_value'] < P_VALUE_INTEGRATIVE).rename('is_integrative')
	if is_integrative.any():
		is_integrative.values[:] = True

	is_absent = not (enhancement_index.notna().any() or additivity_index.notna().any() or (av_ranksum['adjusted_p_value'] < 1.00000).any())

	classification_lookup_df = is_integrative.to_frame()
	classification_lookup_df['has_auditory_response'] = has_auditory_response
	classification_lookup_df['has_visual_response'] = has_visual_response
	classification_lookup_df['is_absent'] = is_absent

	def lookup_category(row):
		if (row['is_absent']):
			return 'Absent'
		boolean_key = (row['has_auditory_response'], row['has_visual_response'], row['is_integrative'])

		if boolean_key in CLASSIFICATION_DICT:
			return CLASSIFICATION_DICT[boolean_key]
		else:
			return 'Error: Unable to classify'

	def lookup_delay_category(row):
		if row['statistic'] < -Z_SCORE_INTEGRATIVE and row['adjusted_p_value'] < 0.05:
			return 'enhanced'
		elif row['statistic'] > Z_SCORE_INTEGRATIVE and row['adjusted_p_value'] < 0.05:
			return 'suppressed'
		else:
			return 'non-integrative'

	categories = classification_lookup_df.apply(lookup_category, axis=1).rename('category')
	delay_integrations = av_ranksum.apply(lookup_delay_category, axis=1).rename('delay_integration') # for boolean logic, suppress not needed

	return categories.to_frame() \
		.join(delay_integrations) \
		.join(enhancement_index) \
		.join(additivity_index) \
		.join(av_ranksum['statistic'].rename('int_delay_statistic')) \
		.join(av_ranksum['p_value'].rename('int_delay_p_value')) \
		.join(av_ranksum['adjusted_p_value'].rename('int_delay_adjusted_p_value'))

def analyze_units(df, response_stats, psth):
	# for each unit, get the response stats, remove blocks
	unit_stats = df.groupby(level=('cat','penetration','channel','unit')).apply(analyze_unit, response_stats, psth)
	return unit_stats

def extract_misc_levels_to_response_stats(df):
	# We drop_duplicates to remove any rows with redundant index information
	stats = df.index.to_frame().drop_duplicates().droplevel((6,7,8))[['duration','frequency','attenuation']]    
	return stats

def remove_misc_levels(df):
	return df.droplevel((6,7,8))

def analyze_psth(df, response_stats):
	to_return = df.groupby(level=(0,1,2,3,4,5)).mean().subtract(response_stats['baseline'], axis=0)  
	to_return = to_return.rolling(10, win_type='gaussian', axis=1).mean(std=1) 
	return to_return

def analyze_peak_responses(psth):
	return pd.DataFrame(psth.loc[:,0.009:0.499].max(axis=1).rename("peak_response"))

def plot_sample(df, lim, n, m, o, p):
	#figure 5
	# view and save individual psth from auditory, visual, or multisensory
	#multiple samples: params = [[df, 440, 1,13,7,0], [df,160,1,16,13,0], [df,440,1,13,13,0]]
	#results = [plot_sample(*args) for args in params]

	output = './plot_sample_' + str(n)+ '.'+str(m)+'.' + str(o) +'.'+ str(p)
	#makedirs(output)
	#n = cat
	#m = penetration
	#o = channel
	#p = unit
	for x in [0, 120]:
		if x == 0:
			for b in [2,3]:
				best = df.loc[:, -0.301:0.4].xs((n, m, b, o, p, x), level=('cat', 'penetration', 'block', 'channel', 'unit', 'delay'))
				best_tmp = best.reset_index(drop=True) / 1000
				best_tmp = best_tmp[best_tmp > 0] * best_tmp.columns
				to_eventplot = best_tmp.apply(lambda trial_row: trial_row.dropna().values, axis=1)

				if b == 3:
					plt.subplot(232, sharey=plt.subplot(231), sharex=plt.subplot(235))
					#plt.gca().set_title('Visual')
					plt.eventplot(to_eventplot)
					plt.margins(x=0)
					plt.ylabel('Trial Number')
					plt.axvline(x=0, color='k', linestyle='--')
				else:
					plt.subplot(231)
					#plt.gca().set_title('Auditory')
					plt.eventplot(to_eventplot)
					plt.margins(x=0)
					plt.ylabel('Trial Number')
					plt.axvline(x=0, color='k')
				plt.tick_params(
					axis='x',  # changes apply to the x-axis
					which='both',  # both major and minor ticks are affected
					bottom=False,  # ticks along the bottom edge are off
					top=False,  # ticks along the top edge are off
					labelbottom=False, labeltop=False)  # labels along the bottom edge are off
				if b == 3:
					plt.subplot(235, sharey=plt.subplot(234))
				else:
					plt.subplot(234)
				best.divide(0.001).mean().rolling(5, center=True, win_type='gaussian').mean(std=10).plot.line()  # ax2
				plt.ylim(0,lim)
				plt.ylabel('Spikes/s')
				plt.xlabel('Time (s)')
				if b == 3:
					plt.axvline(x=0, color='k', linestyle='--')
				else:
					plt.axvline(x=0, color='k')
		else:
			best = df.loc[:, -0.301:0.4].xs((n, m, 4, o, p, x),level=('cat', 'penetration', 'block', 'channel', 'unit', 'delay'))
			best_tmp = best.reset_index(drop=True) / 1000
			best_tmp = best_tmp[best_tmp > 0] * best_tmp.columns
			to_eventplot = best_tmp.apply(lambda trial_row: trial_row.dropna().values, axis=1)
			plt.subplot(233, sharey=plt.subplot(231), sharex=plt.subplot(236))
			#plt.gca().set_title('Audiovisual')
			plt.eventplot(to_eventplot)
			plt.margins(x=0)
			plt.ylabel('Trial Number')
			plt.axvline(x=-x/1000000, color='k', linestyle='--')
			plt.axvline(x=0, color='k')
			plt.tick_params(
				axis='x',  # changes apply to the x-axis
				which='both',  # both major and minor ticks are affected
				bottom=False,  # ticks along the bottom edge are off
				top=False,  # ticks along the top edge are off
				labelbottom=False, labeltop=False)  # labels along the bottom edge are off
			plt.subplot(236, sharey=plt.subplot(234))
			best.divide(0.001).mean().rolling(5, center=True, win_type='gaussian').mean(std=10).plot.line()  # ax2
			plt.ylim(0,lim)
			plt.ylabel('Spikes/s')
			plt.xlabel('Time (s)')
			plt.axvline(x=-x/1000, color='k', linestyle='--')
			plt.axvline(x=0, color='k')
	plt.savefig(output + '.png', dpi=300)
	plt.close()
	return output

def plot_soa(df, lim, n, m, o, p):
	#figure 9 1.9.7.0
	# view and save individual psth from auditory, visual, or multisensory
	#n = cat
	#m = penetration
	#o = channel
	#p = unit
	for x in [0, 1, 120, 240]:
		if x == 0:
			for b in [2,3]:
				best = df.loc[:, -0.301:0.4].xs((n, m, b, o, p, x), level=('cat', 'penetration', 'block', 'channel', 'unit', 'delay'))
				best_tmp = best.reset_index(drop=True) / 1000
				best_tmp = best_tmp[best_tmp > 0] * best_tmp.columns
				to_eventplot = best_tmp.apply(lambda trial_row: trial_row.dropna().values, axis=1)
				if b == 3:
					plt.subplot(152, sharey=plt.subplot(151))
					plt.gca().set_title('Visual')
				else:
					plt.subplot(151)
					plt.gca().set_title('Auditory')
				best.divide(0.001).mean().rolling(5, center=True, win_type='gaussian').mean(std=10).plot.line()  # ax2
				plt.ylim(0,lim)
				plt.ylabel('Spikes/s')
				plt.xlabel('Time (s)')
				if b == 3:
					plt.axvline(x=0, color='k', linestyle='--')
				else:
					plt.axvline(x=0, color='k')
		else:
			best = df.loc[:, -0.301:0.4].xs((n, m, 4, o, p, x),level=('cat', 'penetration', 'block', 'channel', 'unit', 'delay'))
			best_tmp = best.reset_index(drop=True) / 1000
			best_tmp = best_tmp[best_tmp > 0] * best_tmp.columns
			to_eventplot = best_tmp.apply(lambda trial_row: trial_row.dropna().values, axis=1)
			if x == 1:
				plt.subplot(153, sharey=plt.subplot(151))
				plt.gca().set_title('V1A')
			elif x == 120:
				plt.subplot(154, sharey=plt.subplot(151))
				plt.gca().set_title('V120A')
			elif x == 240:
				plt.subplot(155, sharey=plt.subplot(151))
				plt.gca().set_title('V240A')
			best.divide(0.001).mean().rolling(5, center=True, win_type='gaussian').mean(std=10).plot.line()  # ax2
			plt.ylim(0,lim)
			plt.ylabel('Spikes/s')
			plt.xlabel('Time (s)')
			plt.axvline(x=-x/1000, color='k', linestyle='--')
			plt.axvline(x=0, color='k')
	#plt.savefig(output + '/' + str(x) + '.png', dpi=300)
	#plt.close()

def plot_mean_population(psth):
	#figure 11
	plt.figure(2)
	psth.xs((2, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains("Bimodal")]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	psth.xs((2, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Auditory')]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	psth.xs((2, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Visual')]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	leg = plt.legend(['Bimodal', 'Auditory', 'Visual'])
	for legobj in leg.legendHandles:
		legobj.set_linewidth(5.0)
	plt.xlabel("Time (s)")
	plt.ylabel("Spikes/s")
	plt.title('Mean Auditory Response')
	plt.axvline(x=0, color='k')
	plt.ylim(-10, 110)
	plt.show()

	plt.figure(3)
	psth.xs((3, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains("Bimodal")]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	psth.xs((3, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Auditory')]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	psth.xs((3, 0), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Visual')]).index].loc[:,
	-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
	leg = plt.legend(['Bimodal', 'Auditory', 'Visual'])
	for legobj in leg.legendHandles:
		legobj.set_linewidth(5.0)
	plt.xlabel("Time (s)")
	plt.ylabel("Spikes/s")
	plt.title('Mean Visual Response')
	plt.axvline(x=0, color='k', linestyle='--')
	plt.ylim(-10, 110)
	plt.show()

	for x in [120]:
		plt.figure(x)
		psth.xs((4, x), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains("Bimodal")]).index].loc[:,
		-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
		psth.xs((4, x), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Auditory')]).index].loc[:,
		-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
		psth.xs((4, x), level=(2, 5)).loc[(unit_stats[unit_stats['category'].str.contains('Visual')]).index].loc[:,
		-0.301:0.5].mean().multiply(1000).interpolate('cubic').plot.line()
		leg = plt.legend(['Bimodal', 'Auditory', 'Visual'])
		for legobj in leg.legendHandles:
			legobj.set_linewidth(5.0)
		plt.xlabel("Time (s)")
		plt.ylabel("Spikes/s")
		plt.title('Mean Audiovisual Response (V{}A)'.format(x))
		plt.axvline(x=-x/1000, color='k', linestyle='--')
		plt.axvline(x=0, color='k')
		plt.ylim(-10, 110)
		plt.show()

def plot_auditory_binding(unit_stats):
	#figure 10
	filtered_units = unit_stats[~unit_stats['category'].isin(['Unresponsive'])].reset_index()
	filtered_units['subject'] = filtered_units.apply(lambda row: '{}_{}_{}_{}'.format(row['cat'], row['penetration'], row['channel'], row['unit']), axis=1)
	to_pivot = filtered_units[['delay_integration', 'enhancement_index', 'subject','delay']]
	to_pivot.loc[:, 'type'] = np.where(to_pivot['enhancement_index'] > 0, 'enhancement', 'suppression')
	to_plot = to_pivot.groupby(['delay', 'type']).aggregate(['mean', 'sem']).unstack()['enhancement_index']

	#to_pivot.to_csv("anova_fig10.csv")

	index = np.arange(len(to_plot.index.tolist()))
	label = pd.Series(to_plot.index.tolist())
	plt.bar(index, height=to_plot['mean','enhancement'], tick_label=label, yerr=to_plot['sem', 'enhancement'])
	plt.bar(index, height=to_plot['mean','suppression'],tick_label=label, yerr=to_plot['sem', 'suppression'])
	#plt.setp(plt xticks=index, xticklabels=label, ylabel='Index', xlabel= 'Stimulus Onset Asynchrony (ms)')
	plt.xlabel('Stimulus Onset Asynchrony (ms)')
	plt.ylabel('Enhancement Index')
	plt.show()

def plot_temporal_binding(unit_stats):
	fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
	fig.subplots_adjust(hspace=1)
	auditory_enhanced_summary = pd.concat([unit_stats([unit_stats['category'].str.contains("Subthreshold") & unit_stats['category'].str.contains("enhanced")]).groupby(level=4).mean()['enhancement_index'].rename('Mean'),  unit_stats[unit_stats['category'].str.contains("Subthreshold"and "enhanced")].groupby(level=4).sem()['enhancement_index'].rename('Standard Error')], axis=1)
	auditory_suppressed_summary = pd.concat([unit_stats[unit_stats['category'].str.contains("Subthreshold" and "suppressed")].groupby(level=4).mean()['enhancement_index'].rename('Mean'),  unit_stats[unit_stats['category'].str.contains("Subthreshold" and "suppressed")].groupby(level=4).sem()['enhancement_index'].rename('Standard Error')], axis=1)
	bimodal_enhanced_summary = pd.concat([unit_stats[unit_stats['category'].str.contains("Bimodal" and "enhanced")].groupby(level=4).mean()['enhancement_index'].rename('Mean'),unit_stats[unit_stats['category'].str.contains("Bimodal" and "enhanced")].groupby(level=4).sem()['enhancement_index'].rename('Standard Error')], axis=1).reindex([1.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0])
	bimodal_suppressed_summary = pd.concat([unit_stats[unit_stats['category'].str.contains("Bimodal" and "suppressed")].groupby(level=4).mean()['enhancement_index'].rename('Mean'),unit_stats[unit_stats['category'].str.contains("Bimodal" and "suppressed")].groupby(level=4).sem()['enhancement_index'].rename('Standard Error')], axis=1).reindex([1.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0])
	index = np.arange(len(auditory_enhanced_summary))
	label = pd.Series(auditory_enhanced_summary.index)
	ax1.title.set_text('Subthreshold')
	#ax2.title.set_text('Audiovisual Only')
	ax3.title.set_text('Bimodal')
	ax1.bar(index, height=auditory_enhanced_summary['Mean'], yerr=auditory_enhanced_summary['Standard Error'])
	ax1.bar(index, height=auditory_suppressed_summary['Mean'], yerr=auditory_suppressed_summary['Standard Error'])
	#ax2.bar(index, height=multisensory_enhanced_summary['Mean'], yerr=multisensory_enhanced_summary['Standard Error'])
	ax3.bar(index, height=bimodal_enhanced_summary['Mean'], yerr=bimodal_enhanced_summary['Standard Error'])
	ax3.bar(index, height=bimodal_suppressed_summary['Mean'], yerr=bimodal_suppressed_summary['Standard Error'])
	plt.setp((ax1,ax3), xticks=index, xticklabels=label, ylabel='Index')
	plt.show()

def plot_enhance_additive_histograms(unit_stats):
	unit_160_delay = unit_stats[unit_stats['category'] != 'Unresponsive'][unit_stats['category'] != 'Error - Multisensory suppressed'].xs(120, level=4)
	plt.figure(3)
	plt.subplots_adjust(hspace=0.5)
	plt.subplot(2,1,1)
	unit_160_delay['enhancement_index'].hist(grid=False)
	plt.title('Distribution of the Enhancement Index')
	plt.xlim(-100, 100)
	plt.subplot(2,1,2)
	unit_160_delay['additivity_index'].hist(grid=False)
	plt.title('Distribution of the Additivity Index')
	plt.xlim(-100, 100)
	plt.show()

def plot_bubble(response_stats):
	#figure 4
	response_stats['frequency'] = (response_stats['frequency'] / 1000).round()
	response_stats['attenuation'] = 75-response_stats['attenuation']
	count = response_stats.droplevel((0, 2, 3, 4, 5))[['frequency', 'attenuation']].reset_index().drop_duplicates().groupby(
		['frequency', 'attenuation']).size().to_frame(name='Instances').reset_index()
	print(sum(count['Instances']))
	cmap = sns.cubehelix_palette(dark=.2, light=.6, as_cmap=True)
	ax = sns.scatterplot(x='frequency', y='attenuation', hue='Instances', size='Instances', sizes=(50, 200), palette=cmap,
						 legend='full', data=count)
	ax.set(xlabel='Frequency (kHz)', ylabel='Sound Level (dB SPL)')

def plot_anatomyScatter(response_stats, unit_stats):
	#figure 7
	categories = unit_stats[unit_stats['category'] != 'Unresponsive']['category'].xs(1,level=4)
	data = response_stats.xs((2, 0), level=(2, 5)).loc[
		categories.index].join(categories).reset_index().drop(
		['duration', 'peak_response', 'p_value', 'baseline', 'statistic'], axis=1)
	data['category'] = data['category'].replace(
		{'Bimodal subthreshold': 'Bimodal', 'Bimodal integrative': 'Bimodal',
		 'Bimodal non-integrative': 'Bimodal', 'Auditory subthreshold': 'Auditory',
		 'Visual subthreshold': 'Visual', 'Auditory unimodal':'Auditory', 'Visual unimodal':'Visual'})
	data['channel'] = data['channel'].replace(
		{6: 12, 4: 11, 2: 10, 0: 9, 1: 8, 3: 7, 5: 6, 7: 5, 9: 4, 11: 3, 13: 2, 15: 1, 14: 1, 8:5})
	data['frequency'] = data['frequency'].astype(int)
	data.rename(columns={"category":'Category'})
	data.loc[data['frequency'] == 11313, 'frequency']= 11319

	hz_graph_data = ((data.groupby(['frequency', 'category']).count()['cat'] / data.count()['cat'])*100).reset_index()
	channel_graph_data = ((data.groupby(['channel', 'category']).count()['cat'] / data.count()['cat'])*100).reset_index()

	plt.subplots_adjust(hspace=0.2)
	plt.rcParams.update({'font.size': 22})
	plt.subplot(211)
	sns.set(font_scale=1, style="whitegrid")
	ax1 = sns.barplot(x='frequency', y='cat', hue='category', data=hz_graph_data, saturation=1,
					  palette=sns.xkcd_palette(["windows blue", "orange", "grass green"]))
	ax1.set(xlabel='Frequency (kHz)', ylabel='Percentage of Units')
	ax1.set_xticklabels([0.7, 1.2, 2.0, 2.8, 3.3, 4.0, 4.8, 5.7, 9.5, 11.3, 13.5, 16.0, 19.0, 22.6, 26.9])
	plt.legend(loc='upper right')
	plt.subplot(212)
	sns.set(font_scale=1, style="whitegrid")
	plt.rcParams.update({'font.size': 22})
	ax2 = sns.barplot(y='channel', x='cat', hue='category', orient='h', data=channel_graph_data, saturation=1,
					  palette=sns.xkcd_palette(["windows blue", "orange", "grass green"]))
	ax2.invert_yaxis()
	ax2.set(xlabel='Percentage of Units', ylabel='Channel')
	plt.legend(loc='lower right')

def mark_unresponsive_manual(unit_stats):
	for key in UNRESPONSIVE_KEYS: unit_stats.loc[key,'category'] = 'Unresponsive' 
	return unit_stats

def run():
	source_df = pd.DataFrame([])
	try:
		print(datetime.datetime.utcnow(), "Reading fixed file if exists...")
		source_df = pd.read_hdf('results_fixed.h5')
		print(datetime.datetime.utcnow(), "Successfully read fixed file!")
	except FileNotFoundError:
		print(datetime.datetime.utcnow(), "Fixed file doesnt exist. Reading long HDF...")
		df = pd.read_hdf('results_long.h5')
		print(datetime.datetime.utcnow(), "Cleaning NaNs")
		df.reset_index(inplace=True)
		df.fillna({'delay': 0}, inplace=True)
		df.set_index(['cat', 'penetration', 'block', 'channel', 'unit', 'delay', 'duration', 'frequency', 'attenuation'],
					 inplace=True)
		print(datetime.datetime.utcnow(), "Realigning plotly.graph_objs.carpet.Aaxis          delays")
		df = df.groupby(level=(0, 1, 2, 3, 4, 5)).apply(delay_fix)
		print(datetime.datetime.utcnow(), "Writing fixed file...")
		df.to_hdf('results_fixed.h5', key='cats', mode='w', data_columns=True, complevel=9)

	# To run analysis and plot
	print(datetime.datetime.utcnow(), "Analyzing...")
	#df = source_df[:10000]
	df = source_df
	# take out duration, frequency, and attenuation (from 75dbSPL) and put them in response stats as columns
	response_stats = extract_misc_levels_to_response_stats(df)
	# removes those 3 levels from df
	df = remove_misc_levels(df)
	# adding more columns, based off the response_stats indices, to response_stats from wilcoxon ranksum test (statistic and p value)
	response_stats = analyze_responses(df).join(response_stats)
	psth = analyze_psth(df, response_stats)
	response_stats = analyze_peak_responses(psth).join(response_stats)
	all_unit_stats = analyze_units(df, response_stats, psth)
	unit_stats = mark_unresponsive_manual(all_unit_stats)
	unit_stats = unit_stats[unit_stats['category'] != 'Absent'][unit_stats['category'] != 'Unresponsive']
	#unit_stats = unit_stats.groupby(group_keys=False, level=(0,1)).apply(lambda group_rows: None if (group_rows['category'] == 'Unresponsive').all() else group_rows)
	print(unit_stats[unit_stats['category'] != 'Unresponsive'].xs(120, level=4).groupby('category').count())
	print(unit_stats[unit_stats['category'].str.contains("Bimodal")])
	print(datetime.datetime.utcnow(), "Plotting...")
	plot_auditory_binding(unit_stats)
	plot_enhance_additive_histograms(unit_stats)
	plot_mean_population(psth)
 

#run()
# to run... exec(open('calc_results.py').read())

#to change categories a.loc[(penetration,channel,block,unit),'category'] = 'poop'
#to edit all of the false positive units

#syntactic sugar: makes code more concise
#example: lambda have implicit return
#ternary operator (one line if statement)
# 'Yes' if fruit=='Apple' else 'No'