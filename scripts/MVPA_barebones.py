# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:49:34 2022
- mvpa demo from Richard Höchenberger's workshop
- Refer https://nbviewer.org/github/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021/blob/main/Scripts/notebooks/Figure4_5_temporaldecodingstimuli.ipynb

@author: Rahul Venugopal
"""
#%% Loading libraries

import matplotlib.pyplot as plt
import numpy as np

# scikit-learn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# mne decoding libraries
import mne
from mne.decoding import SlidingEstimator, LinearModel, cross_val_multiscore,  get_coef, GeneralizingEstimator

#%% Load epochs*channels*timepoints

raw_fname = 'sample_audvis_filt-0-40_raw.fif'
tmin, tmax = -0.200, 0.500
event_id = {'Auditory/Left': 1, 'Visual/Left': 3}  # just use two
raw = mne.io.read_raw_fif(raw_fname)
raw.pick_types(meg='grad', stim=True, eog=True, exclude=())

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the example to run faster. The 2 Hz high-pass helps improve CSP.
raw.load_data().filter(2, 20)
events = mne.find_events(raw, 'STI 014')

# Set up bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443']  # bads + 2 more

# Read epochs
epochs = mne.Epochs(raw, events, event_id,
                    tmin, tmax,
                    proj=True,
                    picks=('grad', 'eog'),
                    baseline=(None, 0.), preload=True,
                    reject=dict(grad=4000e-13,
                                eog=150e-6),
                    decim=3,
                    verbose='error')

epochs.pick_types(meg=True, exclude='bads')  # remove stim and EOG
del raw

# To keep chance level at 50% accuracy, we first equalize the number of epochs in each condition
epochs.equalize_event_counts(epochs.event_id)
epochs

# Extracting data and labels
X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
y = epochs.events[:, 2]  # target: auditory left vs visual left

#%% We can do this more simply using the mne.decoding module! Let's go. 🚀
'''
Decoding over time: Comparisons at every single time point.

In the previous examples, we have trained a classifier to discriminate between experimental conditions by using the spatio-temporal patterns of entire trials. Consequently, the classifier was (hopefully!) able to predict which activation patterns belonged to which condition.

However, an interesting neuroscientific is: Exactly when do the brain signals for two conditions differ?

We can try to answer this question by fitting a classifier at every single time point. If the classifier can successfully discriminate between the two conditions, we can conclude that the spatial activation patterns measured by the MEG or EEG sensors differed at this very time point.

'''

# Classifier pipeline. No need for vectorization as in the previous example.
clf = make_pipeline(StandardScaler(),
                    LinearModel(LogisticRegression(max_iter=1000)))

# The "sliding estimator" will train the classifier at each time point.
scoring = 'roc_auc'
time_decoder = SlidingEstimator(clf,
                                scoring=scoring,
                                n_jobs=-1,
                                verbose=True)

# Run cross-validation.
n_splits = 5
scores = cross_val_multiscore(time_decoder,
                              X,
                              y,
                              cv=n_splits,
                              n_jobs=-1)

# Mean scores across cross-validation splits, for each time point.
mean_scores = np.mean(scores, axis=0)

# Mean score across all time points.
mean_across_all_times = round(np.mean(scores), 3)
print(f'\n=> Mean CV score across all time points: {mean_across_all_times:.3f}')

#%% Plot the classification results!
fig, ax = plt.subplots()

ax.axhline(0.5, color='k', linestyle='--', label='chance')  # AUC = 0.5
ax.axvline(0, color='k', linestyle='-')  # Mark time point zero.
ax.plot(epochs.times, mean_scores, label='score')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean ROC AUC')
ax.legend()
ax.set_title('Left vs Right')
fig.suptitle('Sensor Space Decoding')

#%% Retrieve the spatial filters and spatial patterns if you explicitly use a LinearModel
clf = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression(solver='liblinear', max_iter=1000))
)

time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
time_decod.fit(X, y)

coef = get_coef(time_decod, 'patterns_', inverse_transform=True)

evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])

joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))

evoked_time_gen.plot_joint(times=np.arange(0., .500, .100), title='patterns',
                           **joint_kwargs)

# %%
# Temporal generalization
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Temporal generalization is an extension of the decoding over time approach.
# It consists in evaluating whether the model estimated at a particular
# time instant accurately predicts any other time instant. It is analogous to
# transferring a trained model to a distinct learning problem, where the
# problems correspond to decoding the patterns of brain activity recorded at
# distinct time instants.
#
# The object to for Temporal generalization is
# :class:`mne.decoding.GeneralizingEstimator`. It expects as input :math:`X`
# and :math:`y` (similarly to :class:`~mne.decoding.SlidingEstimator`) but
# generates predictions from each model for all time instants. The class
# :class:`~mne.decoding.GeneralizingEstimator` is generic and will treat the
# last dimension as the one to be used for generalization testing. For
# convenience, here, we refer to it as different tasks. If :math:`X`
# corresponds to epochs data then the last dimension is time.
#
# This runs the analysis used in :footcite:`KingEtAl2014` and further detailed
# in :footcite:`KingDehaene2014`:

# define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc',
                                 verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')

# %%
# Plot the full (generalization) matrix:

fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('AUC')

#%% Decoding shadow plots and cluster stats

# Extra libraries
from scipy import stats
import scipy

from mne.stats import ttest_1samp_no_p
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       permutation_cluster_1samp_test)

def movingaverage(y, window_length):
    y_smooth = scipy.convolve(y,np.ones(window_length,dtype='float'), 'same')/scipy.convolve(np.ones(len(y)),                                       np.ones(window_length), 'same')
    return y_smooth

# Decoding plot
def decodingplot(scores_cond, p_values_cond,
                 times, rts = None, alpha=0.05,
                 color = 'r', tmin= -0.3, tmax = 1.6):

    scores = np.array(scores_cond)
    sig= p_values_cond < alpha

    scores_m = np.nanmean(scores, axis=0)
    n = len(scores)
    n -= sum(np.isnan(np.mean(scores, axis=1))) #identify the nan subjs and remove them..
    sem = np.nanstd(scores, axis=0) / np.sqrt(n)


    fig, ax1 = plt.subplots(nrows=1, figsize=[20, 3])

    ax1.plot(times, scores_m, 'k',linewidth=1,)
    ax1.fill_between(times, scores_m-sem, scores_m+sem, color=color, alpha=0.3)

    split_ydata = scores_m
    split_ydata[~sig] = np.nan

    #shade the significant regions..
    ax1.plot(times, split_ydata,color='k', linewidth=3)
    ax1.fill_between(times, y1=split_ydata, y2=0.5, alpha=0.7, facecolor=color)



    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.grid(True)


    ax1.axhline(y=0.5, linewidth=0.75, color='k',linestyle = '--')
    ax1.axvline(x=0, linewidth=0.75, color='k',linestyle = '--')

    timeintervals = np.arange(tmin, tmax, 0.1)
    timeintervals = timeintervals.round(decimals=2)

    ax1.set_xticks(timeintervals)

    for patch in ax1.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    ax1.patch.set_edgecolor('black')

    ax1.patch.set_linewidth('0')


    for a in fig.axes:
        a.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=True,
        top=False,
        labelbottom=True)    # labels along the bottom edge are on

    class Scratch(object):
        pass

    returnval1 = Scratch()

    returnval1.axes = ax1
    returnval1.times = times[sig]
    returnval1.scores = scores_m[sig]

    return returnval1

#%% Stats stuff
chance = .5
alpha = 0.05

#adapted from https://github.com/kingjr/decod_unseen_maintenance/blob/master/scripts/base.py
#performs stats of the group level..
#X is usually nsubj x ntpts -> composed of mean roc scores per subj per timepoint..
#performs cluster stats on X to identify regions of tpts that have roc significantly
#differ from chance..

def _stats(X, connectivity=None, n_jobs=-1):
    """Cluster statistics to control for multiple comparisons.
    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    connectivity : None | array, shape (n_space, n_times)
        The connectivity matrix to apply cluster correction. If None uses
        neighboring cells of X.
    n_jobs : int
        The number of parallel processors.
    """
    n_subjects = len(X)
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    #this functions gets the t-values and performs a cluster permutation test on them to determine
    #p-values..
    p_threshold = 0.05
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('number of subjects:', n_subjects)
    print('t-threshold is:', t_threshold)
    print('p-threshold is:', p_threshold)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=2**12, seed = 1234,
        n_jobs=n_jobs, connectivity=connectivity,threshold=t_threshold)
    p_values_ = np.ones_like(X[0]).T
    #rearrange the p-value per cluster..
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T

def _stat_fun(x, sigma=0, method='relative'):
    """This secondary function reduces the time of computation of p-values and adjusts for small-variance
       values
    """
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values
#%% Plot the MVPA decoding

#compute the stats..
p_values_alert = _stats(np.array(scores_alert)[:, :, None] - chance)

funcreturn = decodingplot(scores_alert,p_values_alert,
                          times_alert,
                          alpha = 0.05, color = 'r',
                          tmin= -0.2, tmax = 1.6)

funcreturn.axes.set_ylim(0, 1)
funcreturn.axes.set_ylabel('AUC')

funcreturn.axes.set_xlabel('\nTime (sec)')
plt.savefig("DecodedPlot.png",
            dpi=240,
            bbox_inches = 'tight',
            transparent=False,
            pad_inches = 0.1)