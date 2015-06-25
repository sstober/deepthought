__author__ = 'sstober'

from deepthought.analysis.tempo.autocorrelation import *
import scipy.stats as stats

def sliding_window_tempo_analysis(trial, sfreq,
                                  bpm_min, bpm_max,
                                  start_sample, stop_sample,
                                  window_length, hop_size,
                                  mode='median',
                                  verbose=False):

    bpm_values = np.arange(bpm_min, bpm_max+1, 1)

    best_pairs = []  # only stores lower index of pair
    best_scores = [] # stores max score of each pair

    ac_bpm_mat = []
    trial = trial[:,:stop_sample].mean(axis=0) # reduce 64 channels to 1 mean value - TODO: could be improved
#     print trial.shape
    for i in range(start_sample, stop_sample - window_length, hop_size):
        data = trial[i:i+window_length]

        ac = compute_autocorrelation(data)
        tempo_curve = compute_tempo_histogram_from_ac(ac, sfreq)
        y = tempo_curve(bpm_values)

        ac_bpm_mat.append(y)

        # TODO: In the following, only even BPM values are considered for upper. This could be improved.

        # aggregate scores
        pair_scores = []
        # Note: l and u are indices in bpm_values
        for l in range(len(y)):
            u = l + bpm_values[l] # to get the tempo octave, we have to add the actual tempo to the lower index
            if u >= len(y):
                break
            # get the scores for these 2 tempo values
            lower = y[l]
            upper = y[u]

            pair_scores.append(max(lower, upper))

#             print l, x[l], x[u], pair_scores[-1], lower, upper

        pair_scores = np.asarray(pair_scores)
        best_pair = np.argmax(pair_scores)

        best_score = pair_scores[best_pair]
#         print i, 'best pair:', x[best_pair], x[best_pair + x[best_pair]]

        best_pairs.append(best_pair)
        best_scores.append(best_score)

    ac_bpm_mat = np.asarray(ac_bpm_mat)

    best_pairs = np.asarray(best_pairs)
    best_scores = np.asarray(best_scores)

#     print best_pairs + bpm_min
#     print best_scores
    best_i_mean = int(np.mean(best_pairs + bpm_min) - bpm_min) # add bpm offset for mean computation and remove afterwards
    best_i_median = int(np.median(best_pairs))
    best_i_mode = stats.mode(best_pairs)[0][0]
    best_i_max = best_pairs[best_scores.argmax()]

    if verbose:
        print 'mean:', bpm_values[best_i_mean]
        print 'median:', bpm_values[best_i_median]
        print 'mode:', bpm_values[best_i_mode]
        print 'max:', bpm_values[best_i_max]


    if mode == 'mean':
        best_lower = best_i_mean
    elif mode == 'median':
        best_lower = best_i_median # median index of best (lower) BPM value
    elif mode == 'mode':
        best_lower = best_i_mode # index that appears most often
    elif mode == 'max':
        best_lower = best_i_max
    else:
        best_lower = best_i_median # default

    best_upper = best_lower + bpm_values[best_lower]
#     print 'best pair:', x[best_lower], x[beat_upper]

    ac_bpm_mean = ac_bpm_mat.mean(axis=0)
#     ac_bpm_max = ac_bpm_mat.max(axis=0)

    if ac_bpm_mean[best_lower] > ac_bpm_mean[best_upper]:
        best = [best_lower, best_upper]
    else:
        best = [best_upper, best_lower]

    return bpm_values[best], ac_bpm_mat, bpm_values[best_pairs], best_scores