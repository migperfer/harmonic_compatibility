from joblib import Parallel, delayed
import numpy as np
import cupy as cp
from itertools import combinations
import essentia.standard as estd


def mix_dissonance(audio_vector1, audio_vector2):
    """
    Dissonance created by the mix of two audios. Sampling rate should be 44100
    :param audio_vector1: Numpy array of audio 1
    :param audio_vector2: Numpy array of audio 2
    :return: overall dissonance, framewise roughness
    """
    f1, m1 = nSinesRead(audio_vector1)
    f2, m2 = nSinesRead(audio_vector2)

    f2shift = octavescale(f2)
    framewiseRoughness, ov = final(f1, f2shift, m1, m2)
    return ov, framewiseRoughness


def final(f1, f2shift, m1, m2):
    """
    Returns by-frame and overall dissonance values between two tracks, the second pitch-shifted and compared to the first for each pitch-shift
    Input: frequency and magnitude arrays for each of the tracks (size each 64*60)
    Output: framewise dissonace, overall dissonance
    """
    shftD = shiftdiss(f1, f2shift, m1, m2)  # np.array of framewise roughness-matrices for each of the 97F pitch-shifts
    shiftedOverallDiss = np.sum(shftD, axis=1)  # computes overall dissonance over all the frames for each pitchshift

    return shftD, shiftedOverallDiss


def shiftdiss(f1, f2shift, m1, m2):
    """
    function to call roughness/dissonance function for every single pitch-shift
    """
    n_shifts = f2shift.shape[0]
    n_frames = np.min((len(f1), f2shift.shape[1]))
    framewise_dissonance = Parallel(n_jobs=4, verbose=2)(delayed(roughdiss)
                                                         (f1, f2shift[i], m1, m2) for i in range(n_shifts))

    return np.array(framewise_dissonance)


def dissonance(f1, f2):
    """
    Calculate the dissonance created by two frequencies
    :param f1: First frequency
    :param f2: Second frequency
    :return: Dissonance created
    """
    fdif = abs(f1 - f2)  # compute abolute difference of two frequencies
    fI = (f1 + f2) / 2  # compute frequncy value between two frequencies
    CBW = CBWMooreGlasb(fI)  # compute critical bandwidth of middle frequency
    y = fdif / CBW  # compute critical bandwidth interval (HUTCHINSON AND KNOPOFF)

    if y < 1.2:  # check if critical bandwidth interval is lower than threshold of roughness (Gilberto)
        g = (np.e * (y / 0.25) * np.exp(-y / .25)) ** 2  # compute roughness
    else:
        g = 0
    return g


def roughdiss(f1, f2, m1, m2):
    """
    Computes roughness values for each dyad of a merged timbre of two track's frequency array per frame
    Input: frequency and magnitude arrays of two tracks
    Output: List of 100 matrices of roughness values of each dyad of each of the merged timbre's sinusoids (120x120)
    """
    min_length = int(np.amin(np.array([len(f1), len(f2)])))  # In case the number of frames are not equal
    framewise_dissonance = cp.zeros(min_length)  # initialise list for all framewise dissonance values to be inserted
    f1 = f1[:min_length]
    f2 = f2[:min_length]
    m1 = m1[:min_length]
    m2 = m2[:min_length]

    n_comb = 780  # Number of combinations without repetition for pairs in 40 dim array

    mergedf = np.hstack((f1, f2))
    mergedm = np.hstack((m1, m2))
    for i in range(min_length):  # iterate over all frames
        ffus = mergedf[i]  # Current frame frequencies
        mfus = mergedm[i]  # Current frame magnitudes
        roughness_kernel = cp.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 g',
            r'''
            float midfreq = 0;
            float cbw = 0;
            float critint = 0;
            float diffreq = 0;
            g = 0.;
            
            diffreq = fabs(x-y);
            midfreq = (x+y)/2;
            cbw = 6.23 * ((midfreq / 1000.) * (midfreq / 1000.)) + 93.39 * (midfreq / 1000.) + 28.52;
            critint = diffreq/cbw;
            if (critint < 1.2){
                g = exp(1.) * (critint/0.25) * exp(-critint/0.25) * exp(1.) * (critint/0.25) * exp(-critint/0.25);
            }
            ''',
            'roughness_kernel'
        )
        all_comb_f = np.array(list(combinations(ffus, 2)))
        all_comb_m = np.array(list(combinations(mfus, 2)))
        x_gpu = all_comb_f[:, 0]
        y_gpu = all_comb_f[:, 1]
        m1_gpu = cp.array(all_comb_m[:, 0], np.float32)
        m2_gpu = cp.array(all_comb_m[:, 1], np.float32)
        freqr = roughness_kernel(cp.array(x_gpu, np.float32),
                             cp.array(y_gpu, np.float32))

        num = cp.sum(freqr*m1_gpu*m2_gpu)
        denom = cp.sum(m2_gpu * m2_gpu)*2
        if denom != 0:
            framewise_dissonance[i] = num/denom
        else:
            framewise_dissonance[i] = 0

    return cp.asnumpy(framewise_dissonance)



def CBWMooreGlasb(f):
    """
    estimate equivalent rectangular bandwidth for specific frequency value with formula by Moore & Glasberg
    """
    CBW = 6.23 * (f / 1000.) ** 2 + 93.39 * (f / 1000.) + 28.52  # Terhardt (9.28)
    return CBW


def octavescale(f2):
    """
    Perform a octave scale shifting for all frequencies in f2. A pitch shift is divided into 96 equal steps.
    :param f2: MxN matrix. M is the number of frames, and N is the number of sinusoids for each frame
    :return: 97xMxN matrix. Where 97 ranges between -48 pitch shift and 49 pitch shift
    """
    n_shifts = 48
    zero_indexes = np.where(f2 == 0)
    f2shift = np.zeros((n_shifts*2 + 1, ) + f2.shape)  # Create a numpy array consisting of 97 possible for all frame & freqs

    for j in range(n_shifts*2 + 1):  # For each pitch shift
        shift = n_shifts - j
        p_shift = 2 ** (np.log2(f2 + shift * (1 / 96.)))
        p_shift[zero_indexes] = 0
        f2shift[j] = p_shift
    return f2shift


def nSinesRead(audio_vector):
    sineanal = estd.SineModelAnal(maxnSines=20)
    fft_calc = estd.FFT(size=2048)
    results = []
    for frame in estd.FrameGenerator(audio_vector, 2048, 1024):
        spec = fft_calc(frame)
        results.append(sineanal(spec))
    results = np.array(results)
    freqs = results[:, 0, :]
    mags = results[:, 1, :]
    return freqs, mags
