import numpy as np
import essentia.standard as std

__all__ = ['get_sines_per_frame', 'get_hpeaks_per_frame']

def get_sines_per_frame(audio, sr=44100, onlyfrecuencies=False, nsines=20):
    """
    Perform framewise sinusoidal model in an audio
    :param audio: Audio either mono or stereo. Will be downsampled to mono
    :param sr: Samplerate used for the audio
    :return: Nx2x100. N is the number of resulting frames. 2x100 are the frequencies and magnitudes respectively.
    """
    if audio.ndim > 1:
        audio = std.MonoMixer()(audio, audio.shape[1])

    len_arrays = 0
    for i, _ in enumerate(std.FrameGenerator(audio, frameSize=4096, hopSize=2048)):
        len_arrays = i

    fft_algo = std.FFT()
    sine_anal = std.SineModelAnal(maxnSines=nsines, orderBy='frequency', minFrequency=1)
    sines = np.zeros([len_arrays + 1, 2, nsines], dtype=np.float32) + eps
    for i, frame in enumerate(std.FrameGenerator(audio, frameSize=4096, hopSize=2048)):
        fft = fft_algo(frame)
        freqs, mags, _ = sine_anal(fft)
        sorting_indexes = np.argsort(freqs)
        freqs = freqs[sorting_indexes]
        mags = mags[sorting_indexes]
        sines[i, :] = [freqs, mags]
    if onlyfrecuencies:
        return sines[:, 0, :]
    else:
        return sines[:, 0, :], sines[:, 1, :]


def get_hpeaks_per_frame(audio, sr=44100, onlyfrecuencies=False, nsines=20):
    """
    Get Harmonic peaks in an audio
    :param audio: Audio either mono or stereo. Will be downsampled to mono
    :param sr: Samplerate used for the audio
    :return: Nx2x100. N is the number of resulting frames. 2x100 are the frequencies and magnitudes respectively.
    """
    if audio.ndim > 1:
        audio = std.MonoMixer()(audio, audio.shape[1])

    fft_algo = std.FFT()
    pyin = std.PitchYin()
    hpeaks = std.HarmonicPeaks()
    sine_anal = std.SineModelAnal(maxnSines=nsines, orderBy='frequency', minFrequency=1)
    sines = []
    for i, frame in enumerate(std.FrameGenerator(audio, frameSize=4096, hopSize=2048)):
        pitch, _ = pyin(frame)
        fft = fft_algo(frame)
        freqs, mags, _ = sine_anal(fft)
        sorting_indexes = np.argsort(freqs)
        freqs = freqs[sorting_indexes]
        mags = mags[sorting_indexes]
        non_zero_freqs = np.where(freqs != 0)
        freqs = freqs[non_zero_freqs]
        mags = mags[non_zero_freqs]
        freqs, mags = hpeaks(freqs, mags, pitch)
        sines.append([freqs, mags])
    sines = np.array(sines)
    if onlyfrecuencies:
        return sines[:, 0, :]
    else:
        return sines[:, 0, :], sines[:, 1, :]