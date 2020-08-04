import essentia.standard as std
import ffmpeg
import numpy as np
import pandas as pd
import os
from pyrubberband.pyrb import pitch_shift, frequency_multiply

eps = np.finfo(float).eps

__all__ = ['mix', 'create_mashabilities_examples', 'create_tiv_examples',
           'create_roman_dissonance_examples', 'get_sines_per_frame',
           , 'create_p_harmonicity_examples',
           'create_inharmonicity_examples', 'create_dissonances_examples',
           'create_hutchinson_examples']

def mix(audio1, audio2, sr):
    """
    Function to mix audios with a normalised loudness
    :param audio1: Audio vector to normalize
    :param audio2: Audio vector to normalize
    :param sr: Sample rate of the final mix
    :return: Audio vector of the normalised mix
    """
    if audio1.ndim > 1:
        audio1 = std.MonoMixer()(audio1, audio1.shape[1])
    if audio2.ndim > 1:
        audio2 = std.MonoMixer()(audio2, audio2.shape[1])
    std.MonoWriter(filename='temporal1.wav', sampleRate=sr)(audio1)
    std.MonoWriter(filename='temporal2.wav', sampleRate=sr)(audio2)

    stream1 = (
        ffmpeg
            .input('temporal1.wav')
            .filter('loudnorm')
    )

    stream2 = (
        ffmpeg
            .input('temporal2.wav')
            .filter('loudnorm')
    )
    merged_audio = ffmpeg.filter([stream1, stream2], 'amix')
    ffmpeg.output(merged_audio, 'temporal_o.wav').overwrite_output().run()

    audio_numpy = std.MonoLoader(filename='temporal_o.wav')()
    return audio_numpy

def create_mashabilities_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100, spectr=True):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :param spectr: Boolean indicating if spectral balance should be applied.
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    toreturnlist = []
    if spectr:
        sorteddf = df.sort_values(by=['mashability'], ascending=False)
    else:
        sorteddf = df.sort_values(by=['h_contr'], ascending=False)
    top = sorteddf[['filename', 'pitch_offset']][:n_examples]
    for idx, candidate in top.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch_offset']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        audio_candidate = pitch_shift(audio_candidate, sr, pshift).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        toreturnlist.append(audio)
    return toreturnlist


def create_tiv_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100, type_generate=['framewise', 'beatwise', 'whole']):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :param type_generate: List with the time resolutions to be used. By default all framewise, beatwise and whole are used.
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []
    df_fwise = df[['compatibility_framewise', 'pitch_shift_framewise', 'filename']].sort_values(by=['compatibility_framewise'], ascending=True).iloc[:n_examples, :]
    df_bwise = df[['compatibility_beatwise', 'pitch_shift_beatwise', 'filename']].sort_values(by=['compatibility_beatwise'], ascending=True).iloc[:n_examples, :]
    df_whole = df[['compatibility_whole', 'pitch_shift_whole', 'filename']].sort_values(by=['compatibility_whole'], ascending=True).iloc[:n_examples, :]
    # Framewise examples
    if 'framewise' in type_generate:
        for idx, candidate in df_fwise.iterrows():
            cand_f = candidate['filename'].split('/')[-1]
            pshift = candidate['pitch_shift_framewise']
            audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
            audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
            audio_candidate = pitch_shift(audio_candidate, sr, pshift).astype(np.float32)
            audio = mix(audio_target, audio_candidate, sr=sr)
            listoreturn.append(audio)

    # Beatwise examples
    if 'beatwise' in type_generate:
        for idx, candidate in df_bwise.iterrows():
            cand_f = candidate['filename'].split('/')[-1]
            pshift = candidate['pitch_shift_beatwise']
            audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
            audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
            audio_candidate = pitch_shift(audio_candidate, sr, pshift).astype(np.float32)
            audio = mix(audio_target, audio_candidate, sr=sr)
            listoreturn.append(audio)

    # Whole audio examples
    if 'whole' in type_generate:
        for idx, candidate in df_whole.iterrows():
            cand_f = candidate['filename'].split('/')[-1]
            pshift = candidate['pitch_shift_whole']
            audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
            audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
            audio_candidate = pitch_shift(audio_candidate, sr, pshift).astype(np.float32)
            audio = mix(audio_target, audio_candidate, sr=sr)
            listoreturn.append(audio)
    return listoreturn


def create_roman_dissonance_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []

    df_sorted = df.sort_values(by=['dissonance'], ascending=True).iloc[:n_examples, :]
    for idx, candidate in df_sorted.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        tunning = np.mean(std.TuningFrequencyExtractor()(audio_candidate))
        tunning_main = np.mean(std.TuningFrequencyExtractor()(audio_target))
        factor_tuning = tunning/tunning_main
        pitch_factor = factor_tuning*np.exp2(pshift/96)
        audio_candidate = frequency_multiply(audio_candidate, 44100, pitch_factor).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        listoreturn.append(audio)

    return listoreturn


def create_p_harmonicity_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []

    df_sorted = df.sort_values(by=['compatibility_framewise'], ascending=True).iloc[:n_examples, :]
    for idx, candidate in df_sorted.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch_shift_framewise']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        audio_candidate = pitch_shift(audio_candidate, 44100, pshift).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        listoreturn.append(audio)

    return listoreturn

def create_hutchinson_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []

    df_sorted = df.sort_values(by=['dissonance'], ascending=True).iloc[:n_examples, :]
    for idx, candidate in df_sorted.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        tunning = np.mean(std.TuningFrequencyExtractor()(audio_candidate))
        tunning_main = np.mean(std.TuningFrequencyExtractor()(audio_target))
        factor_tuning = tunning/tunning_main
        pitch_factor = factor_tuning*np.exp2(pshift/96)
        audio_candidate = frequency_multiply(audio_candidate, 44100, pitch_factor).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        listoreturn.append(audio)

    return listoreturn


def create_inharmonicity_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []

    df_sorted = df.sort_values(by=['compatibility_framewise'], ascending=True).iloc[:n_examples, :]
    for idx, candidate in df_sorted.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch_shift_framewise']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        audio_candidate = pitch_shift(audio_candidate, 44100, pshift).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        listoreturn.append(audio)

    return listoreturn

def create_dissonances_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
    """
    Given an audio and a file with its compatibilities create the mixes
    :param audio_target_file: The audio itself
    :param file_compatibilities: The compatibility file for the target audio
    :param audios_folder: The folder where are located all the audios
    :param n_examples: Number of examples to generate
    :param sr: sample rate of the final mix
    :return: A list where each element is a mix
    """
    df = pd.read_csv(file_compatibilities)
    listoreturn = []

    df_sorted = df.sort_values(by=['compatibility_framewise'], ascending=True).iloc[:n_examples, :]
    for idx, candidate in df_sorted.iterrows():
        cand_f = candidate['filename'].split('/')[-1]
        pshift = candidate['pitch_shift_framewise']
        audio_target = std.MonoLoader(filename=os.path.join(audios_folder, audio_target_file), sampleRate=sr)()
        audio_candidate = std.MonoLoader(filename=os.path.join(audios_folder, cand_f), sampleRate=sr)()
        audio_candidate = pitch_shift(audio_candidate, 44100, pshift).astype(np.float32)
        audio = mix(audio_target, audio_candidate, sr=sr)
        listoreturn.append(audio)

    return listoreturn


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