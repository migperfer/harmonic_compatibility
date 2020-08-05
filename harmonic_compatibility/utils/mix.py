import essentia.standard as std
import ffmpeg
import numpy as np
import pandas as pd
import os
from pyrubberband.pyrb import pitch_shift, frequency_multiply

eps = np.finfo(float).eps

__all__ = ['mix', 'create_mashabilities_examples', 'create_tiv_examples',
            'create_gebhardt_dissonance_examples', 'create_p_harmonicity_examples', 
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


def create_gebhardt_dissonance_examples(audio_target_file, file_compatibilities, audios_folder, n_examples=3, sr=44100):
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