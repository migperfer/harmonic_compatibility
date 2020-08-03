import csv
import glob
import shutil
import sys
from essentia.standard import MonoLoader
import numpy as np
from scipy import signal
from .segmentation import get_beat_sync_chroma_and_spectrum
from .utilities import mix_songs
import os


class ShorterException(Exception):
    pass


def mashability(base_beat_sync_chroma, base_beat_sync_spec, audio_file_candidate):
    """
    Calculate the mashability of two songs.
    :param base_beat_sync_chroma: The beat synchronous chroma.
    :param base_beat_sync_spec: The beat synchronous spectrogram.
    :param audio_file_candidate: The path to the candidate for mashability.
    :return: A tuple containing: mashability value, the pitch offset, beat offset.
    """
    try:
        y = MonoLoader(filename=audio_file_candidate)()
        sr = 44100
    except:
        raise ShorterException("EOF error")
    if len(y)/sr < 3:
        raise ShorterException("Candidate is smaller than 3 seconds")
    # 1st step: Calculate harmonic compatibility
    c_bsc, c_bss = get_beat_sync_chroma_and_spectrum(audio_file_candidate)
    c_bsc = np.flip(c_bsc)  # Flip to make correlation, no convolution
    stacked_beat_sync_chroma = np.vstack([c_bsc, c_bsc])
    conv = signal.convolve2d(stacked_beat_sync_chroma, base_beat_sync_chroma, )
    base_n = np.linalg.norm(base_beat_sync_chroma)
    cand_n = np.linalg.norm(c_bsc)
    h_mas = conv / (base_n * cand_n)
    offset = base_beat_sync_chroma.shape[1]-1
    h_mas = np.flip(h_mas[11:-11, offset:-offset], axis=0)
    h_mas_k = np.max(h_mas, axis=0)  # Maximum mashability for each beat displacement

    # 3rd step: Calculate Spectral balance compatibility
    if c_bss.shape[1] >= base_beat_sync_spec.shape[1]:
        beat_length = base_beat_sync_spec.shape[1]
        n_max_b_shifts = c_bss.shape[1] - base_beat_sync_spec.shape[1]
        r_mas_k = np.zeros(n_max_b_shifts+1)
        for i in range(n_max_b_shifts+1):
            beta = np.mean(base_beat_sync_spec + c_bss[:, i:i + beat_length], axis=1)
            beta_norm = beta/np.sum(beta)
            r_mas_k[i] = 1 - np.std(beta_norm)  # Spectral balance for i beat displacement
    else:
        raise ShorterException("Candidate song has lesser beats than base song")
    res_mash = h_mas_k + 0.2 * r_mas_k
    b_offset = np.argmax(res_mash)
    p_shift = np.argmax(h_mas[:, b_offset])
    if p_shift > 6:
        p_shift = 12 - p_shift

    h_contr = h_mas_k[b_offset]
    r_contr = r_mas_k[b_offset]
    return np.max(res_mash), p_shift, b_offset, h_contr, r_contr


def get_mashability(audio1_vector, audio2_vector, bpm1=None, bpm2=None, sr=44100):
    """
    Takes to audio vectors and calculate the mashability
    :param audio1_vector: Numpy array or similar. Audio of the target excerpt.
    :param audio2_vector: Numpy array or similar. Audio of the candidate excerpt.
    :param sr: Samplerate of the audio. Both audio vectors should have the same samplerate
    :return: A tuple containing: mashability value, the pitch offset, beat offset, harmonic contribution,
    spectral contribution
    """
    c_bsc, c_bss = get_beat_sync_chroma_and_spectrum(audio2_vector, sr=sr, bpm=bpm2)
    base_beat_sync_chroma, base_beat_sync_spec = get_beat_sync_chroma_and_spectrum(audio1_vector, sr=sr, bpm=bpm1)
    c_bsc = np.flip(c_bsc)  # Flip to make correlation, no convolution
    stacked_beat_sync_chroma = np.vstack([c_bsc, c_bsc])
    conv = signal.convolve2d(stacked_beat_sync_chroma, base_beat_sync_chroma, )
    base_n = np.linalg.norm(base_beat_sync_chroma)
    cand_n = np.linalg.norm(c_bsc)
    h_mas = conv / (base_n * cand_n)
    offset = base_beat_sync_chroma.shape[1]-1
    h_mas = np.flip(h_mas[11:-11, offset:-offset], axis=0)
    h_mas_k = np.max(h_mas, axis=0)  # Maximum mashability for each beat displacement

    # 3rd step: Calculate Spectral balance compatibility
    if c_bss.shape[1] >= base_beat_sync_spec.shape[1]:
        beat_length = base_beat_sync_spec.shape[1]
        n_max_b_shifts = c_bss.shape[1] - base_beat_sync_spec.shape[1]
        r_mas_k = np.zeros(n_max_b_shifts+1)
        for i in range(n_max_b_shifts+1):
            beta = np.mean(base_beat_sync_spec + c_bss[:, i:i + beat_length], axis=1)
            beta_norm = beta/np.sum(beta)
            r_mas_k[i] = 1 - np.std(beta_norm)  # Spectral balance for i beat displacement
    else:
        raise ShorterException("Candidate song has lesser beats than base song")
    res_mash = h_mas_k + 0.2 * r_mas_k
    b_offset = np.argmax(res_mash)
    p_shift = np.argmax(h_mas[:, b_offset])
    if p_shift > 6:
        p_shift = 12 - p_shift

    h_contr = h_mas_k[b_offset]
    r_contr = r_mas_k[b_offset]
    return np.max(res_mash), p_shift, b_offset, h_contr, r_contr