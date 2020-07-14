import csv
import glob
import shutil
import sys
from librosa.core import load
import numpy as np
from librosa.output import write_wav
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
        y, sr = load(audio_file_candidate)
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


def main(base_song=None):
    """
    Main function, takes the name of a song and calculate the mashabilities for each song.
    If -p is used, skip the computation of mashability and goes directly to mix the song
    according to the csv generated during the mashability process.
    """
    if (len(sys.argv)) < 2 and (base_song == None):
        print("Usage: python mashability.py <base_song>")
        return
    else:
        if base_song == None:
            base_song = sys.argv[1]
    if '-p' not in sys.argv:
        base_schroma, base_spec= get_beat_sync_chroma_and_spectrum(base_song)
        songs = glob.glob("%s/*.mp3" % base_song.split('/')[0])  # Search for more mp3 files in the target's directory
        mashabilities = {}
        valid_songs = []
        # Calculate mashability for each of the candidate songs
        # Songs containing less beats than the target one will be discarded
        for cand_song in songs:
            try:
                mashabilities[cand_song] = mashability(base_schroma,
                                                       base_spec,
                                                       cand_song)
                valid_songs.append(cand_song)
            except ShorterException as e:
                print("Skipping song %s, because %s" % (cand_song, str(e)))
        # Sort all songs according to their mashabilities
        valid_songs.sort(key=lambda x: mashabilities[x][0], reverse=True)
        top_10 = valid_songs

        # Write the results of the mashabilities in a csv with the same name as the main loop
        with open(base_song.split('/')[-1].replace('.mp3', '.csv'), 'w') as csvfile:
            csvfile.write("file,mashability,pitch_shift,beat_offset,h_contr,r_contr\n")
            for cand_song in top_10:
                out_file = "out_loops/%s" % (cand_song.split('/')[-1])
                csvfile.write("%s,%s,%s,%s,%s,%s\n" % (out_file, mashabilities[cand_song][0], mashabilities[cand_song][1],
                                                 mashabilities[cand_song][2], mashabilities[cand_song][3], mashabilities[cand_song][4]))


def write_songs_mash(base_song):
    # Read the mashabilities results and create the mixes
    results_dir = 'results/mash/%s' % base_song.split('/')[-1].replace('.mp3', '')
    if not os.path.isdir(results_dir):
        try: 
            os.mkdir('results/mash/')
        except:
            pass

        os.mkdir(results_dir)
    with open(base_song.split('/')[-1].replace('.mp3', '.csv'), 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        i = 0
        for row in csv_reader:
            cand_song = row['file']
            pitch_shift = int(row['pitch_shift'])
            beat_offset = int(row['beat_offset'])
            mix = mix_songs(base_song, cand_song, beat_offset, pitch_shift)
            out_file = "%s/%s_MIXED_%s" % (results_dir, row['mashability'], cand_song.split('/')[-1].replace('.mp3', '.wav'))
            # Copy the original candidate to the results folder
            original_cand_copy = "%s/ORIGINAL_%s" % (results_dir, cand_song.split('/')[-1])
            shutil.copyfile(cand_song, original_cand_copy)
            write_wav(out_file, mix, 44100)
            i += 1
            if i == 150:
                break


if __name__ == '__main__':
    main()
