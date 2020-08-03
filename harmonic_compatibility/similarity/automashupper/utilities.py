import numpy as np
import essentia.standard as estd
from pyrubberband import pyrb


def match_target_amplitude(sound, target_dBFS=0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def zapata14bpm(y):
    essentia_beat = estd.BeatTrackerMultiFeature()
    mean_tick_distance = np.mean(np.diff(essentia_beat(y)[0]))
    return 60/mean_tick_distance


def self_tempo_estimation(y, sr, tempo=None):
    """
    A function to calculate tempo based on a confidence measure
    :param y: The audio signal to which calculate the tempo
    :param sr: The sample rate of the signal
    :param tempo: Precalculated bpm
    :return: An array containing tempo, and an array of beats (in seconds)
    """
    if tempo is None:
        confidence_estimator = estd.LoopBpmConfidence(sampleRate=sr)
        percivalbpm = int(estd.PercivalBpmEstimator(sampleRate=sr)(y))
        try:
            zapatabpm = int(zapata14bpm(y))
        except:
            tempo = percivalbpm
        else:
            confidence_zapata = confidence_estimator(y, zapatabpm)
            confidence_percival = confidence_estimator(y, percivalbpm)
            if confidence_percival >= confidence_zapata:
                tempo = percivalbpm
            else:
                tempo = zapatabpm
    sec_beat = (60/tempo)
    beats = np.arange(0, len(y)/sr, sec_beat)
    return tempo, beats


def rotate_audio(audio, sr, n_beats):
    """
    Apply rotation to the audio in a given number of beats
    :param audio: The audio signal to rotate
    :param sr: The sample rate
    :param n_beats: Number of beats to rotate the audio
    :return:
    """
    tempo, _ = self_tempo_estimation(y, sr)
    samples_rotation = tempo * sr
    n_rotations = int(samples_rotation * n_beats)
    return np.roll(audio, n_rotations)


def adjust_tempo(song, final_tempo):
    """
    Adjust audio to the desired tempo
    :param song: The song which tempo should be adjusted
    :param final_tempo:
    :return:
    """
    actual_tempo, _ = self_tempo_estimation(song, 44100)
    song = pyrb.change_tempo(song, 44100, actual_tempo, final_tempo)
    return song

def mix_songs(main_song, cand_song, beat_offset, pitch_shift):
    """
    Mixes two loops with a given beat_offset and a pitch_shift (applied to the candidate song)
    :param main_song: The path to the main loop
    :param cand_song: The path to the candidate loop
    :param beat_offset: The beat offset
    :param pitch_shift: The pitch shift
    :return: The resulting signal of the audio mixing with sr=44100
    """
    sr = 44100
    main_song = estd.MonoLoader(filename=main_song)
    cand_song = estd.MonoLoader(filename=cand_song)
    #Make everything mono
    final_tempo, _ = self_tempo_estimation(main_song, sr)
    final_len = len(main_song)

    beat_sr = final_tempo/(60 * sr)  # Number of samples per beat
    cand_song = cand_song[int(beat_offset*beat_sr):int(beat_offset*beat_sr + final_len)]
    tunning = np.mean(estd.TuningFrequencyExtractor()(cand_song))
    tunning_main = np.mean(estd.TuningFrequencyExtractor()(main_song))
    cand_song = adjust_tempo(cand_song, final_tempo)
    factor_tuning = tunning/tunning_main
    pitch_factor = factor_tuning*np.exp2(-pitch_shift/12)
    cand_song = pyrb.frequency_multiply(cand_song, 44100, pitch_factor)
    cand_song = estd.Resample(  inputSampleRate=44100, 
                                outputSampleRate=44100 / cand_song.shape[0] * main_song.shape[0],
                                quality=0)(cand_song)
    try:
        aux = np.zeros(main_song.shape[0])
        aux[:cand_song.shape[0]] = cand_song
        cand_song = aux
    except ValueError:
        aux = np.zeros(cand_song.shape[0])
        aux[:main_song.shape[0]] = main_song
        main_song = aux
    cand_song = cand_song.astype('float32')
    # main_song_replaygain = estd.ReplayGain()(main_song)
    # cand_song = estd.EqloudLoader(replayGain=main_song_replaygain)(cand_song)
    cand_song = cand_song/max(cand_song)
    main_song = main_song/max(main_song)
    return cand_song*0.5 + main_song*0.5

