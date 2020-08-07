from glob import glob
import essentia.standard as std
import numpy as np
import csv, re
import master_thesis as mt

def zeropad_next_power_2(signal):
    act_len = len(signal)
    new_len = 1<<(act_len-1).bit_length()
    diff_len = new_len - act_len
    return np.pad(signal, (0, diff_len), 'constant')

def get_beat_chunks(filename, bpm_restrict=None):
    audio = std.MonoLoader(filename=filename)()
    hpcp = std.HPCP()
    spectrum = std.Spectrum()
    speaks = std.SpectralPeaks()
    large_speaks = std.SpectralPeaks(maxPeaks=2000)
    tivs = []
    sr = 44100
    bpm = get_tempo(filename)
    tivs_framewise = []
    if bpm_restrict != None and bpm_restrict != bpm:
        raise ValueError
    sec_beat = (60 / bpm)
    beats = np.arange(0, len(audio) / sr, sec_beat)
    beats = np.append(beats, len(audio) / sr)
    for i in range(1, len(beats)):
        segmented_audio = audio[int(beats[i - 1] * sr):int(beats[i] * sr)]
        cutter = std.FrameGenerator(segmented_audio)
        for sec in cutter:
            spec = spectrum(sec)
            freq, mag = speaks(spec)
            chroma = hpcp(freq, mag)
            tivs_framewise.append(chroma)
        np2_seg_audio = zeropad_next_power_2(segmented_audio)
        spec = spectrum(np2_seg_audio)
        freq, mag = speaks(spec)
        chroma = hpcp(freq, mag)
        tivs.append(chroma)

    # Calculate the whole TIV
    np2_whole = zeropad_next_power_2(audio)
    spec = spectrum(np2_whole)
    freq, mag = large_speaks(spec)
    chroma_whole = hpcp(freq, mag)
    return mt.TIVCollection.from_pcp(np.array(tivs).T), mt.TIV.from_pcp(chroma_whole), mt.TIVCollection.from_pcp(np.array(tivs_framewise).T)


def get_number_beats(filename):
    audio = std.MonoLoader(filename=filename)()
    sr = 44100
    bpm = get_tempo(filename)
    sec_beat = (60 / bpm)
    beats = np.arange(0, len(audio) / sr, sec_beat)
    beats = np.append(beats, len(audio) / sr)
    return len(beats)


def get_tempo(filename):
    try:
        bpm = int(re.search(r"(\d+)bpm", filename).group(1))
    except:
        bpm = int(re.search(r"/(\d+)-", filename).group(1))
    return bpm

def main():
    filelist = glob('audios/*.mp3')

    targets = [
        '140-l-0173158-0154547-cmk-hard-piano-140bpm.mp3',
        '140-l-0321618-0117600-dj4kat-susurri-synth-loop.mp3',
        '140-l-0420981-0051273-zatch-trance-pad-2-edit.mp3',
        '140-l-0920952-0134130-bass-never-turn-back-akaleboss.mp3',
        '140-l-0643834-0056295-xray731-trap-muzik-arp.mp3',
        '140-l-0857244-0070940-rojo95-lights-remix-vocal-2.mp3',
        '140-l-0159051-0190162-minor2go-minor2go-type-guitar-alynia.mp3',
        '140-l-0340535-0144544-piano-chord-progression-for-edm-or-dubstep.mp3',
        '140-l-0345547-0055403-cufool-air-synth.mp3',
        '140-l-0480098-0056683-jamievega-stuck-on-flat-wobble.mp3'
    ]

    targets = [ 'audios/' + target for target in targets ]

    for target in targets:
        print("Target:", target)
        print("======================")
        dissonances_beatwise = {}
        dissonances_whole = {}
        dissonances_framewise = {}
        t_beat, t_whole,  t_framewise = get_beat_chunks(target)
        for candidate in filelist:
            print("Candidate:", candidate)
            beatwise_tiv, whole_tiv, framewise_tiv = get_beat_chunks(candidate, bpm_restrict=None)
            dissonances_beatwise[candidate] = t_beat.get_max_compatibility(beatwise_tiv)
            dissonances_framewise[candidate] = t_framewise.get_max_compatibility(framewise_tiv)
            dissonances_whole[candidate] = t_whole.get_max_compatibility(whole_tiv)

        with open('%s_tiv_comptability.csv' % target.replace('audios/', './'), 'w') as file:
            dictcsv = csv.DictWriter(file, ['filename', 'compatibility_framewise', 'pitch_shift_framewise',
                                            'compatibility_beatwise', 'pitch_shift_beatwise',
                                            'compatibility_whole', 'pitch_shift_whole'])
            dictcsv.writeheader()
            for song in dissonances_beatwise.keys():
                dictcsv.writerow({
                'filename': song,
                'compatibility_framewise': dissonances_framewise[song][1],
                'pitch_shift_framewise': dissonances_framewise[song][0],
                'compatibility_beatwise': dissonances_beatwise[song][1],
                'pitch_shift_beatwise': dissonances_beatwise[song][0],
                'compatibility_whole': dissonances_whole[song][1],
                'pitch_shift_whole': dissonances_whole[song][0],
                })


if __name__ == '__main__':
    main()
