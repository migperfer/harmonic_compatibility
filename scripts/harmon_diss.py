from harmonic_compatibility.consonance.harmonicity import transform_to_pc, ph_harmon, milne_pc_spectrum
from harmonic_compatibility import utils
from glob import glob
import essentia.standard as std
import numpy as np
from librosa.effects import pitch_shift
import csv
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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
    p_harmonicities_dict = {}
    e_inharmonicies_dict = {}
    e_dissonance_dict = {}
    audio_target = std.MonoLoader(filename=target)()
    print("Target:", target)
    print("======================")
    for candidate in filelist:
        print("Candidate:", candidate)
        audio_candidate = std.MonoLoader(filename=candidate)()
        harmonicities = np.zeros(12)
        inharmonicities = np.zeros(12)
        dissonances = np.zeros(12)
        for pshift in range(12):
            if pshift == 0:
                mod_candidate = audio_candidate
            else:
                if pshift > 5:
                    mod_candidate = pitch_shift(audio_candidate, 44100, pshift - 12).astype(np.float32)
                else:
                    mod_candidate = pitch_shift(audio_candidate, 44100, pshift).astype(np.float32)
            mix_audio = mix(audio_target, mod_candidate, 44100)
            spf, mpf = utils.get_sines_per_frame(mix_audio, 44100)
            hpf, hpm = utils.get_hpeaks_per_frame(mix_audio, 44100)
            pcs = transform_to_pc(spf)
            n_frames = pcs.shape[0]
            har_frame = np.zeros(n_frames)
            inh_frame = np.zeros(n_frames)
            dis_frame = np.zeros(n_frames)
            for i in range(n_frames):
                # Peter's harmonicity part
                mspec = milne_pc_spectrum(pcs[i])
                har_frame[i] = ph_harmon(mspec)

                # Inharmonicity Measure (essentia)
                inh_frame[i] = std.Inharmonicity()(hpf[i], hpm[i])


                # Dissonance Measure (essentia)
                dis_frame[i] = std.Dissonance()(spf[i], mpf[i])

            harmonicities[pshift] = np.mean(har_frame)
            inharmonicities[pshift] = np.mean(inh_frame)
            dissonances[pshift] = np.mean(dis_frame)

        fpitch = np.argmin(harmonicities)
        if fpitch > 5:
            fpitch -= 12
        p_harmonicities_dict[candidate] = (fpitch, np.min(harmonicities))

        fpitch = np.argmin(inharmonicities)
        if fpitch > 5:
            fpitch -= 12
        e_inharmonicies_dict[candidate] = (fpitch, np.min(inharmonicities))

        fpitch = np.argmin(dissonances)
        if fpitch > 5:
            fpitch -= 12
        e_dissonance_dict[candidate] = (fpitch, np.min(dissonances))

    with open('%s_p_harmonicity_comptability.csv' % target.replace('audios/', './'), 'w') as file:
        dictcsv = csv.DictWriter(file, ['filename', 'compatibility_framewise', 'pitch_shift_framewise'])
        dictcsv.writeheader()
        for song in p_harmonicities_dict.keys():
            dictcsv.writerow({
            'filename': song,
            'compatibility_framewise': p_harmonicities_dict[song][1],
            'pitch_shift_framewise': p_harmonicities_dict[song][0],
            })

    with open('%s_inharmonicity_comptability.csv' % target.replace('audios/', './'), 'w') as file:
        dictcsv = csv.DictWriter(file, ['filename', 'compatibility_framewise', 'pitch_shift_framewise'])
        dictcsv.writeheader()
        for song in e_inharmonicies_dict.keys():
            dictcsv.writerow({
            'filename': song,
            'compatibility_framewise': e_inharmonicies_dict[song][1],
            'pitch_shift_framewise': e_inharmonicies_dict[song][0],
            })

    with open('%s_dissonance_comptability.csv' % target.replace('audios/', './'), 'w') as file:
        dictcsv = csv.DictWriter(file, ['filename', 'compatibility_framewise', 'pitch_shift_framewise'])
        dictcsv.writeheader()
        for song in e_dissonance_dict.keys():
            dictcsv.writerow({
            'filename': song,
            'compatibility_framewise': e_dissonance_dict[song][1],
            'pitch_shift_framewise': e_dissonance_dict[song][0],
            })