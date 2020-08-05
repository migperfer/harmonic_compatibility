from harmonic_compatibility.consonance.roughness import gebhardt_dissonance
from glob import glob
import essentia.standard as std
import numpy as np
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
    print("Target:", target)
    print("======================")
    dissonances = []
    audioloader = std.MonoLoader(filename=target)
    target_audio = audioloader()
    for candidate in filelist:
        print("Candidate:", candidate)
        audioloader.configure(filename=candidate)
        candidate_audio = audioloader()
        ov, fwise= gebhardt_dissonance(target_audio, candidate_audio)
        dissonances.append((ov, fwise))

    ov_diss = np.array([song[0] for song in dissonances])
    p_shift = np.argmin(ov_diss, axis=1) - 48
    dissonance = np.min(ov_diss, axis=1)

    with open('%s_dissonances.csv' % target.replace('audios/', './'), 'w') as file:
        dictcsv = csv.DictWriter(file, ['filename', 'pitch', 'dissonance'])
        dictcsv.writeheader()
        for candidate_idx in range(len(filelist)):
            dictcsv.writerow({'filename': filelist[candidate_idx],
                            'pitch': p_shift[candidate_idx],
                            'dissonance': [dissonance[candidate_idx]]})