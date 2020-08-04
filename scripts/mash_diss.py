import master_thesis as mt
from glob import glob
import essentia.standard as std
import numpy as np
import csv


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
    bpm = 140
    for target in targets:
        print("Target:", target)
        print("======================")
        dissonances = {}
        audioloader = std.MonoLoader(filename=target)
        target_audio = audioloader()
        for candidate in filelist:
            print("Candidate:", candidate)
            audioloader.configure(filename=candidate)
            candidate_audio = audioloader()
            dissonances[candidate] = mt.get_mashability(target_audio, candidate_audio, bpm, bpm)

        with open('%s_mashabilities.csv' % target.replace('audios/', './'), 'w') as file:
            dictcsv = csv.DictWriter(file, ['filename', 'mashability', 'pitch_offset', 'beat_offset', 'h_contr', 'r_contr'])
            dictcsv.writeheader()
            for song in dissonances.keys():
                dictcsv.writerow({
                'filename': song,
                'mashability': dissonances[song][0],
                'pitch_offset': dissonances[song][1],
                'beat_offset': dissonances[song][2],
                'h_contr': dissonances[song][3],
                'r_contr': dissonances[song][4],
                })


if __name__ == '__main__':
    main()