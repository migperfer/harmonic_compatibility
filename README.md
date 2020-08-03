#Harmonic compatibility measures

This repository contains the code used for my Master Thesis _Harmonic Compatibility for loops in electronic music_.

## Algorithms classification

![Image with the algorithm tree](media/algorithms_tree.png)
We splitted the package in a similar fashion:

```text
harmonic_compatibility
|
|
------- consonance 
|       |
|       ---------- roughness
|       |          |__________ Plompt & Levelt (pl)
|       |          |__________ Hutchinson & Knopoff (hk)
|       |          |__________ Gebhardt et al. (r_diss)
|       |          |__________ Hutchinson & Knopoff (hk_nocuda)
|       |          |__________ Gebhardt et al. (hk_nocuda)
|       |
|       ---------- harmonicity
|                  |__________ Essentia's Inharmonicity (inharmonicity)
|                  |__________ Harrison & pearce (p_harmon)
|
------- similarity
        |__________ automashupper
        |__________ tiv
|
|
|
------- utils
``` 
The *_nocuda* are versions of the algorithms that *doesn't requiere CUDA* to work. Notice that this versions were
not used in our experiments and still under development. We include those algorithms 

 
## Utils

The utils subpackage contains functions that helps to recreate the methodology used in our work.
The functions allows to mix two audios where both tracks have equal loudness, also has functions that given
a _.csv_ with the compatibilities for one loops, create the top _n_ best mixes. 


## Scripts folder

Here are located the scripts that I used our my work to calculate harmonic compatibility for all the possible algorithms.
They work by setting up different _target loops_, and a directory where all the _candidate loops_ are located.

## Requirements
This repository needs:
* essentia~=2.1b5
* numpy~=1.16.2
* pandas~=0.25.1
* pyrubberband~=0.3.0
* matplotlib~=3.0.2
* setuptools~=41.6.0
* joblib~=0.14.1
* scipy~=1.1.0
* librosa~=0.7.2
* madmom~=0.16.1
* pydub~=0.23.0

And also for _Hutchinson & Knopoff_ and _Gebhardt et al._, CUDA is used to speed up the calculation.
So aditionally for those algorithms you will need:
- CUDA 
- PyCuda (You must install PyCuda according to your CUDA version)

##  How to use it