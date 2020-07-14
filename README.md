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
|       |
|       ---------- harmonicity
|                  |__________ Essentia's Inharmonicity (inharmonicity)
|                  |__________ Harrison & pearce (p_harmon)
|
------- similarity
        |__________ automashupper
        |__________ tiv
``` 

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