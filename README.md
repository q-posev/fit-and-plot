# fit-and-plot

The fit-and-plot set of scripts aims to facilitate the post-processing of TD-DFT(B) and Trajectory Surface Hopping (TSH) simulations
performed with the deMon-Nano code on High-Performance Computing clusters. 

However, these scripts can be easily adapted for any other software.

#### Currently provided:

- plot-tsh-pops.py : plot averaged populations and fit the initial one with a single exponent
- plot-tsh-occs.py : plot averaged occupations and fit the initial one with a single exponent 
- plot-spectra-2in1.py : plot two absorption spectra for comparison

#### Requirements:
- python 3
- matplotlib
- numpy
- scipy
- pandas

_**Note: I recommend using virtual environments to avoid compatibility issues.**_

## Clone the repository

```
git clone https://github.com/q-posev/fit_and_plot
cd fit_and_plot
```

You're ready to go!

## Populations example

The script can be launched using:

```
python plot-tsh-pops.py -h oly -f 666/ -t 100 -n 9 -s 8 --sum 4 -m chrysene
```

#### Required arguments:
- -h is the name of the HPC machine (s10 or oly)
- -f is the name of the folder with TSH results (not required for s10)
- -t is the number of trajectories to analyze 
- -n is the total number of excited states in the propagation
- -s is the initially excited state (its population will be exponentially fitted)

_**Note: You need to have a todolist file (with ID-s of trajectories) in the same directory as the script. This is particularly important for results from s10 cluster.**_

#### Optional arguments:
- --sum can be used if accumulated population has to be plotted for low-lying states (in this example, accumulated population of 4 lowest states will be plotted)
- -m or --mol_name or  is the name of the molecule (will be added to the output file name)
- -pe or --plt_err can be used to plot statistical error for the fitted population
- -i or --info to output the full list of required and optional arguments

## Occupations example

```
python plot-tsh-occs.py -h s10 -t 100 -n 9 -s 8 --sum 4 -m chrysene --gaps
```

All arguments are the same as in previous (populations) case. 

One additional argument (--gaps or -g) can be used to plot the energy gap 
between the intially excited state and the one below in energy 
along each trajectory together with the averaged (over an entire ensemble) value.

## Spectra example

TODO
