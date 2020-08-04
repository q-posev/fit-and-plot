# fit-and-plot

The fit-and-plot set of script was developed during my PhD at Paul Sabatier University in Toulouse.
It aims to facilitate the post-processing of Trajectory Surface Hopping (TSH) simulations
performed with the deMon-Nano code on High-Performance Computing clusters. 
Nevertheless, these scripts can be easily adapted for TSH results computed with a different software.

##### Currently provided:

- plot-tsh-pops.py : plot averaged populations and fit the initial one with a single exponent
- plot-tsh-occs.py : plot averaged occupations and fit the initand fit the initial one with a single exponent 
[additional option: plot energy gaps (both along the trajectory and averaged) between the initial state and the one below in energy]
- plot-spectra-2in1.py : plot two absorption spectra for comparison within the same energy range
[optional: DFT vs DFTB or DFTB/MIO versus DFTB/MAT]

##### Requirements:
- python 3
- matplotlib
- numpy
- SciPy
- pandas

_**Note: I recommend using virtual environments to avoid compatibility issues.**_

## Clone the repository

```
git clone https://github.com/q-posev/fit_and_plot
cd fit_and_plot
```

You're ready to go!

### Populations example

The QM9 example scripts allows to train and evaluate both SchNet and wACSF neural networks.
The training can be started using:

```
spk_run.py train <schnet/wacsf> qm9 <dbpath> <modeldir> --split num_train num_val [--cuda]
```

where num_train and num_val need to be replaced by the number of training and validation datapoints respectively.

### Occupations example


#### Gaps example


### Spectra example

