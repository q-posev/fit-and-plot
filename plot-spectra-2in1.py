from sys import argv,exit
from imp import find_module
# perform sanity check of required modules
try:
    find_module('numpy')
except ImportError:
    print("Numpy is required")
    exit()
try:
    find_module('matplotlib')
except ImportError:
    print("Matplotlib is required")
    exit()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#===================================================================================================#
# This scrupt is designed to plot UV absorption spectra computed with TD-DFT(B) 
# The spectra have to be written in the .txt file where first line is a header with some comments
# 1-st column are energies of singlet excited states in eV
# 2-nd column are oscillator strengths (dimensionless)
# Resulting spectrum can be (see -st or --spec_type keywords)
# (i)  vertical (sticks);
# (ii) convoluted (with Gaussian or Lorentzian functions);

#               Example of an execution line is presented below:
# python this_script_name.py --dft -f blyp --dftb -b bio -m phenanthrene -st vertical

# -f stands for a functional used in TD-DFT (if --dft keyword is activated)
# -b stands for a basis (parameters set) used in TD-DFTB (if --dftb is activated)
# -m stands for a molecule name 
# (spectra can be plotted for one molecule (DFT vs DFTB) or for two molecules, see -m2 keyword below)
# (spectra can be plotted with a zoomed inset, see -z keyword but legend color has to be fixed)
#===================================================================================================#
print("The arguments are: " , str(argv))
# initialize settings
molecule1 = ''
molecule2 = ''
dftb_par = 0
spec_type = 0
do_dftb = True
do_dft = False
do_zoom = False
molecule_list = ['naphthalene', 'anthracene', 'phenanthrene', 'tetracene', 
        'chrysene', 'pentacene', 'hexacene', 'heptacene', 'octacene']

# read the arguments and process them
for index, arg in enumerate(argv):    
    if arg in ['--info', '-i']:
        arg_list=['--molecule', '-m']+['--spec_type', '-st']+['--basis', '-b']
        opt_list=['--molecule2', '-m2']+['--dft', '-dft']+['--functional', '-f']+['--dftb', '-dftb']+['--zoom','-z']
        print('Required keywords/arguments: ',arg_list)
        print('Optional keywords/arguments: ',opt_list)
        exit()

for index, arg in enumerate(argv):    
    if arg in ['--dftb','-dftb']:
        do_dftb = True
        del argv[index]
        break

for index, arg in enumerate(argv):    
    if arg in ['--dft','-dft']:
        do_dft = True
        del argv[index]
        break

for index, arg in enumerate(argv):    
    if arg in ['--zoom','-z']:
        do_zoom = True
        del argv[index]
        break

only_dftb = (do_dftb and (not do_dft))
print('Only DFTB spectra? {}'.format(only_dftb))

for index, arg in enumerate(argv):    
    if arg in ['--molecule', '-m']:
        if len(argv) > index + 1:
          molecule1 = str(argv[index + 1])
          if molecule1 not in molecule_list:
            print('Available molecules: ', molecule_list)
            print('Add your molecules to the list and rerun')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the molecule name: (after --molecule or -m keyword)')
            exit()

for index, arg in enumerate(argv):    
    if arg in ['--spec_type', '-st']:
        if len(argv) > index + 1:
          spectrum_type = str(argv[index + 1])
          if spectrum_type not in ['convoluted','vertical']:
            print('Available types of spectra: convoluted or vertical')
            exit()
          else:
            if spectrum_type == 'convoluted' :
                spec_type = 1
            else :
                spec_type = 2
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the type of spectra: (after --spec_type or -st keyword)')
            exit()

for index, arg in enumerate(argv):    
    if arg in ['--basis', '-b']:
        if len(argv) > index + 1:
          basis = str(argv[index + 1])
          if basis not in ['mat','bio']:
            print('Available DFTB basis: mat or bio')
            exit()
          else:
            if basis == 'mat' :
                dftb_par = 1
            else :
                dftb_par = 2
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the DFTB basis: (after --basis or -b keyword)')
            exit()

for index, arg in enumerate(argv):    
    if arg in ['--molecule2', '-m2']:
        if len(argv) > index + 1:
            molecule2 = str(argv[index + 1])
            if molecule2 not in molecule_list:
                print('Available molecules: ', molecule_list)
                print('Add your molecules to the list and rerun')
                exit()
            else:
                del argv[index]
                del argv[index]
            break
        else:
            print('Enter the second molecule name: (after --molecule2 or -m2 keyword)')
            exit()
# sanity check
if dftb_par==0 or spec_type==0 or molecule1=='':
    print('One of the main arguments is missing, use --info or -i argument for more info')
    exit()
# this is to extract the name of the DFT functional
# note: DFT vs DFTB spectra should be plotted for the same molecule (molecule1 be default)
if do_dft:
    for index, arg in enumerate(argv):    
        if arg in ['--functional', '-f']:
            if len(argv) > index + 1:
                dft_functional = str(argv[index + 1])
                del argv[index]
                del argv[index]
                break
            else:
                print('Enter the DFT functional: (after --functional or -f keyword)')
                exit()
# set datafiles
if only_dftb:
    datafile1 = molecule1 + '/tddftb-' + basis + '-' + molecule1[0:5] + '.txt'
    datafile2 = molecule2 + '/tddftb-' + basis + '-' + molecule2[0:5] + '.txt'
else:
    datafile1 = molecule1 + '/tddftb-' + basis + '-' + molecule1[0:5] + '.txt'
    datafile2 = molecule1 + '/tddft-' + dft_functional + '-' + molecule1[0:5] + '.txt'
# load the data
inp=np.loadtxt(datafile1, delimiter=' ')
inp2=np.loadtxt(datafile2, delimiter=' ')
# initialize parameters
l0 = len(inp)
bands = np.zeros(l0)
f = np.zeros(l0)
l2 = len(inp2)
bands2 = np.zeros(l2)
f2 = np.zeros(l2)
# sqrt(2) * standard deviation of 0.4 eV is 3099.6 nm. 0.1 eV is 12398.4 nm. 0.2 eV is 6199.2 nm.
stdev = 12398.4
# for Lorentzians, gamma is half bandwidth at half peak height (nm)
gamma = 12.5
# transform excitation energies from eV to nm
bands = 1239.84193/inp[0:l0-1,0]
bands2 = 1239.84193/inp2[0:l2-1,0]

# oscillator strengths (dimensionless)
f = inp[0:l0-1,1]
f2 = inp2[0:l2-1,1]

# adjust the following variables to change the area of the spectrum that is plotted 
mi = np.zeros(3)
ma = np.zeros(3)

mi[0] = min(bands)
mi[1] = min(bands2)
ma[0] = max(bands)
ma[1] = max(bands2)

start=min(mi)-50.0
finish=max(ma)+50.0
points=10000

# basic check that we have the same number of bands and oscillator strengths
if len(bands) != len(f):
    print('Number of bands does not match the number of oscillator strengths.')
    exit()
# information on producing spectral curves (Gaussian and Lorentzian) is adapted from:
# P. J. Stephens, N. Harada, Chirality 22, 229 (2010).
# Gaussian curves are often a better fit for UV/Vis.
def gaussBand(x, band, strength, stdev):
    "Produces a Gaussian curve"
    bandshape = 1.3062974e8 * (strength / (1e7/stdev))  * np.exp(-(((1.0/x)-(1.0/band))/(1.0/stdev))**2)
    return bandshape
def lorentzBand(x, band, strength, stdev, gamma):
    "Produces a Lorentzian curve"
    bandshape = 1.3062974e8 * (strength / (1e7/stdev)) * ((gamma**2)/((x - band)**2 + gamma**2))
    return bandshape

x = np.linspace(start,finish,points)
mi[0] = min(inp[:,0])
mi[1] = min(inp2[:,0])
ma[0] = max(inp[:,0])
ma[1] = max(inp2[:,0])

start2=min(mi)-1239.84193/50.0
finish2=max(ma)+1239.84193/50.0
x2 = np.linspace(start2,finish2,points)

# convolute the data
composite = 0
for count,peak in enumerate(bands):
    thispeak = gaussBand(x, peak, f[count], stdev)
#    thispeak = lorentzBand(x, peak, f[count], stdev, gamma)
    composite += thispeak
composite2 = 0
for count2,peak2 in enumerate(bands2):
    thispeak2 = gaussBand(x, peak2, f2[count2], stdev)
#    thispeak = lorentzBand(x, peak, f[count], stdev, gamma)
    composite2 += thispeak2

# matplotlib setting
font = {'size'   : 18}
plt.rc('font', **font)
font2 = {'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

ax1 = plt.subplot(111)
# additional part for a zoomed-in plot
if do_zoom:
    zoom_factor = 2.0
    axins = zoomed_inset_axes(ax1, zoom_factor, loc=4)
    plt.xticks(visible=False)
    #plt.yticks(visible=False)
    plt.rc('ytick', labelsize=8)     # labelsize of the y ticks
    plt.yticks(fontsize=12)          # fontsize of the y ticks

if do_zoom:
    # specify and apply the limits for zoomed inset
    if spec_type == 1:
        x1, x2, y1, y2 = 230, 250, -0.1, 6.0 
    else :
        x1, x2, y1, y2 = 210, 230, 0.0, 0.4 
    axins.set_xlim(x1, x2) 
    axins.set_ylim(y1, y2) 
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# plot convoluted spectra (with Gaussians by default)
if spec_type == 1 :
    ax1.plot(x,composite/1e4,linewidth=1.2)
    ax1.plot(x,composite2/1e4,linewidth=1.2)
    ax1.set_ylabel('$\epsilon$ [$10^4$ L mol$^{-1}$ cm$^{-1}$]')
    # y limit for convoluted spectra
    ax1.set_ylim(-0.5, 15.0)
    if do_zoom:
        axins.plot(x,composite/1e4,linewidth=1.2)
        axins.plot(x,composite2/1e4,linewidth=1.2)

# plot vertical (sticks) spectra (based on oscillator strengths)
else :
    ax1.bar(bands, f, width=1.5,edgecolor='None',align='center')
    ax1.bar(bands2, f2, width=1.5,edgecolor='None',align='center')
    ax1.set_ylabel('Oscillator strength')
    # y limit for vertical spectra
    ax1.set_ylim(0.0, 0.8)
    if do_zoom:
        axins.bar(bands, f, width=1.5,edgecolor='None',align='center')
        axins.bar(bands2, f2, width=1.5,edgecolor='None',align='center')

x3 = [1,2,3,4,5,6,7]
ax1.set_xlim(200,370)
ax1.set_xlabel('$\lambda$ [nm]')
# create a second X-axis in eV and put it on top
nm2eV = lambda t: 1239.84193/t
newpos = [nm2eV(t) for t in x3]
ax2 = ax1.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(x3)
ax2.set_xlabel('$E$ [eV]')
ax2.set_xlim(ax1.get_xlim())
# set ticks
ax1.minorticks_on()
ax1.tick_params(axis='both',which='minor',length=4,width=1,labelsize=18)
ax1.tick_params(axis='both',which='major',length=8,width=1,labelsize=18)
# set legend
if only_dftb:
    ax1.legend((molecule1.capitalize(), molecule2.capitalize()),
            loc='upper right',ncol=1, fancybox=True, shadow=True)
else :
#    plt.title('{} DFT versus DFTB'.format(molecule1.capitalize()), y=1.16, fontdict=font2)
    ax1.legend(('DFTB ({})'.format(basis.upper()), 'DFT ({})'.format(dft_functional.upper())),
            loc='upper right',ncol=1, fancybox=True, shadow=True)
# indicate specific bands in eV (e.g. from experiments) with vertical dashed lines
if molecule1 == 'tetracene' and molecule2 == 'chrysene':
    band1 = 4.51
    band2 = 4.59
    ax1.axvline(x=1239.842/band1,color='#1f77b4', linestyle='--')
    ax1.axvline(x=1239.842/band2,color='#ff7f0e', linestyle='--')
if molecule1 == 'phenanthrene':
    # from Halasinki 2004
    bands_matrix = [341.1,284.3,273.4,262.4,243.0,229.0]
    bands_jet = [340.9,282.6]
    [ax1.axvline(_x, linestyle='--', color='green') for _x in bands_matrix]
    [ax1.axvline(_x, linestyle='--', color='red') for _x in bands_jet]
# output the figure in a given format
fileformat = 'eps'
if only_dftb:
    output_name ='{0}-{1}-{2}-dftb-{3}.{4}'.format(molecule1[0:5],molecule2[0:5],spectrum_type,basis,fileformat)
else:
    output_name = '{0}-{1}-dftb-{2}-dft-{3}.{4}'.format(molecule1[0:5],spectrum_type,basis,dft_functional,fileformat)

if do_zoom:
   temp_name = output_name[0:len(output_name)-len(fileformat)-1]
   output_name = temp_name + '-wZOOM.' + fileformat

print('Output file: {}'.format(output_name))
plt.savefig(output_name,dpi=600,format=fileformat,bbox_inches='tight')

