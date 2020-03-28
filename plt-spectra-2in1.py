import sys
# check for numpy and matplotlib, try to exit gracefully if not found
import imp
try:
    imp.find_module('numpy')
    foundnp = True
except ImportError:
    foundnp = False
try:
    imp.find_module('matplotlib')
    foundplot = True
except ImportError:
    foundplot = False
if not foundnp:
    print("Numpy is required. Exiting")
    sys.exit()
if not foundplot:
    print("Matplotlib is required. Exiting")
    sys.exit()
import numpy as np
import matplotlib.pyplot as plt

print("The arguments are: " , str(sys.argv))

# initialize settings
molecule1 = ''
molecule2 = ''
dftb_par = 0
spec_type = 0
do_dftb = True
do_dft = False
molecule_list = ['naphthalene', 'anthracene', 'tetracene', 'chrysene',
        'pentacene', 'hexacene', 'heptacene', 'octacene']

# read the arguments and process them
for index, arg in enumerate(sys.argv):    
    if arg in ['--info', '-i']:
        arg_list=['--molecule', '-m']+['--spec_type', '-st']+['--basis', '-b']
        print('Required keywords/arguments: ',arg_list)
        sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--dftb']:
        do_dftb = True
        del sys.argv[index]
        break

for index, arg in enumerate(sys.argv):    
    if arg in ['--dft']:
        do_dft = True
        del sys.argv[index]
        break

only_dftb = (do_dftb and (not do_dft))
print('Only DFTB spectra? {}'.format(only_dftb))

for index, arg in enumerate(sys.argv):    
    if arg in ['--molecule', '-m']:
        if len(sys.argv) > index + 1:
          molecule1 = str(sys.argv[index + 1])
          if molecule1 not in molecule_list:
            print('Available molecules: ', molecule_list)
            print('Add your molecules to the list and rerun')
            sys.exit()
          else:
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the molecule name: (after --molecule or -m keyword)')
            sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--spec_type', '-st']:
        if len(sys.argv) > index + 1:
          spectrum_type = str(sys.argv[index + 1])
          if spectrum_type not in ['convoluted','vertical']:
            print('Available types of spectra: convoluted or vertical')
            sys.exit()
          else:
            if spectrum_type == 'convoluted' :
                spec_type = 1
            else :
                spec_type = 2
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the type of spectra: (after --spec_type or -st keyword)')
            sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--basis', '-b']:
        if len(sys.argv) > index + 1:
          basis = str(sys.argv[index + 1])
          if basis not in ['mat','bio']:
            print('Available DFTB basis: mat or bio')
            sys.exit()
          else:
            if basis == 'mat' :
                dftb_par = 1
            else :
                dftb_par = 2
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the DFTB basis: (after --basis or -b keyword)')
            sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--molecule2', '-m2']:
        if len(sys.argv) > index + 1:
            molecule2 = str(sys.argv[index + 1])
            if molecule2 not in molecule_list:
                print('Available molecules: ', molecule_list)
                print('Add your molecules to the list and rerun')
                sys.exit()
            else:
                del sys.argv[index]
                del sys.argv[index]
            break
        else:
            print('Enter the second molecule name: (after --molecule2 or -m2 keyword)')
            sys.exit()

if dftb_par==0 or spec_type==0 or molecule1=='':
    print('One of the main arguments is missing, use --info or -i argument for more info')
    sys.exit()
# set datafiles
if only_dftb:
    datafile1 = molecule1 + '/tddftb-' + basis + '-spec-' + molecule1[0:5] + '.txt'
    datafile2 = molecule2 + '/tddftb-' + basis + '-spec-' + molecule2[0:5] + '.txt'
else:
    datafile1 = molecule1 + '/tddftb-' + basis + '-spec-' + molecule1[0:5] + '.txt'
    datafile2 = molecule2 + '/tddftb-' + basis + '-spec-' + molecule2[0:5] + '.txt'
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
    sys.exit()
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
print(min(mi),max(ma))

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
font2 = {'family': 'sansserif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
fig = plt.figure()
ax1 = fig.add_subplot(111)
# plot convoluted spectra (with Gaussians by default)
if spec_type == 1 :
    ax1.plot(x,composite/1e4,linewidth=1.2)
    ax1.plot(x,composite2/1e4,linewidth=1.2)
    ax1.set_ylabel('$\epsilon$ [$10^4$ L mol$^{-1}$ cm$^{-1}$]')
    # y limit for convoluted spectra
    ax1.set_ylim(-0.5, 25.0)
# plot vertical (sticks) spectra (based on oscillator strengths)
else :
    ax1.bar(bands, f, width=1.5,edgecolor='None',color='#1f77b4',align='center')
    ax1.bar(bands2, f2, width=1.5,edgecolor='None',color='#ff7f0e',align='center')
    ax1.set_ylabel('Oscillator strength [a.u.]')
    # y limit for vertical spectra
    ax1.set_ylim(0.0, 1.2)

x3 = [1,2,3,4,5,6,7]
ax1.set_xlim(200,400)
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
ax1.legend((molecule1.capitalize(), molecule2.capitalize()),loc='upper right',ncol=1, fancybox=True, shadow=True)
# indicate specific bands in eV (e.g. from experiments) with vertical dashed lines
if molecule1 == 'tetracene' and molecule2 == 'chrysene':
    band1 = 4.51
    band2 = 4.59
ax1.axvline(x=1239.842/band1,color='#1f77b4', linestyle='--')
ax1.axvline(x=1239.842/band2,color='#ff7f0e', linestyle='--')
# output the figure in a given format
dash = '-'
fileformat = '.png'
output_name = molecule1[0:5]+dash+molecule2[0:5]+dash+spectrum_type+'-dftb-'+basis+fileformat

plt.savefig(output_name,dpi=600,bbox_inches='tight')

