from sys import argv,exit
from re import search as reg_search
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
try:
    find_module('pandas')
except ImportError:
    print("Pandas is required")
    exit()
try:
    find_module('scipy')
except ImportError:
    print("Scipy is required")
    exit()

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

print("The arguments are: " , str(argv))
print('')
plt_err = False
prt_mol = False
do_sum = False
l1 = 0
hpc = ''
n_st = 0
init_st = 0
#------------------- SEE THE EXECUTION EXAMPLE BELOW: ----------------#
#------ python this_script_name.py -h s10 -t 100 -n 9 -s 8 --sum 4 ---#
#------ Data from: s10; Trajectories: 100; Number of states: 9; ------#
#-----  Initial state: 8; Sum populations of 4 lowest states    ------#
#---------------------------------------------------------------------#

# read file todolist with list of jobs to be processed
todo = read_csv('todolist',skiprows=0,header=None)
todo.columns = ["job"]

# read arguments and settings
for index, arg in enumerate(argv):    
    if arg in ['--info', '-i']:
        arg_list=['--hpc', '-h']+['--traj_number', '-t']
        arg_list=arg_list+['--n_states', '-n']+['--init_state', '-s']
        print('Required keywords/arguments: ',arg_list)
        print('Optional keywords/arguments:' +
                '[\'--sum\', \'--folder\', \'-f\', \'--plt_err\', \'-pe\', \'--mol_name\', \'-m\']')
        exit()

for index, arg in enumerate(argv):    
    if arg in ['--hpc', '-h']:
        if len(argv) > index + 1:
          hpc = str(argv[index + 1])
          if hpc not in ['s10','oly']:
            print('Available option for HPC: s10, oly')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the HPC name: s10 or oly (after --hpc or -h keyword)')
            exit()

if hpc=='oly':
    for index, arg in enumerate(argv):
        if arg in ['--folder', '-f']:
            if len(argv) > index + 1:
                oly_jobid = str(argv[index + 1])
                if oly_jobid=='':
                    print('Folder from OLYMPE has to be the same as slurm job_id')
                    exit()
                else:
                    del argv[index]
                    del argv[index]
                    break    
            else:
                print('Enter the name of slurm job_id from OLYMPE (after --folder or -f keyword)')
                exit()

for index, arg in enumerate(argv):    
    if arg in ['--traj_number', '-t']:
        if len(argv) > index + 1:
          l1 = int(argv[index + 1])
          if l1 < 1:
            print('Number of trajetories has to be > 0')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the number of trajectories in TSH (after --traj_number or -t keyword)')
            exit()

for index, arg in enumerate(argv):
    if arg in ['--n_states', '-n']:
        if len(argv) > index + 1:
          n_st = int(argv[index + 1])
          if n_st < 2:
            print('Number of states has to be > 1')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the number of states in TSH (after --n_states or -n keyword)')
            exit()
    
for index, arg in enumerate(argv):
    if arg in ['--init_state', '-s']:
        if len(argv) > index + 1:
          init_st = int(argv[index + 1])
          if init_st < 1:
            print('Initial state has to be > 0')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the initial state in TSH (after --init_state or -s keyword)')
            exit()

for index, arg in enumerate(argv):
    if arg in ['--sum']:
        if len(argv) > index + 1:
          pop_to_sum = int(argv[index + 1])
          do_sum = True
          if pop_to_sum > n_st-1:
            print('Too many states to sum')
            exit()
          else:
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the number of lower states to be summed (after --sum keyword)')
            exit()

for index, arg in enumerate(argv):    
    if arg in ['--plt_err', '-pe']:
        plt_err= True
        print('Population will be plotted with error bar')
        del argv[index]
        break

for index, arg in enumerate(argv):    
    if arg in ['--mol_name', '-m']:
        if len(argv) > index + 1:
            mol_name = str(argv[index + 1])
            prt_mol = True
            del argv[index]
            del argv[index]
            break
        else:
            print('Enter the molecule name for the output file (after --mol_name or -m keyword)')

if l1==0 or n_st==0 or init_st==0 or hpc=='':
    print('One of the main arguments is missing, use --info or -i argument for more info')
    exit()

if hpc=='oly':
    print('Data from OLYMPE machine')
if hpc=='s10':
    print('Data from s10 machine')

print('Number of trajectories   = {}'.format(l1))
print('Number of states in TSH  = {}'.format(n_st))
print('Initial state in TSH     = {}\n'.format(init_st))

# initialize the population list
pop_init = 'pop' + str(init_st)
pop_list = []
for k in range(1,n_st+1):
    pop_list.append('pop'+str(k))

# loop over all trajectories/jobs
traj_count=0
for i in range(0,l1):
    # set files with data to be read
    if hpc=='oly':
        pop_file = oly_jobid+str(i) +'.demon' +'/'+'deMon.pop'
    else :
        pop_file = str(todo.job[i])+'/'+'deMon.pop'
    # read populations
    if i==0:
        df0 = read_csv(pop_file,skiprows=0,sep='    ',header=None,engine='python')
        df0.columns = ["time"]+pop_list
        l2 = len(df0.time)
    # accumulate populations
    if i>0:
        df = read_csv(pop_file,skiprows=0,sep='    ',header=None,engine='python')
        df.columns = ["time"]+pop_list
        for population in pop_list:
            df0[population] += df[population]
    
    traj_count += 1

#print("  Number of processed trajectories: {}".format(traj_count))
# average populations over the ensemble of trajectories
for population in pop_list:
    df0[population] /= traj_count

# fitting parameters
A_step = min(df0[pop_init])
A_decay = 1./(1.+A_step)
A_step2 = A_step/(1.+A_step)
# fitting function
def exp_func(x, b):
        return (np.exp(-b*x)+A_step)*A_decay
t=df0.time
fit_pop = np.zeros(l2)
#---------------------------------------------------------------------#
#--------- FIT THE INITIAL STATE POPULATION WITH EXPONENT ------------#
#---------------------------------------------------------------------#
popt,pcov = curve_fit(exp_func,t,df0[pop_init],p0=(0.025))
#---------------------------------------------------------------------#
print('  Decay time of S{0} = {1:.3f} fs \n'.format(init_st,float(1./popt)))
fit_pop = exp_func(t,*popt)
#---------------------------------------------------------------------#
#---------------- STATISTICAL ERROR ESTIMATION -----------------------#
#---------------------------------------------------------------------#
if plt_err:
    fit_max = np.zeros(l2)
    fit_min = np.zeros(l2)
    eps = 0.98/np.sqrt(traj_count)
    tau_fit = 1.0/popt
    tau_max = (1.+eps)*tau_fit
    tau_min = (1.-eps)*tau_fit
    #print(tau_max,tau_min)
    fit_max = exp_func(t,1./tau_max)
    fit_min = exp_func(t,1./tau_min)
#---------------------------------------------------------------------#
#--------------------- MATPLOTLIB SETTINGS ---------------------------#
#---------------------------------------------------------------------#
font = {'size'   : 18}
plt.rc('font', **font)
ax1 = plt.subplot(111)
ax1.grid()
#-------- DO NOT FORGET TO CHANGE THE X-AXIS LIMIT/RANGE BELOW -------#
ax1.set_xlim((0.0, 300.0)) 
ax1.set_ylim((0.0, 1.0))
#---------------------------------------------------------------------#
ax1.minorticks_on()
ax1.tick_params(axis='both',which='minor',length=4,width=1,labelsize=18)
ax1.tick_params(axis='both',which='major',length=8,width=1,labelsize=18)
ax1.set_xlabel('Time [fs]')
ax1.set_ylabel('Population')
#---------------------------------------------------------------------#
#-----------------  PLOT THE POPULATIONS OF INTEREST  ----------------#
#---------------------------------------------------------------------#
if do_sum:
    for i in range(2,pop_to_sum+1):
        pop_add = 'pop'+str(i)
        df0['pop1'] += df0[pop_add]
        pop_list.remove(pop_add)

for population in pop_list:
    ax1.plot(t,df0[population],linewidth=2.0)
ax1.plot(t,fit_pop,dashes=[6, 2],color='black',linewidth=2.0) 
#------------------- PLOT STATISTICAL ERROR --------------------------#
if plt_err:
    ax1.fill_between(t, fit_min, fit_max, facecolor='lightcoral', alpha=0.5)
#---------------------------------------------------------------------#
#---------- GENERATE LEGENDS ACCORDING TO THE PLOTTED POPULATIONS ----#
#---------------------------------------------------------------------#
legend_list=[]
if do_sum:
    legend_sum = '$S_{1-'+str(pop_to_sum)+'}$'
    legend_list.append(legend_sum)
else:
    legend_list.append('$S_1$')
for population in pop_list:
    if population != 'pop1':
        legend_list.append('$S_{}$'.format(int(reg_search(r'\d+', population)[0])))
legend_list.append('$S_{}$ fit'.format(init_st))

ax1.legend(legend_list,loc='upper center', bbox_to_anchor=(0.515, 1.28), ncol=3, fancybox=True, shadow=True) 
#------------------------- SET OUTPUT FILENAME -----------------------#
if prt_mol:
    output_name = mol_name + '-'
else:
    output_name = ''
output_name = output_name + 'pop-{0}traj-initST-{1}-totalST-{2}-tau-{3:.0f}.'.format(l1,init_st,n_st,float(1./popt))
#------------------------- SET OUTPUT FORMAT -------------------------#
fileformat = 'png'
output_name = output_name + fileformat
#------------------------- SAVE THE FIGURE  --------------------------#
print('Output file: {}'.format(output_name))
plt.savefig(output_name, bbox_inches='tight',format=fileformat,dpi=600)

