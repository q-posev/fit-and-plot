import sys
# perform sanity check of required modules
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
try:
    imp.find_module('pandas')
    foundpd = True
except ImportError:
    foundpd = False
try:
    imp.find_module('scipy')
    foundsci = True
except ImportError:
    foundsci = False
if not foundnp:
    print("Numpy is required. Exiting")
    sys.exit()
if not foundplot:
    print("Matplotlib is required. Exiting")
    sys.exit()
if not foundpd:
    print("Pandas is required. Exiting")
    sys.exit()
if not foundsci:
    print("Scipy is required. Exiting")
    sys.exit()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
print('')
plt_err = False
prt_mol = False
l1 = 0
hpc = ''
n_st = 0
init_st = 0
#------------------- SEE THE EXECUTION EXAMPLE BELOW: ----------------#
#--------   python this_script_name.py -h s10 -t 100 -n 9 -s 8 -pe ---#
#------ Data from: s10; Trajectories: 100; Number of states: 9; ------#
#-----      Initial state: 8; Plot WITH error bar (-pe option)  ------#
#---------------------------------------------------------------------#

# read file todolist with list of jobs to be processed
todo = pd.read_csv('todolist',skiprows=0,header=None)
todo.columns = ["job"]

# read arguments and settings
for index, arg in enumerate(sys.argv):    
    if arg in ['--info', '-i']:
        arg_list=['--hpc', '-h']+['--traj_number', '-t']
        arg_list=arg_list+['--n_states', '-n']+['--init_state', '-s']
        print('Required keywords/arguments: ',arg_list)
        print('Optional keywords/arguments: [\'--folder\', \'-f\', \'--plt_err\', \'-pe\', \'--mol_name\', \'-m\']')
        sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--hpc', '-h']:
        if len(sys.argv) > index + 1:
          hpc = str(sys.argv[index + 1])
          if hpc not in ['s10','oly']:
            print('Available option for HPC: s10, oly')
            sys.exit()
          else:
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the HPC name: s10 or oly (after --hpc or -h keyword)')
            sys.exit()

if hpc=='oly':
    for index, arg in enumerate(sys.argv):
        if arg in ['--folder', '-f']:
            if len(sys.argv) > index + 1:
                oly_jobid = str(sys.argv[index + 1])
                if oly_jobid=='':
                    print('Folder from OLYMPE has to be the same as slurm job_id')
                    sys.exit()
                else:
                    del sys.argv[index]
                    del sys.argv[index]
                    break    
            else:
                print('Enter the name of slurm job_id from OLYMPE (after --folder or -f keyword)')
                sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--traj_number', '-t']:
        if len(sys.argv) > index + 1:
          l1 = int(sys.argv[index + 1])
          if l1 < 1:
            print('Number of trajetories has to be > 0')
            sys.exit()
          else:
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the number of trajectories in TSH (after --traj_number or -t keyword)')
            sys.exit()

for index, arg in enumerate(sys.argv):
    if arg in ['--n_states', '-n']:
        if len(sys.argv) > index + 1:
          n_st = int(sys.argv[index + 1])
          if n_st < 2:
            print('Number of states has to be > 1')
            sys.exit()
          else:
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the number of states in TSH (after --n_states or -n keyword)')
            sys.exit()
    
for index, arg in enumerate(sys.argv):
    if arg in ['--init_state', '-s']:
        if len(sys.argv) > index + 1:
          init_st = int(sys.argv[index + 1])
          if init_st < 1:
            print('Initial state has to be > 0')
            sys.exit()
          else:
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the initial state in TSH (after --init_state or -s keyword)')
            sys.exit()

for index, arg in enumerate(sys.argv):    
    if arg in ['--plt_err', '-pe']:
        plt_err= True
        print('Population will be plotted with error bar')
        del sys.argv[index]
        break

for index, arg in enumerate(sys.argv):    
    if arg in ['--mol_name', '-m']:
        if len(sys.argv) > index + 1:
            mol_name = str(sys.argv[index + 1])
            prt_mol = True
            del sys.argv[index]
            del sys.argv[index]
            break
        else:
            print('Enter the molecule name that will go to the the output file name (after --mol_name or -m keyword)')

if l1==0 or n_st==0 or init_st==0 or hpc=='':
    print('One of the main arguments is missing, use --info or -i argument for more info')
    sys.exit()

if hpc=='oly':
    print('Data from OLYMPE machine')
if hpc=='s10':
    print('Data from s10 machine')

print('Number of trajectories   = '+str(l1))
print('Number of states in TSH  = '+str(n_st))
print('Initial state in TSH     = '+str(init_st))

# initialize the population list
pop_list=[]
for k in range(1,n_st+1):
    pop_list.append('pop'+str(k))

# loop over all trajectories/jobs
j=0
for i in range(0,l1):

    # filename setting
    if hpc=='oly':
        pop_file = oly_jobid+str(i) +'.demon' +'/'+'deMon.pop'
    else :
        pop_file = str(todo.job[i])+'/'+'deMon.pop'
    
    # read populations
    if i==0 :
        df0 = pd.read_csv(pop_file,skiprows=0,sep='    ',header=None,engine='python')
        df0.columns = ["time"]+pop_list
        l2 = len(df0.time)

    # accumulate populations
    if i>0 :
        df = pd.read_csv(pop_file,skiprows=0,sep='    ',header=None,engine='python')
        df.columns = ["time"]+pop_list

        df0.pop1 = df0.pop1 + df.pop1
        df0.pop2 = df0.pop2 + df.pop2
        df0.pop3 = df0.pop3 + df.pop3
        df0.pop4 = df0.pop4 + df.pop4
        df0.pop5 = df0.pop5 + df.pop5
        df0.pop6 = df0.pop6 + df.pop6
        df0.pop7 = df0.pop7 + df.pop7
        df0.pop8 = df0.pop8 + df.pop8
        #df0.pop9 = df0.pop9 + df.pop9
        #df0.pop10 = df0.pop10 + df.pop10
        #df0.pop11 = df0.pop11 + df.pop11
        #df0.pop12 = df0.pop12 + df.pop12
        #df0.pop13 = df0.pop13 + df.pop13
        #df0.pop14 = df0.pop14 + df.pop14
        #df0.pop15 = df0.pop15 + df.pop15
    
    j+=1


print("Processed jobs: "+str(j))

# compute averaged population
df0.pop1 = df0.pop1/j
df0.pop2 = df0.pop2/j
df0.pop3 = df0.pop3/j
df0.pop4 = df0.pop4/j
df0.pop5 = df0.pop5/j
df0.pop6 = df0.pop6/j
df0.pop7 = df0.pop7/j
df0.pop8 = df0.pop8/j
#df0.pop9 = df0.pop9/j
#df0.pop10 = df0.pop10/j
#df0.pop11 = df0.pop11/j
#df0.pop12 = df0.pop12/j
#df0.pop13 = df0.pop13/j
#df0.pop14 = df0.pop14/j
#df0.pop15 = df0.pop15/j

#---------------------------------------------------------------------#
#--------- REPLACE df0.popX BELOW WITH df0.popINIT_STATE -------------#
#---------------------------------------------------------------------#
A_step = min(df0.pop7)
#---------------------------------------------------------------------#
# fitting parameters
A_decay = 1./(1.+A_step)
A_step2 = A_step/(1.+A_step)
#print(A_step)
# fitting function
def exp_func(x, b):
        return (np.exp(-b*x)+A_step)*A_decay
t=df0.time
fit_s2 = np.zeros(l2)
#---------------------------------------------------------------------#
#--------- REPLACE df0.popX BELOW WITH df0.popINIT_STATE -------------#
#---------------------------------------------------------------------#
# do the fitting
popt,pcov = curve_fit(exp_func,t,df0.pop7,p0=(0.025))
#---------------------------------------------------------------------#
print('S'+str(init_st)+' time decay =', (1.0/popt),'fs')
fit_s2 = exp_func(t,*popt)
#---------------------------------------------------------------------#
#---------------- STATISTICAL ERROR ESTIMATION -----------------------#
#---------------------------------------------------------------------#
if plt_err:
    fit_max = np.zeros(l2)
    fit_min = np.zeros(l2)

    eps = 0.98/np.sqrt(l1)
    tau_fit = 1.0/popt
    tau_spec = eps*1.0/popt 
    print(tau_spec, tau_fit)

    t_max = tau_fit+tau_spec
    t_min = tau_fit-tau_spec
    print(t_max,t_min)

    fit_max = exp_func(t,1./t_max)
    fit_min = exp_func(t,1./t_min)
#---------------------------------------------------------------------#
#------------------------- MATPLOTLIB SETTINGS -----------------------#
#---------------------------------------------------------------------#
font = {'size'   : 18}
plt.rc('font', **font)
fig = plt.figure()
plt.grid(True)
ax1 = fig.add_subplot(111)
ax1.set_xlim((0.0, 300.0)) 
ax1.set_ylim((0.0, 1.0))
ax1.minorticks_on()
ax1.tick_params(axis='both',which='minor',length=4,width=1,labelsize=18)
ax1.tick_params(axis='both',which='major',length=8,width=1,labelsize=18)
ax1.set_xlabel('Time [fs]')
ax1.set_ylabel('Population')
#---------------------------------------------------------------------#
#------ MODIFY THE PART BELOW TO PLOT THE POPULATIONS OF INTEREST  ---#
#---------------------------------------------------------------------#
ax1.plot(t,df0.pop1+df0.pop2+df0.pop3+df0.pop4,t,df0.pop5,t,df0.pop6,t,df0.pop7,t,df0.pop8,linewidth=2.0)
#---------------------------------------------------------------------#
ax1.plot(t,fit_s2,dashes=[6, 2],color='black',linewidth=2.0) 
#------------------- PLOT STATISTICAL ERROR --------------------------#
if plt_err:
    ax1.fill_between(t, fit_min, fit_max, facecolor='lightcoral', alpha=0.5)
#---------------------------------------------------------------------#
ax1.legend(('$S_{1-4}$', '$S_{5}$', '$S_{6}$', '$S_{7}$', '$S_{8}$', '$S_{7}$ fit'),loc='upper center', bbox_to_anchor=(0.515, 1.03),ncol=3, fancybox=True, shadow=True) 
#------------------------- SET OUTPUT FILENAME -----------------------#
if prt_mol:
    filename = mol_name+'_pop_'+'traj'+str(l1)+'_init_st'+str(init_st)+'_total_st'+str(n_st)
else:
    filename = 'pop_'+'traj'+str(l1)+'_init_st'+str(init_st)+'_total_st'+str(n_st)
if plt_err:
    filename = filename + '_ERRbar'
#--------------------------- SAVE EPS --------------------------------#
#plt.savefig(filename+'.eps', bbox_inches='tight', format='eps', dpi=600)
#--------------------------- SAVE PNG --------------------------------#
plt.savefig(filename+'.png', bbox_inches='tight',dpi=600)
