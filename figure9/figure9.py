import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cblind as cb
import matplotlib.pyplot as plt
#
# FUNCTION DEFINITIONS
#
def fixPlot(thickness=1.5, fontsize=20, markersize=8, labelsize=15, texuse=False, tickSize = 15):
    '''
        This plot sets the default plot parameters
    INPUT
        thickness:      [float] Default thickness of the axes lines
        fontsize:       [integer] Default fontsize of the axes labels
        markersize:     [integer] Default markersize
        labelsize:      [integer] Default label size
    OUTPUT
        None
    '''
    # Set the thickness of plot axes
    plt.rcParams['axes.linewidth'] = thickness    
    # Set the default fontsize
    plt.rcParams['font.size'] = fontsize    
    # Set the default markersize
    plt.rcParams['lines.markersize'] = markersize    
    # Set the axes label size
    plt.rcParams['axes.labelsize'] = labelsize
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = texuse
    # Tick size
    plt.rcParams['xtick.major.size'] = tickSize
    plt.rcParams['ytick.major.size'] = tickSize
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
#
# USER INPUT DATA
#
save_Fig = False
show_Fig = True
filename = 'figure9.png'
# Flow parameters
kappa = 0.4                         # von Karman constant
kvisc = 1.0e-6                      # Kinematic viscosity
utau_re350 = 3.5e-4                 # Friction velocity for Re500
utau_re500 = 5.0e-4                 # Friction velocity for Re500
#
# LOAD ALL DATA
#
grid_re500 = np.loadtxt('../figure3_and_4/data/grid_re500.out')
U_cansic_re500 = np.load('../figure3_and_4/data/cansic/Re500/Uplan.npy')
U_linic_re500 = np.load('../figure3_and_4/data/linic/Re500/Uplan.npy')
U_synthetic_re500 = np.load('../figure3_and_4/data/syntheticic/Re500/Uplan.npy')
U_restart_re500 = np.load('data/restart_Re500.npy')
#
# CALCULATE 
#
z_re500 = grid_re500[1:-1,2]
gradu_cansic_re500 = kvisc*np.gradient(U_cansic_re500[:,:],z_re500,axis=0)
gradu_linic_re500 = kvisc*np.gradient(U_linic_re500[:,:],z_re500,axis=0)
gradu_synthetic_re500 = kvisc*np.gradient(U_synthetic_re500[:,:],z_re500,axis=0)
time_cansic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
time_linic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
time_synthetic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
# Restart method
gradu_restart_re500 = kvisc*np.gradient(U_restart_re500[:,:],z_re500,axis=0)
time_restart_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
#
# PLOTTING
#
color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=2.5, fontsize=25, markersize=8, labelsize=15, texuse=True, tickSize = 15)
plt.figure(10,figsize=(16,6))
plt.title(r'$Re_{\tau} = 500.0$',fontsize=20)
plt.plot(time_cansic_re500[:]*utau_re500,gradu_cansic_re500[0,:]/utau_re500**2,':',markersize=2,color=color[6],label='Log Profile')
plt.plot(time_linic_re500[:]*utau_re500,gradu_linic_re500[0,:]/utau_re500**2,'-.',markersize=3,color=color[5],markerfacecolor='None',label='Linear Profile')
plt.plot(time_synthetic_re500[:]*utau_re500,gradu_synthetic_re500[0,:]/utau_re500**2,'-',markersize=5,color=color[4],label='Synthetic IC')
# Restart method
plt.plot(time_restart_re500[::30]*utau_re500,gradu_restart_re500[0,::30]/utau_re500**2,'-o',markersize=5,color='k',label='Precursor method')
# FORMATTING
plt.axhline(1,linestyle='--')
plt.axvline(5.0,linestyle='--')
plt.text(2.2,0.25,'Transient phase',fontsize=20)
plt.text(5.1,0.26,'Averaging Begins',fontsize=20)
# Add arrow
x_start = 0; x_end = 5
y_mid = 0.2 
plt.annotate(
    '', xy=(x_start, y_mid), xytext=(x_end, y_mid),
    ha='center', va='bottom', fontsize=12,
    arrowprops=dict(arrowstyle='<->', color='black',linewidth=1.5)
)
# Add avg. arrow
x_start = 5.0; x_end = 6.0
y_mid = 0.2 
plt.annotate(
    '', xy=(x_start, y_mid), xytext=(x_end, y_mid),
    ha='center', va='bottom', fontsize=12,
    arrowprops=dict(arrowstyle='<-[', color='black',linewidth=1.5)
)
#Shade ±5% around the horizontal line at y = 1
y_value = 1
tolerance = 0.05  # 5%
plt.fill_between(
    [0, 10.0],  # X range
    y_value * (1 - tolerance),  # Lower bound (y = 0.95)
    y_value * (1 + tolerance),  # Upper bound (y = 1.05)
    color='gray',               # Shade color
    alpha=1.0,                  # Transparency of the shading
    edgecolor='None'           # Border color
)
plt.ylabel(r'$u_{\tau}^{2+} \equiv \tau_{x_3=0}/ \Pi_c H$',fontsize=25)
plt.xlabel(r'$T_{\epsilon}$',fontsize=25)
plt.xlim([0,10]); plt.ylim([0,1.5])
plt.xticks(np.arange(0,11,1))
plt.legend(frameon=False,markerscale=2)
plt.tight_layout()
plt.grid()
if(save_Fig):
    plt.savefig(filename,format='eps')
if(show_Fig):
    plt.show()