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
# Name of the files to save figures
filenameu_re350 = 'figure4u_re350.png' 
filenameu_re500 = 'figure4u_re500.png'
filenamerms_re350 = 'figure4rms_re350.png'
filenamerms_re500 = 'figure4rms_re500.png'
filenameconvergence = 'figure3.png'
#
# Fluid Flow parameters
#
kappa = 0.4                         # von Karman constant
kvisc = 1.0e-6                      # Kinematic viscosity
utau_re350 = 3.5e-4                 # Friction velocity for Re500
utau_re500 = 5.0e-4                 # Friction velocity for Re500
plotstartindex = 3000               # This is the start of the velocity index (total size == num. planform profiles)
plotstartindex_re500 = 2400         # This is the start of the velocity index (total size == num. planform profiles)
rms_startindex_re350 = 80           # This is the start of rms plotting index (total size == num. 3D snapshots - Re350)
rms_startindex_re500 = 128          # This is the start of rms plotting index (total size == num. 3D snapshots - Re500)
#
# Load all the data
#
grid_re350 = np.loadtxt('data/grid.out')
grid_re500 = np.loadtxt('data/grid_re500.out')
z_re350 = grid_re350[1:-1,2]
z_re500 = grid_re500[1:-1,2]
# Mean velocity data - Re350
U_cansic_re350 = np.load('data/cansic/Re350/Uplan.npy')
U_linic_re350 = np.load('data/linic/Re350/Uplan.npy')
U_synthetic_re350 = np.load('data/syntheticic/Re350/Uplan.npy')
# Mean velocity data - Re500
U_cansic_re500 = np.load('data/cansic/Re500/Uplan.npy')
U_linic_re500 = np.load('data/linic/Re500/Uplan.npy')
U_synthetic_re500 = np.load('data/syntheticic/Re500/Uplan.npy')
# Reference data
MKM = np.loadtxt('data/MKM.dat',skiprows=1)
MKM_RS = np.loadtxt('data/MKM_reystress.dat',skiprows=1)
PBO = np.loadtxt('data/PBO.dat',skiprows=1)
# ums 
urms_cansic_re350 = np.load('data/cansic/Re350/urms.npy')
urms_linic_re350 = np.load('data/linic/Re350/urms.npy')
urms_synthetic_re350 = np.load('data/syntheticic/Re350/urms.npy')
urms_cansic_re500 = np.load('data/cansic/Re500/urms.npy')
urms_linic_re500 = np.load('data/linic/Re500/urms.npy')
urms_synthetic_re500 = np.load('data/syntheticic/Re500/urms.npy')
# vrms
vrms_cansic_re350 = np.load('data/cansic/Re350/vrms.npy')
vrms_linic_re350 = np.load('data/linic/Re350/vrms.npy')
vrms_synthetic_re350 = np.load('data/syntheticic/Re350/vrms.npy')
vrms_cansic_re500 = np.load('data/cansic/Re500/vrms.npy')
vrms_linic_re500 = np.load('data/linic/Re500/vrms.npy')
vrms_synthetic_re500 = np.load('data/syntheticic/Re500/vrms.npy')
# wrms
wrms_cansic_re350 = np.load('data/cansic/Re350/wrms.npy')
wrms_linic_re350 = np.load('data/linic/Re350/wrms.npy')
wrms_synthetic_re350 = np.load('data/syntheticic/Re350/wrms.npy')
wrms_cansic_re500 = np.load('data/cansic/Re500/wrms.npy')
wrms_linic_re500 = np.load('data/linic/Re500/wrms.npy')
wrms_synthetic_re500 = np.load('data/syntheticic/Re500/wrms.npy')
# Stress
uw_cansic_re350 = np.load('data/cansic/Re350/uw.npy')
uw_linic_re350 = np.load('data/linic/Re350/uw.npy')
uw_synthetic_re350 = np.load('data/syntheticic/Re350/uw.npy')
uw_cansic_re500 = np.load('data/cansic/Re500/uw.npy')
uw_linic_re500 = np.load('data/linic/Re500/uw.npy')
uw_synthetic_re500 = np.load('data/syntheticic/Re500/uw.npy')
# Get size of all datasets
dscans_re350 = np.shape(U_cansic_re350); print("Size of CaNS - 350:",dscans_re350)
dslin_re350 = np.shape(U_linic_re350); print("Size of Linear - 350:",dslin_re350)
dssyn_re350 = np.shape(U_synthetic_re350); print("Size of Synthetic - 350:",dssyn_re350)
# Re500
dscans_re500 = np.shape(U_cansic_re500); print("Size of CaNS - 500:",dscans_re500)
dslin_re500 = np.shape(U_linic_re500); print("Size of Linear - 500:",dslin_re500)
dssyn_re500 = np.shape(U_synthetic_re500); print("Size of Synthetic - 500:",dssyn_re500)
#
# PLOTTING
#
color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=2.5, fontsize=25, markersize=8, labelsize=15, texuse=True, tickSize = 15)
plt.figure(1,figsize=(8,6))
plt.semilogx(z_re350[30:]*350,(1/kappa)*np.log(z_re350[30:]*350)+5.2,'k--')
plt.semilogx(z_re350[0:14]*350,z_re350[0:14]*350,'k--')
plt.semilogx(z_re350*350,np.mean(U_synthetic_re350[:,plotstartindex:-1],axis=1)/utau_re350,'1',markersize=6,color=color[4],label='Synthetic')
plt.semilogx(z_re350*350,np.mean(U_linic_re350[:,plotstartindex:-1],axis=1)/utau_re350,'^',markersize=3,color=color[5],label='Linear Profile')
plt.semilogx(z_re350*350,np.mean(U_cansic_re350[:,plotstartindex:-1],axis=1)/utau_re350,'-',color=color[6],label='Log Profile')
plt.semilogx(MKM[:,1],MKM[:,2],'-',label='Moser, Kim, Mansour (1999)')
plt.legend(frameon=False,markerscale=1.5,fontsize=18,loc='upper left')
# FORMATTING
plt.grid(); plt.xlim([0,350*1.5]); plt.ylim([0,25.0])
plt.gca().tick_params(axis='x', pad=5)
plt.xlabel(r'$x_3^+$',fontsize=30, labelpad=5)
plt.ylabel(r'$\langle \overline{U}_1 \rangle^+$',fontsize=30)
# Show plot
plt.tight_layout()
if(save_Fig):
    plt.savefig(filenameu_re350)
## Re500 plots
fixPlot(thickness=2.5, fontsize=25, markersize=8, labelsize=15, texuse=True, tickSize = 15)
plt.figure(4,figsize=(8,6))
plt.semilogx(z_re500[30:]*500,(1/kappa)*np.log(z_re500[30:]*500)+5.2,'k--')
plt.semilogx(z_re500[0:10]*500,z_re500[0:10]*500,'k--')
plt.semilogx(z_re500*500,np.mean(U_synthetic_re500[:,plotstartindex_re500:],axis=1)/utau_re500,'1',markersize=6,color=color[4],label='Synthetic')
plt.semilogx(z_re500*500,np.mean(U_linic_re500[:,plotstartindex_re500:],axis=1)/utau_re500,'^',markersize=3,color=color[5],label='Linear Profile')
plt.semilogx(z_re500*500,np.mean(U_cansic_re500[:,plotstartindex_re500:],axis=1)/utau_re500,'-',color=color[6],label='Log Profile')
plt.semilogx(PBO[:,1],PBO[:,2],'-',label='Bernardini, Pirozzoli, Orlandi (2014)')
plt.legend(frameon=False,markerscale=1.5,fontsize=18,loc='upper left')
# FORMATTING
plt.grid(); plt.xlim([0,500*1.5]); plt.ylim([0,25.0])
plt.gca().tick_params(axis='x', pad=5)
plt.xlabel(r'$x_3^+$',fontsize=30, labelpad=5)
plt.ylabel(r'$\langle \overline{U}_1 \rangle^+$',fontsize=30)
# Show plot
plt.tight_layout()
if(save_Fig):
    plt.savefig(filenameu_re500)
#-------------------------------------------------------------------------#
#                       Time History                                      #
#-------------------------------------------------------------------------#
# Re350
gradu_cansic_re350 = kvisc*np.gradient(U_cansic_re350[:,:],z_re350,axis=0)
gradu_linic_re350 = kvisc*np.gradient(U_linic_re350[:,:],z_re350,axis=0)
gradu_synthetic_re350 = kvisc*np.gradient(U_synthetic_re350[:,:],z_re350,axis=0)
time_cansic_re350 = np.arange(start=0.0,step=0.16976025*50,stop=169000*0.16976025)
time_linic_re350 = np.arange(start=0.0,step=0.16976025*50,stop=169000*0.16976025)
time_synthetic_re350 = np.arange(start=0.0,step=0.16976025*50,stop=169000*0.16976025)
# Re500
gradu_cansic_re500 = kvisc*np.gradient(U_cansic_re500[:,:],z_re500,axis=0)
gradu_linic_re500 = kvisc*np.gradient(U_linic_re500[:,:],z_re500,axis=0)
gradu_synthetic_re500 = kvisc*np.gradient(U_synthetic_re500[:,:],z_re500,axis=0)
time_cansic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
time_linic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
time_synthetic_re500 = np.arange(start=0.0,step=0.0836*50,stop=0.0836*240000)
## Fit a line to cansic and linic to estimate the convergence rate
linefit_cansic_re350 = np.polyfit(time_cansic_re350[plotstartindex:], gradu_cansic_re350[0,plotstartindex:], 1)
linefit_linic_re350 = np.polyfit(time_linic_re350[plotstartindex:], gradu_linic_re350[0,plotstartindex:], 1)
linefit_cansic_re500 = np.polyfit(time_cansic_re500[plotstartindex_re500:], gradu_cansic_re500[0,plotstartindex_re500:], 1)
linefit_linic_re500 = np.polyfit(time_linic_re500[plotstartindex_re500:], gradu_linic_re500[0,plotstartindex_re500:], 1)
print("** Estimated time for convergence **")
conval = 1.0
print("Re350 - Logarithmic Profile: %f"%(utau_re350*(conval*utau_re350**2-linefit_cansic_re350[1])/linefit_cansic_re350[0]))
print("Re350 - Linear Profile: %f"%(utau_re350*(conval*utau_re350**2-linefit_linic_re350[1])/linefit_linic_re350[0]))
print("----")
print("Re500 - Logarithmic Profile: %f"%(utau_re500*(conval*utau_re500**2-linefit_cansic_re500[1])/linefit_cansic_re500[0]))
print("Re500 - Linear Profile: %f"%(utau_re500*(conval*utau_re500**2-linefit_linic_re500[1])/linefit_linic_re500[0]))
# PLOTTING
plt.figure(3,figsize=(16,10))
plt.subplot(2,1,1)
plt.title(r'$Re_{\tau} = 350.0$',fontsize=20)
plt.plot(time_cansic_re350*utau_re350,gradu_cansic_re350[0,:]/utau_re350**2,':',markersize=2,color=color[6],label='Log Profile')
plt.plot(time_linic_re350*utau_re350,gradu_linic_re350[0,:]/utau_re350**2,'-.',markersize=3,color=color[5],markerfacecolor='None',label='Linear Profile')
plt.plot(time_synthetic_re350*utau_re350,gradu_synthetic_re350[0,:]/utau_re350**2,'-',markersize=5,color=color[4],label='Synthetic')
# FORMATTING
plt.axhline(1,linestyle='--')
plt.axvline(5.0,linestyle='--')
plt.text(2.2,0.25,'Transient phase',fontsize=20)
plt.text(6.1,0.175,'Averaging Begins',fontsize=20)
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
# Shade ±5% around the horizontal line at y = 1
y_value = 1
tolerance = 0.05                # 5% band
plt.fill_between(
    [0, 10.0],  # X range
    y_value * (1 - tolerance),  # Lower bound (y = 0.95)
    y_value * (1 + tolerance),  # Upper bound (y = 1.05)
    color='gray',               # Shade color
    alpha=0.3,                  # Transparency of the shading
    edgecolor='None'            # Border color
)
plt.ylabel(r'$u_{\tau}^{2+} \equiv \tau_{x_3=0}/ \Pi_c H$',fontsize=25)
plt.xlabel(r'$T_{\epsilon}$',fontsize=25)
plt.xlim([0,10]); plt.ylim([0,1.5])
plt.xticks(np.arange(0,11,1))
plt.legend(frameon=False,markerscale=2)
# plt.tight_layout()
plt.grid()
plt.subplot(2,1,2)
plt.title(r'$Re_{\tau} = 500.0$',fontsize=20)
plt.plot(time_cansic_re500[:]*utau_re500,gradu_cansic_re500[0,:]/utau_re500**2,':',markersize=2,color=color[6],label='Log Profile')
plt.plot(time_linic_re500[:]*utau_re500,gradu_linic_re500[0,:]/utau_re500**2,'-.',markersize=3,color=color[5],markerfacecolor='None',label='Linear Profile')
plt.plot(time_synthetic_re500[:]*utau_re500,gradu_synthetic_re500[0,:]/utau_re500**2,'-',markersize=5,color=color[4],label='Synthetic IC')
# FORMATTING
plt.axhline(1,linestyle='--')
plt.axvline(5.0,linestyle='--')
plt.text(2.2,0.25,'Transient phase',fontsize=20)
plt.text(6.1,0.175,'Averaging Begins',fontsize=20)
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
    alpha=0.3,                  # Transparency of the shading
    edgecolor='None'           # Border color
)
plt.ylabel(r'$u_{\tau}^{2+} \equiv \tau_{x_3=0}/ \Pi_c H$',fontsize=25)
plt.xlabel(r'$T_{\epsilon}$',fontsize=25)
plt.xlim([0,10]); plt.ylim([0,1.5])
plt.xticks(np.arange(0,11,1))
# plt.legend(frameon=False,markerscale=2)
plt.tight_layout()
plt.grid()
if(save_Fig):
    plt.savefig(filenameconvergence)
#-------------------------------------------------------------------------#
#                       RMS and STRESS PROFILES                           #
#-------------------------------------------------------------------------#
plt.figure(2,figsize=(10,8))
plt.subplot(2,2,1)
# Urms
plt.plot(np.mean(urms_synthetic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'1',markersize=6,color=color[4])
plt.plot(np.mean(urms_linic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'^',markersize=4,color=color[5])
plt.plot(np.mean(urms_cansic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'-',color=color[6])
plt.xlabel(r'$\langle \overline{u_1^{rms}} \rangle^+$',fontsize=20)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.xticks([0,1,2,3],fontsize=15)
plt.yticks([0,200,350],fontsize=15); plt.ylim([0,350]); plt.xlim([0.0,3.0]);plt.grid()
# Vrms
plt.subplot(2,2,2)
plt.plot(np.mean(vrms_synthetic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'1',markersize=4,color=color[4])
plt.plot(np.mean(vrms_linic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'^',markersize=4,color=color[5])
plt.plot(np.mean(vrms_cansic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'-',markersize=4,color=color[6])
plt.xlabel(r'$\langle \overline{u_2^{rms}} \rangle^+$',fontsize=20)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.xticks([0,0.5,1,1.5],fontsize=15)
plt.yticks([0,200,350],fontsize=15);  plt.ylim([0,350]); plt.xlim([0.0,1.5]);plt.grid()
# Wrms
plt.subplot(2,2,3)
plt.plot(np.mean(wrms_synthetic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'1',markersize=4,color=color[4])
plt.plot(np.mean(wrms_linic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'^',markersize=4,color=color[5])
plt.plot(np.mean(wrms_cansic_re350[rms_startindex_re350:,:],axis=0)/utau_re350,z_re350*350,'-',markersize=4,color=color[6])
plt.xlabel(r'$\langle \overline{u_3^{rms}} \rangle^+$',fontsize=20)
plt.xticks([0,0.5,1.0],fontsize=15)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.yticks([0,200,350],fontsize=15);  plt.ylim([0,350]); plt.xlim([0.0,1.2]); plt.grid()
# W'U'
plt.subplot(2,2,4)
plt.plot((np.mean(uw_synthetic_re350[rms_startindex_re350:,:],axis=0))/utau_re350**2,z_re350*350,'1',markersize=6,color=color[4])
plt.plot(np.mean(uw_linic_re350[rms_startindex_re350:,:],axis=0)/utau_re350**2,z_re350*350,'^',markersize=4,color=color[5])
plt.plot((np.mean(uw_cansic_re350[rms_startindex_re350:,:],axis=0))/utau_re350**2,z_re350*350,'-',color=color[6])

plt.xlabel(r'$\langle \overline{u_1^{\prime}u_3^{\prime}} \rangle^+$',fontsize=20)
plt.xticks([-1.0,-0.5,0.0],fontsize=15)
plt.yticks([0,200,350],fontsize=15)
plt.ylabel(r'$x_3^+$',fontsize=20)
## FORMATTING AND REFERENCE PLOTS
plt.plot(z_re350-1,z_re350*350,'k--')
plt.grid(); plt.ylim([0,350]); plt.xlim([-1.0,0.0])
# Show plot
plt.tight_layout()
if(save_Fig):
    plt.savefig(filenamerms_re350)
## RE500
plt.figure(5,figsize=(10,8))
plt.subplot(2,2,1)
# Urms
plt.plot(np.mean(urms_synthetic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'1',markersize=6,color=color[4])
plt.plot(np.mean(urms_linic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'^',markersize=4,color=color[5])
plt.plot(np.mean(urms_cansic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'-',color=color[6])
plt.xlabel(r'$\langle \overline{u_1^{rms}} \rangle^+$',fontsize=20)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.xticks([0,1,2,3],fontsize=15)
plt.yticks([0,250,500],fontsize=15); plt.ylim([0,500]); plt.xlim([0.0,3.0]);plt.grid()
# Vrms
plt.subplot(2,2,2)
plt.plot(np.mean(vrms_synthetic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'1',markersize=4,color=color[4])
plt.plot(np.mean(vrms_linic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'^',markersize=4,color=color[5])
plt.plot(np.mean(vrms_cansic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'-',markersize=4,color=color[6])
plt.xlabel(r'$\langle \overline{u_2^{rms}} \rangle^+$',fontsize=20)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.xticks([0,0.5,1,1.5],fontsize=15)
plt.yticks([0,250,500],fontsize=15);  plt.ylim([0,500]); plt.xlim([0.0,1.5]);plt.grid()
# Wrms
plt.subplot(2,2,3)
plt.plot(np.mean(wrms_synthetic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'1',markersize=4,color=color[4])
plt.plot(np.mean(wrms_linic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'^',markersize=4,color=color[5])
plt.plot(np.mean(wrms_cansic_re500[rms_startindex_re500:,:],axis=0)/utau_re500,z_re500*500,'-',markersize=4,color=color[6])
plt.xlabel(r'$\langle \overline{u_3^{rms}} \rangle^+$',fontsize=20)
plt.xticks([0,0.5,1.0],fontsize=15)
plt.ylabel(r'$x_3^+$',fontsize=20)
plt.yticks([0,250,500],fontsize=15);  plt.ylim([0,500]); plt.xlim([0.0,1.2]); plt.grid()
# W'U'
plt.subplot(2,2,4)
plt.plot((np.mean(uw_synthetic_re500[rms_startindex_re500:,:],axis=0))/utau_re500**2,z_re500*500,'1',markersize=6,color=color[4])
plt.plot(np.mean(uw_linic_re500[rms_startindex_re500:,:],axis=0)/utau_re500**2,z_re500*500,'^',markersize=4,color=color[5])
plt.plot((np.mean(uw_cansic_re500[rms_startindex_re500:,:],axis=0))/utau_re500**2,z_re500*500,'-',color=color[6])
plt.xlabel(r'$\langle \overline{u_1^{\prime}u_3^{\prime}} \rangle^+$',fontsize=20)
plt.xticks([-1.0,-0.5,0.0],fontsize=15)
plt.yticks([0,250,500],fontsize=15)
plt.ylabel(r'$x_3^+$',fontsize=20)
# FORMATTING AND REFERENCE PLOTS
plt.plot(z_re500-1,z_re500*500,'k--')
plt.grid(); plt.ylim([0,500]); plt.xlim([-1.0,0.0])
#
# Show/Save Plot
#
plt.tight_layout()
if(save_Fig):
    plt.savefig(filenamerms_re500)
if(show_Fig):
    plt.show()