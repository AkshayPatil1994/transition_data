import numpy as np
import matplotlib.pyplot as plt
import cblind as cb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
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

def load_budget(casepath):
    '''
        This function loads the budget terms into memory
    '''
    # 
    if os.path.isdir(casepath):
        # Load the data structures
        epsilon = np.load(casepath+'/epsilon.npy')
        pdiff = np.load(casepath+'/pdiff.npy')
        tke = np.load(casepath+'/tke.npy')
        transport = np.load(casepath+'/transport.npy')
        viscous = np.load(casepath+'/viscous.npy')
    else:
        exit("Directory does not exist.")

    return epsilon, pdiff, tke, transport, viscous

#
# USER INPUT DATA
#
save_Fig = False
show_Fig = True
filename = 'figure7.png'
cmap = cm._colormaps['Spectral_r']
kvisc = 1e-6; kappa = 0.4
astart = 84; astart_re500 = 125
H = 1.0; utau_re350 = 3.5e-4; utau_re500 = 5.0e-4
#
# LOAD ALL THE DATA
#
norm_fac_re350 = (utau_re350**4)/(kappa*kvisc)
norm_fac_re500 = (utau_re500**4)/(kappa*kvisc)
# Grid
data1 = np.loadtxt('data/grid_re350.out')
z_re350 = data1[1:-1,1]
data2 = np.loadtxt('data/grid_re500.out')
z_re500 = data2[1:-1,1]
# MKM 395 reference data
MKM = np.loadtxt('data/MKM_balance.dat',skiprows=1)
# PBO 550 reference data
PBO = np.loadtxt('data/PBO_balance.dat',skiprows=1)
#
# Data - Re350
#
uw_cansic_re350 = np.load('../figure3/data/cansic/Re350/uw.npy')
Umean_cansic_Re350 = np.load('../figure3/data/cansic/Re350/Uplan.npy')
P_cansic_re350 = np.nanmean(uw_cansic_re350[astart:,:],axis=0)*np.gradient(np.nanmean(Umean_cansic_Re350[:,1960:],axis=1),z_re350)
[epsilon_cansic_re350, pdiff_cansic_re350, tke_cansic_re350, transport_cansic_re350, viscous_cansic_re350] = load_budget('data/cansic/Re350')
uw_linic_re350 = np.load('../figure3/data/linic/Re350/uw.npy')
Umean_linic_Re350 = np.load('../figure3/data/linic/Re350/Uplan.npy')
P_linic_re350 = np.nanmean(uw_linic_re350[astart:,:],axis=0)*np.gradient(np.nanmean(Umean_linic_Re350[:,1960:],axis=1),z_re350)
[epsilon_linic_re350, pdiff_linic_re350, tke_linic_re350, transport_linic_re350, viscous_linic_re350] = load_budget('data/linic/Re350')
uw_synthetic_re350 = np.load('../figure3/data/syntheticic/Re350/uw.npy')
Umean_synthetic_Re350 = np.load('../figure3/data/syntheticic/Re350/Uplan.npy')
P_synthetic_re350 = np.nanmean(uw_synthetic_re350[astart:,:],axis=0)*np.gradient(np.nanmean(Umean_synthetic_Re350[:,1960:],axis=1),z_re350)
[epsilon_synthetic_re350, pdiff_synthetic_re350, tke_synthetic_re350, transport_synthetic_re350, viscous_synthetic_re350] = load_budget('data/synthetic/Re350')
viscous_synthetic_re350[1:2,:] = 0.5*epsilon_synthetic_re350[1:2,:]
#
# Data - Re500
#
uw_cansic_re500 = np.load('../figure3/data/cansic/Re500/uw.npy')
Umean_cansic_Re500 = np.load('../figure3/data/cansic/Re500/Uplan.npy')
print(np.shape(Umean_cansic_Re500),np.shape(z_re500))
P_cansic_re500 = np.nanmean(uw_cansic_re500[astart_re500:,:],axis=0)*np.gradient(np.squeeze(np.nanmean(Umean_cansic_Re500[:,2400:],axis=1)),z_re500)
[epsilon_cansic_re500, pdiff_cansic_re500, tke_cansic_re500, transport_cansic_re500, viscous_cansic_re500] = load_budget('data/cansic/Re500')
uw_linic_re500 = np.load('../figure3/data/linic/Re500/uw.npy')
Umean_linic_Re500 = np.load('../figure3/data/linic/Re500/Uplan.npy')
P_linic_re500 = np.nanmean(uw_linic_re500[astart_re500:,:],axis=0)*np.gradient(np.nanmean(Umean_linic_Re500[:,2400:],axis=1),z_re500)
[epsilon_linic_re500, pdiff_linic_re500, tke_linic_re500, transport_linic_re500, viscous_linic_re500] = load_budget('data/linic/Re500')
uw_synthetic_re500 = np.load('../figure3/data/syntheticic/Re500/uw.npy')
Umean_synthetic_Re500 = np.load('../figure3/data/syntheticic/Re500/Uplan.npy')
P_synthetic_re500 = np.nanmean(uw_synthetic_re500[astart_re500:,:],axis=0)*np.gradient(np.nanmean(Umean_synthetic_Re500[:,2400:],axis=1),z_re500)
[epsilon_synthetic_re500, pdiff_synthetic_re500, tke_synthetic_re500, transport_synthetic_re500, viscous_synthetic_re500] = load_budget('data/synthetic/Re500')
viscous_synthetic_re500[1:2,:] = 0.5*epsilon_synthetic_re500[1:2,:]
#
# Query the dataset size (assumes all arrays for each case are same sized)
# Re 350
ds_cansic_re350 = np.shape(epsilon_cansic_re350); print("Shape - Re350 - CansIC: ",ds_cansic_re350)
ds_linic_re350 = np.shape(epsilon_linic_re350); print("Shape - Re350 - LinIC: ",ds_linic_re350)
ds_synthetic_re350 = np.shape(epsilon_synthetic_re350); print("Shape - Re350 - Synthetic: ",ds_synthetic_re350)
# Re 500
ds_cansic_re500 = np.shape(epsilon_cansic_re500); print("Shape - Re500 - CansIC: ",ds_cansic_re500)
ds_linic_re500 = np.shape(epsilon_linic_re500); print("Shape - Re500 - LinIC: ",ds_linic_re500)
ds_synthetic_re500 = np.shape(epsilon_synthetic_re500); print("Shape - Re500 - Synthetic: ",ds_synthetic_re500)
#
# PLOTTING
#
color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=2.0, fontsize=15, markersize=8, labelsize=15, texuse=True, tickSize = 10)
plt.figure(1,figsize=(16,10))
plt.subplot(5,2,1)
plt.plot(z_re350*350,-P_synthetic_re350/norm_fac_re350,'1',markerfacecolor='None',markersize=5,color=color[6],label='Synthetic')
plt.plot(z_re350*350,-P_linic_re350/norm_fac_re350,'^',markersize=5,color=color[5],label='Linear Profile')
plt.plot(z_re350*350,-P_cansic_re350/norm_fac_re350,'-',markerfacecolor='None',markersize=5,color=color[4],label='Log Profile')
plt.plot(MKM[:,1],MKM[:,3]*kappa,'-',color=color[0],label='Moser, Kim, Mansour (1999)')
# FORMATTING
plt.xlim([1,120]); plt.ylim([0,0.13])
plt.xticks([1,20,40,60,80,100,120],labels=[]); plt.yticks([0.0,0.05,0.1])
plt.grid()
plt.legend(frameon=False,fontsize=10)
plt.ylabel(r'$\mathcal{P}_k^+$',fontsize=18)
plt.title(r'$Re_{\tau} = 350.0$',fontsize=20)
plt.subplot(5,2,3)
plt.plot(z_re350*350,-np.nanmean(epsilon_synthetic_re350[:,astart:],axis=1)/norm_fac_re350,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re350*350,-np.nanmean(epsilon_linic_re350[:,astart:],axis=1)/norm_fac_re350,'^',markersize=5,color=color[5])
plt.plot(z_re350*350,-np.nanmean(epsilon_cansic_re350[:,astart:],axis=1)/norm_fac_re350,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(MKM[:,1],MKM[:,2]*kappa,'-',color=color[0])
# FORMATTING
plt.xlim([1,120]); plt.ylim([-0.1,0.0])
plt.yticks([-0.1,-0.05,0.0])
plt.xticks([1,20,40,60,80,100,120],labels=[])
plt.grid()
plt.ylabel(r'$\mathcal{\epsilon}_k^+$',fontsize=18)
plt.subplot(5,2,5)
plt.plot(z_re350*350,2*np.nanmean(viscous_synthetic_re350[:,astart:],axis=1)/norm_fac_re350,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re350*350,2*np.nanmean(viscous_linic_re350[:,astart:],axis=1)/norm_fac_re350,'^',markersize=5,color=color[5])
plt.plot(z_re350*350,2*np.nanmean(viscous_cansic_re350[:,astart:],axis=1)/norm_fac_re350,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(MKM[:,1],MKM[:,7]*kappa,'-',color=color[0])
# FORMATTING
plt.xlim([1,120]); plt.ylim([-0.05,0.1])
plt.xticks([1,20,40,60,80,100,120],labels=[]); plt.yticks([-0.05,0.0,0.05,0.1])
plt.grid()
plt.ylabel(r'$\mathcal{V}_k^+$',fontsize=18)
plt.subplot(5,2,7)
plt.plot(z_re350*350,-np.nanmean(transport_synthetic_re350[:,astart:],axis=1)/norm_fac_re350,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re350*350,-np.nanmean(transport_linic_re350[:,astart:],axis=1)/norm_fac_re350,'^',markersize=5,color=color[5])
plt.plot(z_re350*350,-np.nanmean(transport_cansic_re350[:,astart:],axis=1)/norm_fac_re350,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(MKM[:,1],MKM[:,6]*kappa,'-',color=color[0])
plt.xlim([1,120])#; plt.ylim([])
plt.xticks([1,20,40,60,80,100,120],labels=[])
plt.grid()
plt.ylabel(r'$\mathcal{T}_k^+$',fontsize=18)
plt.subplot(5,2,9)
plt.plot(z_re350*350,-2*np.nanmean(pdiff_synthetic_re350[:,astart:],axis=1)/norm_fac_re350,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re350*350,-2*np.nanmean(pdiff_linic_re350[:,astart:],axis=1)/norm_fac_re350,'^',markersize=5,color=color[5])
plt.plot(z_re350*350,-2*np.nanmean(pdiff_cansic_re350[:,astart:],axis=1)/norm_fac_re350,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(MKM[:,1],MKM[:,5]*kappa,'-',color=color[0])
plt.xlim([1,120]); plt.xticks([1,20,40,60,80,100,120])
plt.grid()
plt.ylabel(r'$\Pi_k^+$',fontsize=18)
plt.xlabel(r'$x_3^+$',fontsize=20)
#
# Retau 500
#
plt.subplot(5,2,2)
plt.plot(z_re500*500,-P_synthetic_re500/norm_fac_re500,'1',markerfacecolor='None',markersize=5,color=color[6],label='Synthetic')
plt.plot(z_re500*500,-P_linic_re500/norm_fac_re500,'^',markersize=5,color=color[5],label='Linear Profile')
plt.plot(z_re500*500,-P_cansic_re500/norm_fac_re500,'-',markerfacecolor='None',markersize=5,color=color[4],label='Log Profile')
plt.plot(PBO[:,1],PBO[:,2]*kappa,'-',color=color[0],label='Bernardini, Pirozzoli, Orlandi (2014)')
# FORMATTING
plt.xlim([1,120]); plt.ylim([0,0.13])
plt.xticks([1,20,40,60,80,100,120],labels=[]); plt.yticks([0.0,0.05,0.1])
plt.grid()
plt.legend(frameon=False,fontsize=10)
plt.title(r'$Re_{\tau} = 500.0$',fontsize=20)
plt.subplot(5,2,4)
plt.plot(z_re500*500,-np.nanmean(epsilon_synthetic_re500[:,astart:],axis=1)/norm_fac_re500,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re500*500,-np.nanmean(epsilon_linic_re500[:,astart:],axis=1)/norm_fac_re500,'^',markersize=5,color=color[5])
plt.plot(z_re500*500,-np.nanmean(epsilon_cansic_re500[:,astart:],axis=1)/norm_fac_re500,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(PBO[:,1],PBO[:,6]*kappa,'-',color=color[0],label='Bernardini, Pirozzoli, Orlandi (2014)')
# FORMATTING
plt.xlim([1,120]); plt.ylim([-0.1,0.0])
plt.yticks([-0.1,-0.05,0.0])
plt.xticks([1,20,40,60,80,100,120],labels=[])
plt.grid()
plt.subplot(5,2,6)
plt.plot(z_re500*500,2*np.nanmean(viscous_synthetic_re500[:,astart:],axis=1)/norm_fac_re500,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re500*500,2*np.nanmean(viscous_linic_re500[:,astart:],axis=1)/norm_fac_re500,'^',markersize=5,color=color[5])
plt.plot(z_re500*500,2*np.nanmean(viscous_cansic_re500[:,astart:],axis=1)/norm_fac_re500,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(PBO[:,1],PBO[:,5]*kappa,'-',color=color[0],label='Bernardini, Pirozzoli, Orlandi (2014)')
# FORMATTING
plt.xlim([1,120]); plt.ylim([-0.05,0.1])
plt.xticks([1,20,40,60,80,100,120],labels=[]); plt.yticks([-0.05,0.0,0.05,0.1])
plt.grid()

plt.subplot(5,2,8)
plt.plot(z_re500*500,-np.nanmean(transport_synthetic_re500[:,astart:],axis=1)/norm_fac_re500,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re500*500,-np.nanmean(transport_linic_re500[:,astart:],axis=1)/norm_fac_re500,'^',markersize=5,color=color[5])
plt.plot(z_re500*500,-np.nanmean(transport_cansic_re500[:,astart:],axis=1)/norm_fac_re500,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(PBO[:,1],PBO[:,3]*kappa,'-',color=color[0],label='Bernardini, Pirozzoli, Orlandi (2014)')
plt.xlim([1,120])#; plt.ylim([])
plt.xticks([1,20,40,60,80,100,120],labels=[])
plt.grid()
plt.subplot(5,2,10)
plt.plot(z_re500*500,-2*np.nanmean(pdiff_synthetic_re500[:,astart:],axis=1)/norm_fac_re500,'1',markerfacecolor='None',markersize=5,color=color[6])
plt.plot(z_re500*500,-2*np.nanmean(pdiff_linic_re500[:,astart:],axis=1)/norm_fac_re500,'^',markersize=5,color=color[5])
plt.plot(z_re500*500,-2*np.nanmean(pdiff_cansic_re500[:,astart:],axis=1)/norm_fac_re500,'-',markerfacecolor='None',markersize=5,color=color[4])
plt.plot(PBO[:,1],PBO[:,4]*kappa,'-',color=color[0],label='Bernardini, Pirozzoli, Orlandi (2014)')
plt.xlim([1,120]); plt.xticks([1,20,40,60,80,100,120])
plt.xlabel(r'$x_3^+$',fontsize=20)
plt.grid()
if(save_Fig):
    plt.savefig(filename)
if(show_Fig):
    plt.show()