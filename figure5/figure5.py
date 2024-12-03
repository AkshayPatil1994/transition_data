from scipy.optimize import curve_fit
import numpy as np
import cblind as cb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

def compute_integral_length_scale(acf, lags):
    # Tritton Physical Fluid Dynamics second edition threshold
    # Page 307
    threshold = 1 / np.e
    test = np.where(acf <= threshold)[0]
    if(len(test) == 0):
        idx = -2
    else:
        idx = test[0]
    # Integrate the ACF up to this lag
    integral_length_scale = np.trapz(acf[:idx+1], lags[:idx+1])
    
    return integral_length_scale
#
# USER INPUT PARAMETERS
#
save_Fig = False
show_Fig = True
filename='figure5.png'
cmap = cm._colormaps['Spectral_r']
ydown, yup = 1e-10,1e-1
cdown, cup = 0.0,0.005
# Averaging and location prompt
Niters = 168
Niters_re500 = 239
zloc = 100
zloc_re500 = 190
# Flow and Fluid
H = 1.0; kvisc = 1.0e-6
Lx = 4*np.pi*H; Nx_re350 = 1048; Nz_re350 = 128
Nx_re500 = 1500; Nz_re500 = 256
utau_re350 = 3.5e-4; utau_re500 = 5e-4
#
# Load all the data
#
data = np.loadtxt('data/grid_re350.out')
z_re350 = data[1:-1,1]*350.0
data = np.loadtxt('data/grid_re500.out')
z_re500 = data[1:-1,1]*500.0
x_re350 = np.arange(0,Lx,step=Lx/Nx_re350); xcorr_re350 = x_re350[:int(Nx_re350/2)]
x_re500 = np.arange(0,Lx,step=Lx/Nx_re500); xcorr_re500 = x_re500[:int(Nx_re500/2)]
# Print zlocations
print("Re 350: ",z_re350[zloc])
print("Re 500: ",z_re500[zloc_re500])
#
# Log
# Re 350
Ruu_cansic_re350 = np.load('data/Re350/cansic/Ruu_x.npy'); Ruu_cansic_re350[:,:,:] = Ruu_cansic_re350[:,:,:]/Ruu_cansic_re350[0,:,:]
Rvv_cansic_re350 = np.load('data/Re350/cansic/Rvv_x.npy'); Rvv_cansic_re350[:,:,:] = Rvv_cansic_re350[:,:,:]/Rvv_cansic_re350[0,:,:]
Rww_cansic_re350 = np.load('data/Re350/cansic/Rww_x.npy'); Rww_cansic_re350[:,:,:] = Rww_cansic_re350[:,:,:]/Rww_cansic_re350[0,:,:]
# Re 500
# Log
Ruu_cansic_re500 = np.load('data/Re500/cansic/Ruu_x.npy'); Ruu_cansic_re500[:,:,:] = Ruu_cansic_re500[:,:,:]/Ruu_cansic_re500[0,:,:]
Rvv_cansic_re500 = np.load('data/Re500/cansic/Rvv_x.npy'); Rvv_cansic_re500[:,:,:] = Rvv_cansic_re500[:,:,:]/Rvv_cansic_re500[0,:,:]
Rww_cansic_re500 = np.load('data/Re500/cansic/Rww_x.npy'); Rww_cansic_re500[:,:,:] = Rww_cansic_re500[:,:,:]/Rww_cansic_re500[0,:,:]
#
# Linear
# Re 350
Ruu_linic_re350 = np.load('data/Re350/linic/Ruu_x.npy'); Ruu_linic_re350[:,:,:] = Ruu_linic_re350[:,:,:]/Ruu_linic_re350[0,:,:]
Rvv_linic_re350 = np.load('data/Re350/linic/Rvv_x.npy'); Rvv_linic_re350[:,:,:] = Rvv_linic_re350[:,:,:]/Ruu_linic_re350[0,:,:]
Rww_linic_re350 = np.load('data/Re350/linic/Rww_x.npy'); Rww_linic_re350[:,:,:] = Rww_linic_re350[:,:,:]/Ruu_linic_re350[0,:,:]
# Re 500
Ruu_linic_re500 = np.load('data/Re500/linic/Ruu_x.npy'); Ruu_linic_re500[:,:,:] = Ruu_linic_re500[:,:,:]/Ruu_linic_re500[0,:,:]
Rvv_linic_re500 = np.load('data/Re500/linic/Rvv_x.npy'); Rvv_linic_re500[:,:,:] = Rvv_linic_re500[:,:,:]/Ruu_linic_re500[0,:,:]
Rww_linic_re500 = np.load('data/Re500/linic/Rww_x.npy'); Rww_linic_re500[:,:,:] = Rww_linic_re500[:,:,:]/Ruu_linic_re500[0,:,:]
#
# Synthetic
# Re350
Ruu_synthetic_re350 = np.load('data/Re350/synthetic/Ruu_x.npy'); Ruu_synthetic_re350[:,:,:] = Ruu_synthetic_re350[:,:,:]/Ruu_synthetic_re350[0,:,:]
Rvv_synthetic_re350 = np.load('data/Re350/synthetic/Rvv_x.npy'); Rvv_synthetic_re350[:,:,:] = Rvv_synthetic_re350[:,:,:]/Rvv_synthetic_re350[0,:,:]
Rww_synthetic_re350 = np.load('data/Re350/synthetic/Rww_x.npy'); Rww_synthetic_re350[:,:,:] = Rww_synthetic_re350[:,:,:]/Rww_synthetic_re350[0,:,:]
# Re500
Ruu_synthetic_re500 = np.load('data/Re500/synthetic/Ruu_x.npy'); Ruu_synthetic_re500[:,:,:] = Ruu_synthetic_re500[:,:,:]/Ruu_synthetic_re500[0,:,:]
Rvv_synthetic_re500 = np.load('data/Re500/synthetic/Rvv_x.npy'); Rvv_synthetic_re500[:,:,:] = Rvv_synthetic_re500[:,:,:]/Rvv_synthetic_re500[0,:,:]
Rww_synthetic_re500 = np.load('data/Re500/synthetic/Rww_x.npy'); Rww_synthetic_re500[:,:,:] = Rww_synthetic_re500[:,:,:]/Rww_synthetic_re500[0,:,:]
#
# Compute integral length scale
#
# Re 350
Luu_cansic_Re350 = np.zeros((Nz_re350,Niters));Lvv_cansic_Re350 = np.zeros((Nz_re350,Niters));Lww_cansic_Re350 = np.zeros((Nz_re350,Niters))
Luu_linic_Re350 = np.zeros((Nz_re350,Niters));Lvv_linic_Re350 = np.zeros((Nz_re350,Niters));Lww_linic_Re350 = np.zeros((Nz_re350,Niters))
Luu_synthetic_Re350 = np.zeros((Nz_re350,Niters));Lvv_synthetic_Re350 = np.zeros((Nz_re350,Niters));Lww_synthetic_Re350 = np.zeros((Nz_re350,Niters))
# Re 500
Luu_cansic_Re500 = np.zeros((Nz_re500,Niters_re500));Lvv_cansic_Re500 = np.zeros((Nz_re500,Niters_re500));Lww_cansic_Re500 = np.zeros((Nz_re500,Niters_re500))
Luu_linic_Re500 = np.zeros((Nz_re500,Niters_re500));Lvv_linic_Re500 = np.zeros((Nz_re500,Niters_re500));Lww_linic_Re500 = np.zeros((Nz_re500,Niters_re500))
Luu_synthetic_Re500 = np.zeros((Nz_re500,Niters_re500));Lvv_synthetic_Re500 = np.zeros((Nz_re500,Niters_re500));Lww_synthetic_Re500 = np.zeros((Nz_re500,Niters_re500))
#
time_re350 = np.linspace(0,10,np.shape(Luu_synthetic_Re350)[1])
time_re500 = np.linspace(0,10,np.shape(Luu_linic_Re500)[1])
# Compute the integral length scale in x 
# Re 350
for kk in range(0,Nz_re350):
    for iter in range(0,Niters):
        # Cansic
        Luu_cansic_Re350[kk,iter] = compute_integral_length_scale(Ruu_cansic_re350[:,kk,iter],xcorr_re350)
        Lvv_cansic_Re350[kk,iter] = compute_integral_length_scale(Rvv_cansic_re350[:,kk,iter],xcorr_re350)
        Lww_cansic_Re350[kk,iter] = compute_integral_length_scale(Rww_cansic_re350[:,kk,iter],xcorr_re350)
        # Linear
        Luu_linic_Re350[kk,iter] = compute_integral_length_scale(Ruu_linic_re350[:,kk,iter],xcorr_re350)
        Lvv_linic_Re350[kk,iter] = compute_integral_length_scale(Rvv_linic_re350[:,kk,iter],xcorr_re350)
        Lww_linic_Re350[kk,iter] = compute_integral_length_scale(Rww_linic_re350[:,kk,iter],xcorr_re350)
        # Synthetic
        Luu_synthetic_Re350[kk,iter] = compute_integral_length_scale(Ruu_synthetic_re350[:,kk,iter],xcorr_re350)
        Lvv_synthetic_Re350[kk,iter] = compute_integral_length_scale(Rvv_synthetic_re350[:,kk,iter],xcorr_re350)
        Lww_synthetic_Re350[kk,iter] = compute_integral_length_scale(Rww_synthetic_re350[:,kk,iter],xcorr_re350)
# Re 500
for kk in range(0,Nz_re500):
    for iter in range(0,Niters_re500):
        # Cansic
        Luu_cansic_Re500[kk,iter] = compute_integral_length_scale(Ruu_cansic_re500[:,kk,iter],xcorr_re500)
        Lvv_cansic_Re500[kk,iter] = compute_integral_length_scale(Rvv_cansic_re500[:,kk,iter],xcorr_re500)
        Lww_cansic_Re500[kk,iter] = compute_integral_length_scale(Rww_cansic_re500[:,kk,iter],xcorr_re500)
        # Linear
        Luu_linic_Re500[kk,iter] = compute_integral_length_scale(Ruu_linic_re500[:,kk,iter],xcorr_re500)
        Lvv_linic_Re500[kk,iter] = compute_integral_length_scale(Rvv_linic_re500[:,kk,iter],xcorr_re500)
        Lww_linic_Re500[kk,iter] = compute_integral_length_scale(Rww_linic_re500[:,kk,iter],xcorr_re500)
        # Synthetic
        Luu_synthetic_Re500[kk,iter] = compute_integral_length_scale(Ruu_synthetic_re500[:,kk,iter],xcorr_re500)
        Lvv_synthetic_Re500[kk,iter] = compute_integral_length_scale(Rvv_synthetic_re500[:,kk,iter],xcorr_re500)
        Lww_synthetic_Re500[kk,iter] = compute_integral_length_scale(Rww_synthetic_re500[:,kk,iter],xcorr_re500)
#
# PLOTTING
#
color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=1.5, fontsize=20, markersize=8, labelsize=10, texuse=True, tickSize = 15)
plt.figure(1, figsize=(17, 6))
ax = plt.subplot(2,3,1)
ax.plot(time_re350,Luu_synthetic_Re350[zloc,:]*350,'1',markersize=6, color=color[4])
ax.plot(time_re350,Luu_linic_Re350[zloc,:]*350,'^', markersize=3, color=color[5])
ax.plot(time_re350,Luu_cansic_Re350[zloc,:]*350,'-', color=color[6])
ax.set_ylim([0,3000])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{11}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re350[80:], Luu_synthetic_Re350[zloc,80:]*350, '1', markersize=6, color=color[4])
ax_inset.plot(time_re350[80:], Luu_linic_Re350[zloc,80:]*350, '^', markersize=3, color=color[5])
ax_inset.plot(time_re350[80:], Luu_cansic_Re350[zloc,80:]*350, '-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
# vv
ax = plt.subplot(2,3,2)
ax.plot(time_re350,Lvv_synthetic_Re350[zloc,:]*350,'1',markersize=6, color=color[4])
ax.plot(time_re350,Lvv_linic_Re350[zloc,:]*350,'^', markersize=3, color=color[5])
ax.plot(time_re350,Lvv_cansic_Re350[zloc,:]*350,'-', color=color[6])
ax.set_ylim([0,500])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{22}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re350[80:], Lvv_synthetic_Re350[zloc,80:]*350, '1', markersize=6, color=color[4])
ax_inset.plot(time_re350[80:], Lvv_linic_Re350[zloc,80:]*350, '^', markersize=3, color=color[5])
ax_inset.plot(time_re350[80:], Lvv_cansic_Re350[zloc,80:]*350, '-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
# ww
ax = plt.subplot(2,3,3)
ax.plot(time_re350,Lww_synthetic_Re350[zloc,:]*350,'1',markersize=6, color=color[4])
ax.plot(time_re350,Lww_linic_Re350[zloc,:]*350,'^', markersize=3, color=color[5])
ax.plot(time_re350,Lww_cansic_Re350[zloc,:]*350,'-', color=color[6])
ax.set_ylim([0,300])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{33}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re350[80:], Lww_synthetic_Re350[zloc,80:]*350, '1', markersize=6, color=color[4])
ax_inset.plot(time_re350[80:], Lww_linic_Re350[zloc,80:]*350, '^', markersize=3, color=color[5])
ax_inset.plot(time_re350[80:], Lww_cansic_Re350[zloc,80:]*350, '-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
# Re500 uu
ax = plt.subplot(2,3,4)
ax.plot(time_re500,Luu_synthetic_Re500[zloc_re500,:]*500,'1',markersize=6, color=color[4])
ax.plot(time_re500,Luu_linic_Re500[zloc_re500,:]*500,'^', markersize=3, color=color[5])
ax.plot(time_re500,Luu_cansic_Re500[zloc_re500,:]*500,'-', color=color[6])
# ax.set_ylim([0,3000])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{11}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re500[125:],Luu_synthetic_Re500[zloc_re500,125:]*500,'1',markersize=6, color=color[4])
ax_inset.plot(time_re500[125:],Luu_linic_Re500[zloc_re500,125:]*500,'^', markersize=3, color=color[5])
ax_inset.plot(time_re500[125:],Luu_cansic_Re500[zloc_re500,125:]*500,'-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
ax = plt.subplot(2,3,5)
ax.plot(time_re500,Lvv_synthetic_Re500[zloc_re500,:]*500,'1',markersize=6, color=color[4])
ax.plot(time_re500, Lvv_linic_Re500[zloc_re500,:]*500, '^', markersize=3, color=color[5])
ax.plot(time_re500,Lvv_cansic_Re500[zloc_re500,:]*500,'-', color=color[6])
ax.set_ylim([0,400])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{22}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re500[125:],Lvv_synthetic_Re500[zloc_re500,125:]*500,'1',markersize=6, color=color[4])
ax_inset.plot(time_re500[125:], Lvv_linic_Re500[zloc_re500,125:]*500, '^', markersize=3, color=color[5])
ax_inset.plot(time_re500[125:],Lvv_cansic_Re500[zloc_re500,125:]*500,'-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
ax = plt.subplot(2,3,6)
ax.plot(time_re500,Lww_synthetic_Re500[zloc_re500,:]*500,'1',markersize=6, color=color[4])
ax.plot(time_re500, Lww_linic_Re500[zloc_re500,:]*500, '^', markersize=3, color=color[5])
ax.plot(time_re500,Lww_cansic_Re500[zloc_re500,:]*500,'-', color=color[6])
ax.set_ylim([0,300])
ax.set_xlabel(r'$T_{\epsilon}$',fontsize=20)
ax.set_ylabel(r'$\mathcal{L}_{33}$',fontsize=20)
# Create an inset with a zoomed-in region
ax_inset = inset_axes(ax, width="70%", height="40%", loc='upper right', borderpad=1.5)
# Zoomed data
ax_inset.plot(time_re500[125:],Lww_synthetic_Re500[zloc_re500,125:]*500,'1',markersize=6, color=color[4])
ax_inset.plot(time_re500[125:], Lww_linic_Re500[zloc_re500,125:]*500, '^', markersize=3, color=color[5])
ax_inset.plot(time_re500[125:], Lww_cansic_Re500[zloc_re500,125:]*500,'-', color=color[6])
# Customize inset ticks and labels
ax_inset.tick_params(axis='both', which='both',  
                     bottom=True, top=False, left=True, right=False,  
                     labelbottom=True, labelleft=True,  
                     labelsize=10,
                     direction='in',
                     length=6, width=1.5)
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
# Show and tight figure
plt.tight_layout()
if(save_Fig):
    plt.savefig(filename)
if(show_Fig):
    plt.show()