import numpy as np
import matplotlib.pyplot as plt
import cblind as cb
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
#
# USER INPUT DATA
#
save_Fig = False
show_Fig = True
filename_panela = 'figure5a_re350.png'
filename_panelb = 'figure5b_re350.png'
cmap = cm._colormaps['Spectral_r']
zlocs = [20,40,60,80,100,120]
ydown, yup = 1e-10,1e-1
cdown, cup = 0.0,0.005
astart=80
endInd = 159
kvisc = 1e-6
zloc = 120
Retau = 350.0
H = 1.0
utau = 3.5e-4
# CANSIC
Euu_cansic = np.load('data/Re350/spectrum_cansic.npy')
freqx_cansic = np.load('data/Re350/freqx_cansic.npy')
# LINIC
Euu_linic = np.load('data/Re350/spectrum_linic.npy')
freqx_linic = np.load('data/Re350/freqx_linic.npy')
# SYNIC
Euu_synic = np.load('data/Re350/spectrum_synthetic.npy')
freqx_synic = np.load('data/Re350/freqx_synthetic.npy')
endInd = 159
data = np.loadtxt('data/Re350/grid.out')
z = data[1:-1,1]*350.0
zend = np.where(freqx_cansic<400)[0][-1]
print("Size of datasets")
print("CansIC:",np.shape(Euu_cansic))
print("SynthetisIC:",np.shape(Euu_synic))
#
# PLOT ALL FIGURES
#
color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=1.5, fontsize=20, markersize=8, labelsize=10, texuse=True, tickSize = 15)
plt.figure(1, figsize=(15, 8))
for iter in range(0, len(zlocs)):
    zloc = zlocs[iter]
    ax = plt.subplot(2, 3, iter + 1)
    plt.title(r"$x_3^+ =$ %d" % (round(z[zloc])))
    
    # Main plot
    ax.loglog(freqx_synic[:zend], np.mean(np.squeeze(Euu_synic[:zend, zloc, astart:]) / utau**2, axis=1), '1', markersize=6, color=color[4], label=r'Synthetic')
    ax.loglog(freqx_linic[:zend], np.mean(np.squeeze(Euu_linic[:zend, zloc, astart:]) / utau**2, axis=1), '^', markersize=3, color=color[5], label=r'Linear Profile')
    ax.loglog(freqx_cansic[:zend], np.mean(np.squeeze(Euu_cansic[:zend, zloc, astart:]) / utau**2, axis=1), '-', color=color[6], label=r'Log Profile')    
    # -5/3rd
    # ax.loglog(freqx_synic[4:60], 8e-2 * (freqx_synic[4:60])**(-5 / 3.0), 'r--')
    
    # X and Y axis formatting
    if iter > 2:
        plt.xlabel(r'$k$', fontsize=25)
    else:
        plt.xticks([1, 10, 100], labels=[])
    
    if iter == 0 or iter == 3:
        plt.ylabel(r'$ \langle \overline{E}(k) \rangle_*$', fontsize=25)
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    else:
        plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2], labels=[])
        
    plt.grid()
    if iter == 0:
        plt.legend(frameon=False, markerscale=5)
    
    plt.ylim([1e-12, 5e-2])
    plt.subplots_adjust(bottom=0.15)
    
    # Adding inset for the bottom row panels (iter > 2)
    if iter > 2:
        # Create an inset with a zoomed-in region
        ax_inset = inset_axes(ax, width="50%", height="40%", loc='lower left', borderpad=1.5)
        # Zoomed data
        ax_inset.loglog(freqx_synic[10:50], np.mean(np.squeeze(Euu_synic[10:50, zloc, astart:]) / utau**2, axis=1), '1', markersize=6, color=color[4])
        ax_inset.loglog(freqx_linic[10:50], np.mean(np.squeeze(Euu_linic[10:50, zloc, astart:]) / utau**2, axis=1), '^', markersize=3, color=color[5])
        ax_inset.loglog(freqx_cansic[10:50], np.mean(np.squeeze(Euu_cansic[10:50, zloc, astart:]) / utau**2, axis=1), '-', color=color[6])
        ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)        
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.1")
    if save_Fig:
        plt.savefig(filename_panela)
#
# Time Figure
#
zloc = 100
plt.figure(2,figsize=(14,4))
plt.subplot(1,3,1)
plt.title("Log Profile")
for index in range(0,len(Euu_cansic[1,1,:])):
    plt.loglog(freqx_cansic[:zend],np.squeeze(Euu_cansic[:zend,zloc,index])/utau**2,'-',color=cmap(index/len(Euu_cansic[1,1,:])),alpha=0.2)
plt.loglog(freqx_cansic[:zend],np.mean(np.squeeze(Euu_cansic[:zend,zloc,astart:])/utau**2,axis=1),'k+',markersize=4,label=r'$\langle \overline{E}(k) \rangle_*$')
plt.loglog(freqx_cansic[:zend],np.squeeze(Euu_cansic[:zend,zloc,0])/utau**2,'m.',markersize=4,label=r'$T_{\epsilon} = 0$')
plt.xlabel(r'$k$',fontsize=25)
plt.xticks([1,10,100])
plt.ylabel(r'$ \langle E(k,t) \rangle_*$',fontsize=25)        
plt.grid()
plt.legend(frameon=False, markerscale=3)
plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2])
plt.ylim([1e-12,5e-2])
# Second plot
plt.subplot(1,3,2)
plt.title("Linear Profile")
for index in range(0,len(Euu_linic[1,1,:])):
    plt.loglog(freqx_linic[:zend],np.squeeze(Euu_linic[:zend,zloc,index])/utau**2,'-',color=cmap(index/len(Euu_linic[1,1,:])),alpha=0.2)
plt.loglog(freqx_linic[:zend],np.mean(np.squeeze(Euu_linic[:zend,zloc,astart:])/utau**2,axis=1),'k+',markersize=5)
plt.loglog(freqx_linic[:zend],np.squeeze(Euu_linic[:zend,zloc,0])/utau**2,'m.',markersize=4,label=r'$t = 0$')
plt.xlabel(r'$k$',fontsize=25)
plt.xticks([1,10,100])
plt.grid()
plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2],labels=[])
plt.ylim([1e-12,5e-2])
# Thirds plot
plt.subplot(1,3,3)
plt.title("Synthetic")
for index in range(0,len(Euu_synic[1,1,:])):
    plt.loglog(freqx_synic[:zend],np.squeeze(Euu_synic[:zend,zloc,index])/utau**2,'-',color=cmap(index/len(Euu_synic[1,1,:])),alpha=0.2)
plt.loglog(freqx_synic[:zend],np.mean(np.squeeze(Euu_synic[:zend,zloc,astart:])/utau**2,axis=1),'k+',markersize=5)
plt.loglog(freqx_synic[:zend],np.squeeze(Euu_synic[:zend,zloc,0])/utau**2,'m.',markersize=4,label=r'$t = 0$')
plt.xlabel(r'$k$',fontsize=25)
plt.xticks([1,10,100])
plt.grid()
plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2],labels=[])
plt.ylim([1e-12,5e-2])
# Specify the desired range for the colorbar
vmin = 0.0
vmax = 10.0
# Create a colorbar with the specified limits
cbar_ax = plt.figure(2).add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.set_label(r'$T_{\epsilon}$', fontsize=25)
plt.subplots_adjust(bottom=0.16)
#
# Show all plots
#
if(save_Fig):
        plt.savefig(filename_panelb)
if(show_Fig):
    plt.show()