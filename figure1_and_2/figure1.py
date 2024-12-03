import numpy as np
import matplotlib.pyplot as plt
import cblind as cb
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
Retau=500.0
H = 1.0
nu = 1.0e-6
umax = 25.0
vmax = 3.0
wmax = 3.0
#
# LOAD ALL DATA
#
x = np.linspace(0,4*np.pi*H,1500); y = np.linspace(0,2*np.pi*H,1048); 
dummy = np.loadtxt('data/grid.out'); z = dummy[1:-1,:]
uslice10 = np.load('data/uslice_10.npy'); uslice20 = np.load('data/uslice_20.npy')
vslice10 = np.load('data/vslice_10.npy'); vslice20 = np.load('data/vslice_20.npy')
wslice10 = np.load('data/wslice_10.npy'); wslice20 = np.load('data/wslice_20.npy')
uslice80 = np.load('data/uslice_80.npy'); uslice128 = np.load('data/uslice_128.npy')
vslice80 = np.load('data/vslice_80.npy'); vslice128 = np.load('data/vslice_128.npy')
wslice80 = np.load('data/wslice_80.npy'); wslice128 = np.load('data/wslice_128.npy')
#
# COMPUTE PARAMETERS
#
utau = Retau*nu/H
#
# PLOTTING
#
fixPlot(thickness=2.0, fontsize=20, markersize=8, labelsize=15, texuse=True, tickSize = 15)
plt.figure(1,figsize=(19,8))
# - - - - - - - - - - - - - U SLICE
ax = plt.subplot(3,4,1)
pcm = ax.pcolor(x, y, np.squeeze(uslice10.T) / utau, cmap='magma')
pcm.set_clim([0, umax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[0,10,20]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$'])
plt.yticks([0,np.pi*H,2*np.pi*H],labels=[r'$0$',r'$L_y/2$',r'$L_y$'])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,2)
pcm = ax.pcolor(x, y, np.squeeze(uslice20.T) / utau, cmap='magma')
pcm.set_clim([0, umax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[0,10,20]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,3)
pcm = ax.pcolor(x, y, np.squeeze(uslice80.T) / utau, cmap='magma')
pcm.set_clim([0, umax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[0,10,20]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,4)
pcm = ax.pcolor(x, y, np.squeeze(uslice128.T) / utau, cmap='magma')
pcm.set_clim([0, umax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[0,10,20]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
# - - - - - - - - - - - - - V SLICE
ax = plt.subplot(3,4,5)
pcm = ax.pcolor(x, y, np.squeeze(vslice10.T) / utau, cmap='ocean')
pcm.set_clim([-vmax, vmax]);  
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$'])
plt.yticks([0,np.pi*H,2*np.pi*H],labels=[r'$0$',r'$L_y/2$',r'$L_y$'])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,6)
pcm = ax.pcolor(x, y, np.squeeze(vslice20.T) / utau, cmap='ocean')
pcm.set_clim([-vmax, vmax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,7)
pcm = ax.pcolor(x, y, np.squeeze(vslice80.T) / utau, cmap='ocean')
pcm.set_clim([-vmax, vmax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,8)
pcm = ax.pcolor(x, y, np.squeeze(vslice128.T) / utau, cmap='ocean')
pcm.set_clim([-vmax, vmax]);  
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
# - - - - - - - - - - - - - W SLICE
ax = plt.subplot(3,4,9)
pcm = ax.pcolor(x, y, np.squeeze(wslice10.T) / utau, cmap='nipy_spectral')
pcm.set_clim([-vmax, vmax]);  
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$'])
plt.yticks([0,np.pi*H,2*np.pi*H],labels=[r'$0$',r'$L_y/2$',r'$L_y$'])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,10)
pcm = ax.pcolor(x, y, np.squeeze(wslice20.T) / utau, cmap='nipy_spectral')
pcm.set_clim([-vmax, vmax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,11)
pcm = ax.pcolor(x, y, np.squeeze(wslice80.T) / utau, cmap='nipy_spectral')
pcm.set_clim([-vmax, vmax]); 
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
ax = plt.subplot(3,4,12)
pcm = ax.pcolor(x, y, np.squeeze(vslice128.T) / utau, cmap='nipy_spectral')
pcm.set_clim([-vmax, vmax]);  
plt.colorbar(pcm, ax=ax, pad=0.08, fraction=0.05, shrink=0.75,ticks=[-2.5,0,2.5]) 
plt.plot([x[40],x[40]+(4*100.0*nu/utau)],[0.25,0.25],color='white')
plt.plot([x[40],x[40]],[0.25,0.25+(4*100.0*nu/utau)],color='white')
plt.xticks([0,2*np.pi*H,4*np.pi*H],labels=[r'$0$',r'$L_x/2$',r'$L_x$']); plt.yticks([])
plt.xlim([0,x[-1]]);plt.ylim([0,y[-1]])
plt.gca().set_aspect(1.0)
if(save_Fig):
    plt.savefig('figure_1.png',dpi=600)
if(show_Fig):
    plt.show()