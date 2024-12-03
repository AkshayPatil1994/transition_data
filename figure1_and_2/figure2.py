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
# USER INPUT
#
save_Fig = False
show_Fig = True
Retau=500.0
H = 1.0
nu = 1.0e-6
#
# LOAD ALL DATA
#
dummy = np.loadtxt('data/grid.out'); z = dummy[1:-1,:]
U=np.loadtxt('data/Umean.dat')
urms=np.loadtxt('data/urmsprof.dat')
vrms=np.loadtxt('data/vrmsprof.dat')
wrms=np.loadtxt('data/wrmsprof.dat')
uw=np.loadtxt('data/uw.dat')
meanref = np.loadtxt('data/mean_interpolated_re500.dat',skiprows=1)
covarref = np.loadtxt('data/covar_interpolated_re500.dat',skiprows=1)
#
# SOME COMPUTATIONS
#
utau = Retau*nu/H
#
# PLOTTING
#
lcolor, lstyle = cb.Colorplots().cblind(10)
fixPlot(thickness=2.0, fontsize=20, markersize=5, labelsize=15, texuse=True, tickSize = 15)
plt.figure(1,figsize=(14,7))
plt.subplot(1,2,1)
plt.semilogx(z[:,1]*Retau/H,U/utau,'+',markerfacecolor='None',label='STFG')
plt.semilogx(z[:10,1]*Retau/H,z[:10,1]*Retau/H,'--',label=r'$u_1^+ = x_3^+$')
plt.semilogx(z[20:,1]*Retau/H,2.5*np.log(z[20:,1]*Retau/H)+5.2,'--',label=r'$u_1^+ = \frac{1}{\kappa} log(x_3^+) + 5.2$')
# Formatting
plt.legend(frameon=False,ncols=1,fontsize=15)
plt.xlim([0,Retau*1.4])
plt.xlabel(r'$x_3^+$',fontsize=30); plt.ylabel(r'$\langle u_1^+ \rangle$',fontsize=30)
plt.grid()
plt.subplot(1,2,2)
plt.plot(urms/utau,z[:,1]*Retau/H,'d',markerfacecolor='None',label=r'$\langle u^{rms,+}_1 \rangle^+$')
plt.plot(vrms/utau,z[:,1]*Retau/H,'<',markerfacecolor='None',label=r'$\langle u^{rms,+}_2 \rangle^+$')
plt.plot(wrms/utau,z[:,1]*Retau/H,'x',markerfacecolor='None',label=r'$\langle u^{rms,+}_3 \rangle^+$')
plt.plot(uw/utau**2,z[:,1]*Retau/H,'s',markerfacecolor='None',label=r'$\langle u^{\prime}_1 u^{\prime}_3 \rangle^+$')
# Reference Data
plt.plot(z[:,1]/H-1.0,z[:,1]*Retau/H,'k--',label='Linear Stress')
plt.plot(np.sqrt(covarref[:,2]),covarref[:,1]*Retau/H,'k-')
plt.plot(np.sqrt(covarref[:,3]),covarref[:,1]*Retau/H,'k-')
plt.plot(np.sqrt(covarref[:,4]),covarref[:,1]*Retau/H,'k-')
plt.plot(covarref[:,5],covarref[:,1]*Retau/H,'k-')
# Formatting
plt.grid()
plt.legend(frameon=False,ncols=1,fontsize=15)
plt.ylim([0,Retau]);plt.xlim([-1,3.0])
plt.ylabel(r'$x_3^+$',fontsize=30)
plt.tight_layout()
if(save_Fig):
    plt.savefig('figure_2.png',dpi=600)
if(show_Fig):
    plt.show()