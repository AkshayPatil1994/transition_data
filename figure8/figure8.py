import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cblind as cb
import matplotlib.pyplot as plt
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
filename = 'figure8.png'
kappa = 0.4
H = 1.0
kvisc = 1.0e-6
utau_re350 = 3.5e-4
utau_re500 = 5.0e-4
plotstartindex = 1690
plotstartindex_re500 = 2400
rms_startindex_re350 = 80
tloc = 0
#
# LOAD ALL DATA
#
zo_re350 = kvisc / (9 * utau_re350)
zo_re500 = kvisc / (9 * utau_re500)
Uo_re350 = 2.5 * utau_re350 * (np.log(H / zo_re350) + zo_re350 / H - 1)
Uo_re500 = 2.5 * utau_re500 * (np.log(H / zo_re500) + zo_re500 / H - 1)
linear_shearmagnitude_re350 = -2 * Uo_re350 * kvisc / H
linear_shearmagnitude_re500 = -2 * Uo_re500 * kvisc / H
grid_re350 = np.loadtxt('../figure3_and4/data/grid.out')
grid_re500 = np.loadtxt('../figure3_and4/data/grid_re500.out')
z_re350 = grid_re350[1:-1, 2]
z_re500 = grid_re500[1:-1, 2]
#
# Mean velocity data - Re350
#
U_linic_re350 = np.load('../figure3_and4/data/linic/Re350/Uplan.npy')
U_synthetic_re350 = np.load('../figure3_and4/data/syntheticic/Re350/Uplan.npy')
U_cansic_re350 = np.zeros_like(U_linic_re350)
# Loop setup
for i in range(0,len(z_re350)):
    if(z_re350[i]*350 <= 11.6):
        U_cansic_re350[i] = z_re350[i]*350*3.5e-4
    else:
        U_cansic_re350[i] = (3.5e-4/0.4)*np.log(z_re350[i]*350)+5.2
#
# Mean velocity data - Re500
#
U_linic_re500 = np.load('../figure3_and4/data/linic/Re500/Uplan.npy')
U_synthetic_re500 = np.load('../figure3_and4/data/syntheticic/Re500/Uplan.npy')
U_cansic_re500 = np.zeros_like(U_linic_re500)
# Loop setup
for i in range(0,len(z_re500)):
    if(z_re500[i]*500 <= 11.6):
        U_cansic_re500[i] = z_re500[i]*500*5e-4
    else:
        U_cansic_re500[i] = (5e-4/0.4)*np.log(z_re500[i]*500)+5.2

log_shear_re350 = ((utau_re350 ** 2) / kappa) * (kvisc / (utau_re350 * z_re350))
log_shear_re500 = ((utau_re500 ** 2) / kappa) * (kvisc / (utau_re500 * z_re500))

color, linestyle = cb.Colorplots().cblind(10)
fixPlot(thickness=2.5, fontsize=25, markersize=8, labelsize=15, texuse=True, tickSize=15)

fig, ax = plt.subplots(figsize=(10, 7))

# Re 350
ax.semilogx(z_re350 * utau_re350 / kvisc, np.gradient(U_synthetic_re350[:, tloc], z_re350) * kvisc / utau_re350 ** 2, '1', color=color[0], markersize=4, label=r'$Re_{\tau} = 350$ - Synthetic')
ax.semilogx(z_re350 * utau_re350 / kvisc, np.gradient(U_linic_re350[:, tloc], z_re350) * kvisc / utau_re350 ** 2, '^', color=color[1], markersize=4, label=r'$Re_{\tau} = 350$ - Linear Profile')
ax.semilogx(z_re350 * utau_re350 / kvisc, np.gradient(U_cansic_re350[:, tloc], z_re350) * kvisc / utau_re350 ** 2, '.', color=color[2], label=r'$Re_{\tau} = 350$ - Log Profile')

# Re 500
ax.semilogx(z_re500 * utau_re500 / kvisc, np.gradient(U_synthetic_re500[:, tloc], z_re500) * kvisc / utau_re500 ** 2, '1', color=color[3], markersize=4, label=r'$Re_{\tau} = 500$ - Synthetic')
ax.semilogx(z_re500 * utau_re500 / kvisc, np.gradient(U_linic_re500[:, tloc], z_re500) * kvisc / utau_re500 ** 2, '^', color=color[4], markersize=4, label=r'$Re_{\tau} = 500$ - Linear Profile')
ax.semilogx(z_re500 * utau_re500 / kvisc, np.gradient(U_cansic_re500[:, tloc], z_re500) * kvisc / utau_re500 ** 2, '.', color=color[5], label=r'$Re_{\tau} = 500$ - Log Profile')

# FORMATTING
ax.legend(frameon=False, loc='upper right', fontsize=15, ncols=2)
ax.set_xlabel(r'$x_3^+$', fontsize=25)
ax.set_ylabel(r'$\nu \partial_3 U_1^+ $', fontsize=25)
ax.set_ylim([-0.2, 2.0])
ax.set_yticks([-0.2, 0, 0.5, 1.0, 1.5, 2.0])
ax.grid()
if(save_Fig):
    plt.savefig(filename)
if(show_Fig):
    plt.show()