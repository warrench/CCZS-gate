import itertools as it
import numpy as np
from numpy import array, log2

from packaging.version import parse as parse_version

from qutip.qobj import Qobj
# from qutip.matplotlib_utilities import complex_phase_cmap
from qutip.superoperator import vector_to_operator
from qutip.superop_reps import _super_to_superpauli, _isqubitdims

from qutip import settings

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from matplotlib import colors
    from mpl_toolkits.mplot3d import Axes3D


    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(mpl.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)
except:
    pass




plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics}')


def plot_comparison(a, b, title=""):
    """Plot a comparision between Choi matrices

    Args:
        a ([type]): [description]
        b ([type]): [description]
    """
    plt.clf()
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (7.5, 7.5))

    cmap = "viridis"

    norm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(a)), vcenter=0, vmax=np.max(np.abs(a)))

    ax[0, 0].matshow(a.real, cmap=cmap, norm=norm)
    im = ax[0, 1].matshow(b.real, cmap=cmap, norm=norm)


    ax[1, 0].matshow(a.imag, cmap=cmap, norm=norm)
    im = ax[1, 1].matshow(b.imag, cmap=cmap, norm = norm)
    plt.colorbar(im, ax=[axis for axis in ax.ravel()], fraction=0.046, pad=0.04)

    ax[0, 0].set_title(r"Choi", fontsize=16)
    ax[0, 1].set_title(r"Choi (Reconstruction)", fontsize=16)
    ax[0, 0].set_ylabel(r"$Re[\Phi]$", fontsize=16)
    # ax[0, 1].set_xlabel("Re")

    ax[1, 0].set_ylabel(r"$Im[\Phi]$", fontsize=16)
    # ax[1, 1].set_xlabel("Im")

    plt.suptitle(title, y=0.95, fontsize=20)
    plt.show()




# Plotting functionality for a modified hinton to show phase information in colour
# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, cmap=None, ax=None, complex=False):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = array([x - hs, x + hs, x + hs, x - hs])
    ycorners = array([y - hs, y - hs, y + hs, y + hs])

    if ax is not None:
        handle = ax
    else:
        handle = plt

    if complex:
        norm = mpl.colors.Normalize(-np.pi, np.pi)
        handle.fill(xcorners, ycorners,
                   color=cmap(norm(w)))
    else:
        handle.fill(xcorners, ycorners,
                color=cmap(int((w + w_max) * 256 / (2 * w_max))))

def _add_text(x, y, val, thresh=0, ax=None):
    if ax is not None:
        handle = ax
    else:
        handle = plt
    colors = ['black', 'white']
    if np.abs(val) < 0.01:
        pass
    else:
        text = ax.text(x,
                       y,
                       r'{:.3f}'.format(np.abs(val)),
                       ha='center',
                       va='center',
                       color=colors[int((np.angle(val)<thresh) and (np.abs(val)>0.25))],
                       fontsize=14)


def complex_phase_cmap(cmap=None):
    # Modified from qutip.matplotlib_utilities to accept any cmap
    """
    Create a cyclic colormap for representing the phase of complex variables

    Returns
    -------
    cmap :
        A matplotlib linear segmented colormap.
    """
    if cmap is None:
        cdict = {'blue': ((0.00, 0.0, 0.0),
                          (0.25, 0.0, 0.0),
                          (0.50, 1.0, 1.0),
                          (0.75, 1.0, 1.0),
                          (1.00, 0.0, 0.0)),
                 'green': ((0.00, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 0.0)),
                 'red': ((0.00, 1.0, 1.0),
                         (0.25, 0.5, 0.5),
                         (0.50, 0.0, 0.0),
                         (0.75, 0.0, 0.0),
                         (1.00, 1.0, 1.0))}

        cmap_new = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)
    else:
        cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=0,b=1),
                cmap(np.linspace(0, 1, 5)), 256)

    return cmap_new



def _cb_labels(left_dims):
    """Creates plot labels for matrix elements in the computational basis.

    Parameters
    ----------
    left_dims : flat list of ints
        Dimensions of the left index of a density operator. E. g.
        [2, 3] for a qubit tensored with a qutrit.

    Returns
    -------
    left_labels, right_labels : lists of strings
        Labels for the left and right indices of a density operator
        (kets and bras, respectively).
    """
    # FIXME: assumes dims, such that we only need left_dims == dims[0].
    basis_labels = list(map("".join, it.product(*[
        map(str, range(dim))
        for dim in left_dims
    ])))
    return [
        map(fmt.format, basis_labels) for fmt in
        (
            r"$|{}\rangle$",
            r"$\langle{}|$"
        )
    ]

def hinton_phase(rho, xlabels=None, ylabels=None, title=None, ax=None, cmap=None,
                 label_top=True, phase_limits=None, ideal=None, cbar=True, text=False):
    """Draws a Hinton diagram for visualizing a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

    xlabels : list of strings or False
        list of x labels

    ylabels : list of strings or False
        list of y labels

    title : string
        title of the plot (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    cmap : a matplotlib colormap instance
        Color map to use when plotting.

    label_top : bool
        If True, x-axis labels will be placed on top, otherwise
        they will appear below the plot.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    """

    # Apply default colormaps.
    # TODO: abstract this away into something that makes default
    #       colormaps.
    cmap = (
        (cm.Greys_r if settings.colorblind_safe else cm.RdBu)
        if cmap is None else cmap
    )

    # Extract plotting data W from the input.
    if isinstance(rho, Qobj):
        if rho.isoper:
            W = rho.full()

            # Create default labels if none are given.
            if xlabels is None or ylabels is None:
                labels = _cb_labels(rho.dims[0])
                xlabels = xlabels if xlabels is not None else list(labels[0])
                ylabels = ylabels if ylabels is not None else list(labels[1])

        elif rho.isoperket:
            W = vector_to_operator(rho).full()
        elif rho.isoperbra:
            W = vector_to_operator(rho.dag()).full()
        elif rho.issuper:
            if not _isqubitdims(rho.dims):
                raise ValueError("Hinton plots of superoperators are "
                                 "currently only supported for qubits.")
            # Convert to a superoperator in the Pauli basis,
            # so that all the elements are real.
            sqobj = _super_to_superpauli(rho)
            nq = int(log2(sqobj.shape[0]) / 2)
            W = sqobj.full().T
            # Create default labels, too.
            if (xlabels is None) or (ylabels is None):
                labels = list(map("".join, it.product("IXYZ", repeat=nq)))
                xlabels = xlabels if xlabels is not None else labels
                ylabels = ylabels if ylabels is not None else labels

        else:
            raise ValueError(
                "Input quantum object must be an operator or superoperator."
            )

    else:
        W = rho


    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    else:
        fig = None

    if not (xlabels or ylabels):
        ax.axis('off')

    ax.axis('equal')
    ax.set_frame_on(False)

    height, width = W.shape


    if cmap is None:
        cmap = complex_phase_cmap()
    else:
        cmap = complex_phase_cmap(cmap)


    # colors = cmap(norm(angle(W)))
    if ideal is not None:
        w_max = 1.25 * max(abs(np.diag(np.array(ideal))))
    else:
        w_max = 1.25 * max(abs(np.diag(np.array(W))))
    if w_max <= 0.0:
        w_max = 1.0

    ax.fill(array([0, width, width, 0]), array([0, 0, height, height]),
            color='gray', alpha=0.1)
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            # print(min(1, abs(W[x, y])))
            _blob(_x - 0.5,
                  height - _y + 0.5,
                  np.angle(W[x, y]),
                  w_max,
                  min(1, abs(W[x, y])/w_max),
                  cmap=cmap, ax=ax, complex=True)
            if ideal is not None:
                # print(min(1, abs(ideal[x, y])))
                _patch(_x - 0.5,
                       height - _y + 0.5,
                       abs(ideal[x, y])/w_max,
                       ax=ax)
            if text:
                _add_text(_x - 0.5,
                          height - _y + 0.5+0.25,
                          W[x,y],
                          thresh=0.25,
                          ax=ax)
            # if np.real(W[x, y]) > 0.0:
            #     _blob(_x - 0.5, height - _y + 0.5, abs(W[x, y]), w_max,
            #           min(1, abs(W[x, y]) / w_max), cmap=cmap, ax=ax, complex=True)
            # else:
            #     _blob(_x - 0.5, height - _y + 0.5, -abs(W[
            #           x, y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap, ax=ax, complex=True)

    # color axis
    norm = mpl.colors.Normalize(-np.pi, np.pi)

    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=0.15, anchor=(-1,0.5))
    base = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)
    base.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    base.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    base.ax.tick_params(labelsize=20)
    base.ax.set_title('Phase', fontsize=24)

    if not cbar:
        base.remove()



    xtics = 0.5 + np.arange(width)
    # x axis
    ax.xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=20)

    # y axis
    ytics = 0.5 + np.arange(height)
    ax.yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=20)

    ax.tick_params(axis='both', length=0)

    return fig, ax