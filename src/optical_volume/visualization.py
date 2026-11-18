"""Plotting helpers.

matplotlib functions need ``pip install optical-volume[plot]``; the 3D volume
renderers (``visualize_grid_vol``, ``visualize_grid_iso_surf``) also need mayavi
(``pip install optical-volume[mayavi]`` or ``conda install -c conda-forge mayavi``).
"""
import matplotlib.pyplot as plt
import numpy as np


def _mlab():
    try:
        from mayavi import mlab
    except ImportError as err:
        raise ImportError(
            "3D rendering needs mayavi: pip install optical-volume[mayavi] "
            "(or conda install -c conda-forge mayavi)"
        ) from err
    return mlab


# https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
def visualize_grid_vol(RI_distribution, support=None, n_background=1., factor=2, 
                       title="3D RI Distribution"):
    """3d visualization of the gird.

    Args:
        RI_distribution (3d array): 3d space with RI at each point.
        n_background (float, optional): background RI. Defaults to 1.
        factor (int, optional): downsampling, use every factor-th sample. Defaults to 2.
    """
    mlab = _mlab()
    x = np.arange(0, RI_distribution.shape[0], factor)
    y = np.arange(0, RI_distribution.shape[1], factor)
    z = np.arange(0, RI_distribution.shape[2], factor)
    
    # Axis extent in um
    # Plots differently (blurry). Not sure why.
    if support is not None:
        x = x*support[0]/RI_distribution.shape[0]/1e-6
        y = y*support[1]/RI_distribution.shape[1]/1e-6
        z = z*support[2]/RI_distribution.shape[2]/1e-6

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="ij")
    RI_abs = np.abs(RI_distribution - n_background)
        
    # Plot scatter with mayavi
    figure = mlab.figure(title)
    
    grid = mlab.pipeline.scalar_field(x_mesh, y_mesh, z_mesh, 
                                      RI_abs[::factor, ::factor, ::factor])

    mlab.pipeline.volume(grid, vmin=RI_abs.min(), vmax=RI_abs.max())
    
    mlab.axes()
    mlab.show()
    

# https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
def visualize_grid_iso_surf(RI_distribution, n_background=1., factor=2, 
                            title='3D Iso-surfaces Visualization'):
    """3d visualization of iso-surfaces.

    Args:
        RI_distribution (3d array): 3d space with RI at each point.
        n_background (float, optional): background RI. Defaults to 1.
        factor (int, optional): downsampling, use every factor-th sample. Defaults to 2.
    """
    mlab = _mlab()
    x = np.arange(RI_distribution.shape[0]//factor)
    y = np.arange(RI_distribution.shape[1]//factor)
    z = np.arange(RI_distribution.shape[2]//factor)

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="ij")
    RI_abs = np.abs(RI_distribution - n_background)
    # Plot scatter with mayavi
    figure = mlab.figure(title)
    
    mlab.contour3d(x_mesh, y_mesh, z_mesh, 
                                      RI_abs[::factor, ::factor, ::factor])
    mlab.axes()
    mlab.show()


def visualize_grid(RI_distribution, n_background=1., factor=10, angle=(30, 30, 0)):
    """3d-grid visualization. Will kill program for large grid.

    Args:
        RI_distribution (float): 3d space with RI at each point.
        n_background (float): background RI.
        factor (int): downsampling, use every factor-th sample.  
    """
    x = np.arange(RI_distribution.shape[0]//factor)[:, None, None]
    y = np.arange(RI_distribution.shape[1]//factor)[None, :, None]
    z = np.arange(RI_distribution.shape[2]//factor)[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=(np.abs(RI_distribution-n_background))[::factor, ::factor, ::factor]*factor*10, 
               alpha=0.25)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.view_init(angle[0], angle[1], angle[2])
    
    plt.show()


_UNIT_NAMES = {1.0: "m", 1e-3: "mm", 1e-6: "µm", 1e-9: "nm"}


def _to_numpy(field):
    if hasattr(field, "detach"):                     # torch tensor
        field = field.detach().cpu().numpy()
    return np.asarray(field)


def _show_xy(ax, data, support, units, **imshow_kwargs):
    """Plot an array indexed ``[x, y]`` with x to the right and y up (origin at the bottom left)."""
    extent = [0, support[0] / units, 0, support[1] / units]
    im = ax.imshow(np.asarray(data).T, origin="lower", extent=extent, **imshow_kwargs)
    unit = _UNIT_NAMES.get(units, f"{units:g} m")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    return im


def visualize_field(field, support, title="Field Intensity", units=1e-6, ax=None):
    """Plot the intensity ``|field|^2`` of a 2D field indexed ``[x, y]``.

    Parameters
    ----------
    field : array_like or Tensor
        ``(nx, ny)`` complex field.
    support : tuple of float
        Field size ``(Lx, Ly)`` in metres.
    title : str
        Axes title.
    units : float
        Length unit of the axes in metres (``1e-6`` for µm, ``1e-3`` for mm, ``1`` for m).
    ax : matplotlib Axes, optional
        Axes to draw into; default: a new figure.

    Returns
    -------
    matplotlib Axes
        x runs to the right and y up, with the origin (sample ``[0, 0]``) at the bottom left.
        Call ``plt.show()`` in a script to display it.
    """
    if ax is None:
        _, ax = plt.subplots()
    im = _show_xy(ax, np.abs(_to_numpy(field)) ** 2, support, units, cmap="gray")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    return ax


def visualize_complex_field(field, support, title="Complex Wavefield", units=1e-6):
    """Plot intensity and phase of a 2D field indexed ``[x, y]``, side by side.

    Parameters are as for :func:`visualize_field`. Returns the two Axes (intensity, phase),
    both with x to the right, y up and the origin at the bottom left.
    """
    field = _to_numpy(field)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    for ax, data, name in zip(axs, (np.abs(field) ** 2, np.angle(field)), ("intensity", "phase (rad)")):
        im = _show_xy(ax, data, support, units, cmap="gray")
        fig.colorbar(im, ax=ax)
        ax.set_title(name)
    fig.suptitle(title)
    return axs
