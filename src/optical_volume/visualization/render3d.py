"""3D volume rendering with mayavi (``pip install optical-volume[mayavi]`` or
``conda install -c conda-forge mayavi``). mayavi is imported only when a function is called."""
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
