import matplotlib.pyplot as plt
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS
from matplotlib.projections import get_projection_names
from types import MethodType

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def init_sky(projection='mollweide', ra_center=120,
             galactic_plane_color='red', ecliptic_plane_color='red',
             ax=None):
    """Initialize matplotlib axes with a projection of the full sky.

    Parameters
    ----------
    projection : :class:`str`, optional
        Projection to use. Defaults to 'mollweide'.  To show the available projections,
        call :func:`matplotlib.projections.get_projection_names`.
    ra_center : :class:`float`, optional
        Projection is centered at this RA in degrees. Default is +120°, which avoids splitting
        the DESI northern and southern regions.
    galactic_plane_color : color name, optional
        Draw a solid curve representing the galactic plane using the specified color, or do
        nothing when ``None``.
    ecliptic_plane_color : color name, optional
        Draw a dotted curve representing the ecliptic plane using the specified color, or do
        nothing when ``None``.
    ax : :class:`~matplotlib.axes.Axes`, optional
        Axes to use for drawing this map, or create new axes if ``None``.

    Returns
    -------
    :class:`~matplotlib.axes.Axes`
        A matplotlib Axes object.  Helper methods ``projection_ra()`` and ``projection_dec()``
        are added to the object to facilitate conversion to projection coordinates.

    Notes
    -----
    If requested, the ecliptic and galactic planes are plotted with ``zorder`` set to 20.
    This keeps them above most other plotted objects, but legends should be set to
    a ``zorder`` higher than this value, for example::

        leg = ax.legend(ncol=2, loc=1)
        leg.set_zorder(25)
    """
    #
    # Internal functions.
    #
    def projection_ra(self, ra):
        r"""Shift `ra` to the origin of the Axes object and convert to radians.

        Parameters
        ----------
        ra : array-like
            Right Ascension in degrees.

        Returns
        -------
        array-like
            `ra` converted to plot coordinates.

        Notes
        -----
        In matplotlib, map projections expect longitude (RA), latitude (Dec)
        in radians with limits :math:`[-\pi, \pi]`, :math:`[-\pi/2, \pi/2]`,
        respectively.
        """
        #
        # Shift RA values.
        #
        r = np.remainder(ra + 360 - ra_center, 360)
        #
        # Scale conversion to [-180, 180].
        #
        r[r > 180] -= 360
        #
        # Reverse the scale: East to the left.
        #
        r = -r
        return np.radians(r)

    def projection_dec(self, dec):
        """Shift `dec` to the origin of the Axes object and convert to radians.

        Parameters
        ----------
        dec : array-like
            Declination in degrees.

        Returns
        -------
        array-like
            `dec` converted to plot coordinates.
        """
        return np.radians(dec)
    #
    # Create ax.
    #
    if ax is None:
        fig = plt.figure(figsize=(10.0, 5.0), dpi=100)
        ax = plt.subplot(111, projection=projection)
    #
    # Prepare labels.
    #
    base_tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    base_tick_labels = np.remainder(base_tick_labels+360+ra_center, 360)
    tick_labels = np.array(['{0}°'.format(l) for l in base_tick_labels])
    #
    # Galactic plane.
    #
    if galactic_plane_color is not None:
        galactic_l = np.linspace(0, 2 * np.pi, 1000)
        galactic = SkyCoord(l=galactic_l*u.radian, b=np.zeros_like(galactic_l)*u.radian,
                            frame='galactic').transform_to(ICRS)
        #
        # Project to map coordinates and display.  Use a scatter plot to
        # avoid wrap-around complications.
        #
        paths = ax.scatter(projection_ra(0, galactic.ra.degree),
                           projection_dec(0, galactic.dec.degree),
                           marker='.', s=10, lw=1, alpha=0.75,
                           c=galactic_plane_color, zorder=20,label='Galactic Plane')
        
        # Make sure the galactic plane stays above other displayed objects.
        # paths.set_zorder(20)
    #
    # Ecliptic plane.
    #
    if ecliptic_plane_color is not None:
        ecliptic_l = np.linspace(0, 2 * np.pi, 50)
        ecliptic = SkyCoord(lon=ecliptic_l*u.radian, lat=np.zeros_like(ecliptic_l)*u.radian, distance=1 * u.Mpc,
                            frame='heliocentrictrueecliptic').transform_to(ICRS)
        #
        # Project to map coordinates and display.  Use a scatter plot to
        # avoid wrap-around complications.
        #
        paths = ax.scatter(projection_ra(0, ecliptic.ra.degree),
                           projection_dec(0, ecliptic.dec.degree),
                           marker=".", s=10, lw=1, alpha=0.75,
                           c=ecliptic_plane_color, zorder=20,label='Ecliptic Plane')
       
        # paths.set_zorder(20)
    #
    # Set RA labels.
    #
    labels = ax.get_xticklabels()
    for l, item in enumerate(labels):
        item.set_text(tick_labels[l])
    ax.set_xticklabels(labels,size=12)
    #
    # Set axis labels.
    #
    ax.set_xlabel('R.A. [deg]',fontsize=12)
    # ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel('Dec. [deg]',fontsize=12)
    # ax.yaxis.label.set_fontsize(12)
    ax.grid(True)
    #
    # Attach helper methods.
    #
    if hasattr(ax, '_ra_center'):
        warnings.warn("Attribute '_ra_center' detected.  Will be overwritten!")
    ax._ra_center = ra_center
    if hasattr(ax, 'projection_ra'):
        warnings.warn("Attribute 'projection_ra' detected.  Will be overwritten!")
    ax.projection_ra = MethodType(projection_ra, ax)
    if hasattr(ax, 'projection_dec'):
        warnings.warn("Attribute 'projection_dec' detected.  Will be overwritten!")
    ax.projection_dec = MethodType(projection_dec, ax)
    return fig , ax