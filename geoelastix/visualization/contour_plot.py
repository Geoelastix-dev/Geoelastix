"""Contour plot visualization for displacement fields."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from pathlib import Path

logger = logging.getLogger("geoelastix.visualization")


class ContourPlot:
    """
    Create contour plots for displacement fields.

    Visualizes X, Y, Z displacement and magnitude as filled contour maps.
    """

    @staticmethod
    def create_contour_plot(data, title="Displacement", dpi=300,
                           figsize=(10, 8), colormap='RdBu_r',
                           levels=20, mask=None, show_contour_lines=True,
                           colorbar_label="Displacement", vmin=None, vmax=None,
                           center_zero=False, pixel_spacing=None,
                           geotransform=None):
        """
        Create filled contour plot.

        Parameters
        ----------
        data : numpy.ndarray
            Displacement data
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI. Default: 300
        figsize : tuple, optional
            Figure size. Default: (10, 8)
        colormap : str, optional
            Colormap name. Default: 'RdBu_r'
        levels : int, optional
            Number of contour levels. Default: 20
        mask : numpy.ndarray, optional
            Binary mask for valid regions
        show_contour_lines : bool, optional
            Show contour lines. Default: True
        colorbar_label : str, optional
            Colorbar label
        vmin : float, optional
            Minimum value for color scale
        vmax : float, optional
            Maximum value for color scale
        center_zero : bool, optional
            Center colormap at zero. Default: False
        pixel_spacing : tuple, optional
            Pixel spacing (dx, dy) in real-world units. Default: None (uses pixels)
        geotransform : tuple, optional
            GDAL geotransform. Default: None

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info(f"Creating contour plot: {title}...")

        # Apply mask
        plot_data = data.copy()
        if mask is not None:
            plot_data[mask == 0] = np.nan

        # Determine color scale
        if vmin is None:
            vmin = np.nanpercentile(plot_data, 2)
        if vmax is None:
            vmax = np.nanpercentile(plot_data, 98)

        # Center colormap at zero if requested
        if center_zero:
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max

        # Get dimensions
        height, width = plot_data.shape

        # Determine coordinate system and units
        if pixel_spacing is not None:
            # Use real-world coordinates
            dx, dy = pixel_spacing
            # Create extent for imshow-style display
            # extent = [left, right, bottom, top]
            # For DEM: x from 0 to width*dx, y from height*dy to 0 (flipped)
            extent = [0, width * dx, height * abs(dy), 0]

            # Units follow the raster CRS (e.g. m for UTM, ft for State Plane)
            unit = 'map units'

            xlabel = f'Easting ({unit})'
            ylabel = f'Northing ({unit})'

            # Update colorbar label with units if not already specified
            if colorbar_label == "Displacement" and unit != 'pixels':
                colorbar_label = f'Displacement ({unit})'
        else:
            # Use pixel coordinates
            extent = [0, width, height, 0]
            unit = 'pixels'
            xlabel = 'X (pixels)'
            ylabel = 'Y (pixels)'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Create contour levels
        contour_levels = np.linspace(vmin, vmax, levels)

        # Create coordinate grids for contour plotting
        if pixel_spacing is not None:
            x = np.linspace(0, width * dx, width)
            y = np.linspace(0, height * abs(dy), height)
        else:
            x = np.arange(width)
            y = np.arange(height)

        X, Y = np.meshgrid(x, y)

        # Filled contours
        cf = ax.contourf(X, Y, plot_data, levels=contour_levels, cmap=colormap,
                        vmin=vmin, vmax=vmax, extend='both')

        # Contour lines
        if show_contour_lines:
            cs = ax.contour(X, Y, plot_data, levels=contour_levels[::2],
                           colors='black', linewidths=0.5, alpha=0.3)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

        # Colorbar
        cbar = plt.colorbar(cf, ax=ax, label=colorbar_label)

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set aspect ratio
        ax.set_aspect('equal')

        # Invert y-axis for DEM convention (North up)
        ax.invert_yaxis()

        plt.tight_layout()

        logger.info(f"  Contour plot created")

        return fig

    @staticmethod
    def create_displacement_contours(displacement_x, displacement_y, displacement_z,
                                    magnitude=None, dpi=300, figsize=(16, 12),
                                    colormap='RdBu_r', levels=20, mask=None):
        """
        Create multi-panel contour plot for all displacement components.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement (East-West)
        displacement_y : numpy.ndarray
            Y displacement (North-South)
        displacement_z : numpy.ndarray
            Z displacement (Vertical)
        magnitude : numpy.ndarray, optional
            Horizontal magnitude. If None, computed from X and Y.
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        colormap : str, optional
            Colormap name
        levels : int, optional
            Number of contour levels
        mask : numpy.ndarray, optional
            Binary mask

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info("Creating multi-panel displacement contours...")

        # Compute magnitude if not provided
        if magnitude is None:
            magnitude = np.sqrt(displacement_x**2 + displacement_y**2)

        # Apply mask
        if mask is not None:
            dx = displacement_x.copy()
            dy = displacement_y.copy()
            dz = displacement_z.copy()
            mag = magnitude.copy()

            dx[mask == 0] = np.nan
            dy[mask == 0] = np.nan
            dz[mask == 0] = np.nan
            mag[mask == 0] = np.nan
        else:
            dx = displacement_x
            dy = displacement_y
            dz = displacement_z
            mag = magnitude

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        axes = axes.flatten()

        # Plot X displacement
        vmin_x = np.nanpercentile(dx, 2)
        vmax_x = np.nanpercentile(dx, 98)
        abs_max_x = max(abs(vmin_x), abs(vmax_x))

        cf1 = axes[0].contourf(dx, levels=np.linspace(-abs_max_x, abs_max_x, levels),
                               cmap=colormap, extend='both')
        axes[0].set_title('X Displacement (East-West)', fontweight='bold')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        axes[0].set_aspect('equal')
        plt.colorbar(cf1, ax=axes[0], label='Displacement (m)')

        # Plot Y displacement
        vmin_y = np.nanpercentile(dy, 2)
        vmax_y = np.nanpercentile(dy, 98)
        abs_max_y = max(abs(vmin_y), abs(vmax_y))

        cf2 = axes[1].contourf(dy, levels=np.linspace(-abs_max_y, abs_max_y, levels),
                               cmap=colormap, extend='both')
        axes[1].set_title('Y Displacement (North-South)', fontweight='bold')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        axes[1].set_aspect('equal')
        plt.colorbar(cf2, ax=axes[1], label='Displacement (m)')

        # Plot Z displacement
        vmin_z = np.nanpercentile(dz, 2)
        vmax_z = np.nanpercentile(dz, 98)
        abs_max_z = max(abs(vmin_z), abs(vmax_z))

        cf3 = axes[2].contourf(dz, levels=np.linspace(-abs_max_z, abs_max_z, levels),
                               cmap=colormap, extend='both')
        axes[2].set_title('Z Displacement (Vertical)', fontweight='bold')
        axes[2].set_xlabel('X (pixels)')
        axes[2].set_ylabel('Y (pixels)')
        axes[2].set_aspect('equal')
        plt.colorbar(cf3, ax=axes[2], label='Displacement (m)')

        # Plot magnitude (no centering at zero)
        vmin_mag = 0
        vmax_mag = np.nanpercentile(mag, 98)

        cf4 = axes[3].contourf(mag, levels=np.linspace(vmin_mag, vmax_mag, levels),
                               cmap='viridis', extend='max')
        axes[3].set_title('Horizontal Magnitude', fontweight='bold')
        axes[3].set_xlabel('X (pixels)')
        axes[3].set_ylabel('Y (pixels)')
        axes[3].set_aspect('equal')
        plt.colorbar(cf4, ax=axes[3], label='Magnitude (m)')

        # Main title
        fig.suptitle('Displacement Field Analysis', fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        logger.info("  Multi-panel contour plot created")

        return fig

    @staticmethod
    def create_hillshade_overlay(data, title="Displacement with Hillshade",
                                 dpi=300, figsize=(10, 8), colormap='RdBu_r',
                                 levels=20, mask=None, azimuth=315, altitude=45):
        """
        Create contour plot with hillshade overlay for 3D effect.

        Parameters
        ----------
        data : numpy.ndarray
            Displacement data
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        colormap : str, optional
            Colormap name
        levels : int, optional
            Number of contour levels
        mask : numpy.ndarray, optional
            Binary mask
        azimuth : float, optional
            Light source azimuth angle. Default: 315
        altitude : float, optional
            Light source altitude angle. Default: 45

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info("Creating contour plot with hillshade...")

        from matplotlib.colors import LightSource

        # Apply mask
        plot_data = data.copy()
        if mask is not None:
            plot_data[mask == 0] = np.nan

        # Create hillshade
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        hillshade = ls.hillshade(plot_data, vert_exag=1.0, dx=1, dy=1)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Display hillshade
        ax.imshow(hillshade, cmap='gray', alpha=0.5)

        # Overlay contours
        vmin = np.nanpercentile(plot_data, 2)
        vmax = np.nanpercentile(plot_data, 98)
        abs_max = max(abs(vmin), abs(vmax))

        cf = ax.contourf(plot_data, levels=np.linspace(-abs_max, abs_max, levels),
                        cmap=colormap, alpha=0.7, extend='both')

        # Colorbar
        cbar = plt.colorbar(cf, ax=ax, label='Displacement')

        # Labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_aspect('equal')

        plt.tight_layout()

        logger.info("  Hillshade overlay plot created")

        return fig

    @staticmethod
    def save_plot(fig, output_path):
        """
        Save figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        output_path : str or Path
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, bbox_inches='tight', dpi=fig.dpi)
        plt.close(fig)

        logger.info(f"Contour plot saved to: {output_path}")

    @staticmethod
    def create_difference_map(data1, data2, title="Difference Map",
                             dpi=300, figsize=(10, 8), colormap='RdBu_r',
                             levels=20, mask=None, label1="Dataset 1",
                             label2="Dataset 2"):
        """
        Create contour plot showing difference between two datasets.

        Parameters
        ----------
        data1 : numpy.ndarray
            First dataset
        data2 : numpy.ndarray
            Second dataset
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        colormap : str, optional
            Colormap name
        levels : int, optional
            Number of contour levels
        mask : numpy.ndarray, optional
            Binary mask
        label1 : str, optional
            Label for dataset 1
        label2 : str, optional
            Label for dataset 2

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info(f"Creating difference map: {label1} - {label2}...")

        # Compute difference
        difference = data1 - data2

        # Apply mask
        if mask is not None:
            difference[mask == 0] = np.nan

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Determine symmetric color scale
        vmin = np.nanpercentile(difference, 2)
        vmax = np.nanpercentile(difference, 98)
        abs_max = max(abs(vmin), abs(vmax))

        # Filled contours
        cf = ax.contourf(difference, levels=np.linspace(-abs_max, abs_max, levels),
                        cmap=colormap, extend='both')

        # Contour lines
        cs = ax.contour(difference, levels=np.linspace(-abs_max, abs_max, levels)[::2],
                       colors='black', linewidths=0.5, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(cf, ax=ax, label=f'{label1} - {label2}')

        # Labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_aspect('equal')

        plt.tight_layout()

        logger.info("  Difference map created")

        return fig
