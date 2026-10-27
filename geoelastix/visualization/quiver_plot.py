"""Quiver plot visualization for horizontal displacement."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from pathlib import Path

logger = logging.getLogger("geoelastix.visualization")


class QuiverPlot:
    """
    Create quiver (vector field) plots for horizontal displacement.

    Visualizes X and Y displacement as arrows showing direction and magnitude.
    """

    @staticmethod
    def create_quiver_plot(displacement_x, displacement_y, subsample=50,
                          title="Horizontal Displacement", dpi=300,
                          figsize=(10, 8), colormap='viridis',
                          scale=None, mask=None, pixel_spacing=None,
                          geotransform=None):
        """
        Create quiver plot from displacement fields.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement (East-West)
        displacement_y : numpy.ndarray
            Y displacement (North-South)
        subsample : int, optional
            Subsampling factor for vectors. Default: 50
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI. Default: 300
        figsize : tuple, optional
            Figure size (width, height). Default: (10, 8)
        colormap : str, optional
            Colormap name. Default: 'viridis'
        scale : float, optional
            Arrow scale. If None, auto-computed.
        mask : numpy.ndarray, optional
            Binary mask for valid regions
        pixel_spacing : tuple, optional
            Pixel spacing (dx, dy) in real-world units. Default: None (uses pixels)
        geotransform : tuple, optional
            GDAL geotransform. Default: None

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info(f"Creating quiver plot (subsample={subsample})...")

        # Apply mask if provided
        if mask is not None:
            disp_x = displacement_x.copy()
            disp_y = displacement_y.copy()
            disp_x[mask == 0] = np.nan
            disp_y[mask == 0] = np.nan
        else:
            disp_x = displacement_x
            disp_y = displacement_y

        # Get dimensions
        height, width = displacement_x.shape

        # Determine coordinate system and units
        if pixel_spacing is not None:
            # Use real-world coordinates
            dx, dy = pixel_spacing
            # For DEM convention: origin at top-left, Y increases downward
            x_coords = np.arange(0, width, subsample) * dx
            y_coords = np.arange(0, height, subsample) * dy

            # Units follow the raster CRS (e.g. m for UTM, ft for State Plane)
            unit = 'map units'

            xlabel = f'Easting ({unit})'
            ylabel = f'Northing ({unit})'
        else:
            # Use pixel coordinates
            x_coords = np.arange(0, width, subsample)
            y_coords = np.arange(0, height, subsample)
            unit = 'pixels'
            xlabel = 'X (pixels)'
            ylabel = 'Y (pixels)'

        X, Y = np.meshgrid(x_coords, y_coords)

        # Subsample displacement fields
        U = disp_x[::subsample, ::subsample]
        V = disp_y[::subsample, ::subsample]

        # Compute magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot quiver
        # displacement_x/y are East/North motion. Rows increase southward and
        # the y-axis is inverted below, so North must be drawn as -V.
        quiver = ax.quiver(
            X, Y, U, -V,
            magnitude,
            cmap=colormap,
            scale=scale,
            scale_units='xy' if scale else 'width',
            angles='xy',
            width=0.003
        )

        # Colorbar
        cbar_label = f'Displacement Magnitude ({unit})' if unit != 'pixels' else 'Displacement Magnitude'
        cbar = plt.colorbar(quiver, ax=ax, label=cbar_label)

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set aspect ratio
        ax.set_aspect('equal')

        # Invert y-axis for DEM convention (North up, origin top-left)
        ax.invert_yaxis()

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        logger.info(f"  Quiver plot created with {X.size} vectors")

        return fig

    @staticmethod
    def create_quiver_on_image(displacement_x, displacement_y, background_image=None,
                               subsample=50, title="Horizontal Displacement",
                               dpi=300, figsize=(10, 8), arrow_color='red',
                               arrow_width=0.003, scale=None, mask=None):
        """
        Create quiver plot overlaid on background image.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement
        displacement_y : numpy.ndarray
            Y displacement
        background_image : numpy.ndarray, optional
            Background image to display
        subsample : int, optional
            Subsampling factor
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        arrow_color : str, optional
            Arrow color. Default: 'red'
        arrow_width : float, optional
            Arrow width. Default: 0.003
        scale : float, optional
            Arrow scale
        mask : numpy.ndarray, optional
            Binary mask

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info(f"Creating quiver plot on image (subsample={subsample})...")

        # Apply mask
        if mask is not None:
            disp_x = displacement_x.copy()
            disp_y = displacement_y.copy()
            disp_x[mask == 0] = np.nan
            disp_y[mask == 0] = np.nan
        else:
            disp_x = displacement_x
            disp_y = displacement_y

        # Get dimensions
        height, width = displacement_x.shape

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Display background image if provided
        if background_image is not None:
            ax.imshow(background_image, cmap='gray', alpha=0.7)

        # Create coordinate grids (subsampled)
        y_coords = np.arange(0, height, subsample)
        x_coords = np.arange(0, width, subsample)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Subsample displacement fields
        U = disp_x[::subsample, ::subsample]
        V = disp_y[::subsample, ::subsample]

        # Plot quiver
        # East/North motion; -V because the y-axis (rows) is inverted
        ax.quiver(
            X, Y, U, -V,
            color=arrow_color,
            scale=scale,
            scale_units='xy' if scale else 'width',
            angles='xy',
            width=arrow_width
        )

        # Labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set aspect ratio
        ax.set_aspect('equal')

        # Invert y-axis
        ax.invert_yaxis()

        plt.tight_layout()

        logger.info("  Quiver plot on image created")

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

        logger.info(f"Quiver plot saved to: {output_path}")

    @staticmethod
    def create_magnitude_direction_plot(displacement_x, displacement_y,
                                       title="Displacement Magnitude and Direction",
                                       dpi=300, figsize=(12, 5),
                                       colormap='viridis', mask=None):
        """
        Create side-by-side plots of magnitude and direction.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement
        displacement_y : numpy.ndarray
            Y displacement
        title : str, optional
            Main title
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        colormap : str, optional
            Colormap for magnitude
        mask : numpy.ndarray, optional
            Binary mask

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info("Creating magnitude and direction plot...")

        # Compute magnitude and direction
        magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
        # Angle from East, counterclockwise (displacement is East/North)
        direction = np.arctan2(displacement_y, displacement_x)  # radians

        # Apply mask
        if mask is not None:
            magnitude[mask == 0] = np.nan
            direction[mask == 0] = np.nan

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        # Magnitude plot
        im1 = ax1.imshow(magnitude, cmap=colormap, interpolation='nearest')
        ax1.set_title('Displacement Magnitude', fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Magnitude')

        # Direction plot (in degrees, with custom colormap)
        direction_deg = np.degrees(direction) % 360
        im2 = ax2.imshow(direction_deg, cmap='hsv', interpolation='nearest',
                        vmin=0, vmax=360)
        ax2.set_title('Displacement Direction', fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Direction (degrees)')

        # Main title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        logger.info("  Magnitude and direction plot created")

        return fig

    @staticmethod
    def create_displacement_rose_diagram(displacement_x, displacement_y,
                                        bins=16, title="Displacement Rose Diagram",
                                        dpi=300, figsize=(8, 8), mask=None):
        """
        Create rose diagram showing distribution of displacement directions.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement
        displacement_y : numpy.ndarray
            Y displacement
        bins : int, optional
            Number of directional bins. Default: 16
        title : str, optional
            Plot title
        dpi : int, optional
            Figure DPI
        figsize : tuple, optional
            Figure size
        mask : numpy.ndarray, optional
            Binary mask

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        logger.info("Creating displacement rose diagram...")

        # Apply mask and flatten
        if mask is not None:
            valid_mask = mask > 0
            dx = displacement_x[valid_mask].flatten()
            dy = displacement_y[valid_mask].flatten()
        else:
            dx = displacement_x.flatten()
            dy = displacement_y.flatten()

        # Remove NaN
        valid = ~(np.isnan(dx) | np.isnan(dy))
        dx = dx[valid]
        dy = dy[valid]

        if len(dx) == 0:
            logger.warning("No valid displacement data for rose diagram")
            return None

        # Compute directions (in radians, then degrees)
        # Angle from East, counterclockwise (displacement is East/North)
        directions = np.arctan2(dy, dx)
        directions_deg = (np.degrees(directions) + 360) % 360  # 0-360

        # Compute magnitudes
        magnitudes = np.sqrt(dx**2 + dy**2)

        # Create histogram
        theta_bins = np.linspace(0, 2*np.pi, bins+1)
        hist, bin_edges = np.histogram(directions, bins=theta_bins,
                                       weights=magnitudes)

        # Normalize
        hist = hist / np.max(hist) if np.max(hist) > 0 else hist

        # Create polar plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='polar')

        # Plot bars
        theta = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 2*np.pi / bins

        bars = ax.bar(theta, hist, width=width, bottom=0.0, alpha=0.7)

        # Color bars by direction
        colors = plt.cm.hsv(theta / (2*np.pi))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

        # Configure plot
        ax.set_theta_zero_location('E')  # 0 degrees = East
        ax.set_theta_direction(1)  # Counterclockwise

        # Add cardinal directions
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['E', 'N', 'W', 'S'])

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        logger.info("  Rose diagram created")

        return fig
