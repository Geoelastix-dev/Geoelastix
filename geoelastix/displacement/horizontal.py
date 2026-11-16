"""Horizontal displacement calculation from deformation field."""

import numpy as np
import logging
import itk
from pathlib import Path

logger = logging.getLogger("geoelastix.displacement")


class HorizontalDisplacement:
    """
    Calculate horizontal displacement (X, Y) from elastix deformation field.

    Extracts the deformation field from elastix transformation parameters
    and computes displacement in X (East-West) and Y (North-South) directions.
    """

    @staticmethod
    def compute_from_transform(fixed_image, transform_parameters, spacing=None,
                               geotransform=None):
        """
        Compute horizontal displacement from transformation parameters.

        The elastix transform maps a point x of the fixed image to the
        corresponding point x + u(x) of the moving image, so u points from the
        fixed epoch back to the moving epoch, in the image's (column, row)
        frame. With the fixed image being the NEWER DEM and the moving image
        the OLDER DEM, the ground motion between the two epochs is -u. This
        method returns that motion converted to map coordinates:

        - displacement_x: Easting change (positive = moved East)
        - displacement_y: Northing change (positive = moved North)

        Parameters
        ----------
        fixed_image : itk.Image
            Fixed (reference, newer) image used as template for output
        transform_parameters : itk.ParameterObject
            Elastix transformation parameters
        spacing : tuple, optional
            Pixel size (pixel_width, pixel_height) in map units, used when no
            geotransform is given (assumes a north-up raster).
        geotransform : tuple, optional
            GDAL geotransform of the fixed raster. Takes precedence over
            ``spacing``. If neither is given, the output is in pixels.

        Returns
        -------
        dict
            Dictionary containing:
            - 'displacement_x': numpy array of East displacement (map units)
            - 'displacement_y': numpy array of North displacement (map units)
            - 'spacing': pixel spacing used
        """
        logger.info("Computing horizontal displacement from deformation field...")

        if geotransform is None:
            if spacing is not None:
                geotransform = (0.0, abs(spacing[0]), 0.0, 0.0, 0.0, -abs(spacing[1]))
                logger.info(f"  Using provided spacing: {spacing}")
            else:
                geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
                logger.warning("  No spacing/geotransform given - displacement in pixels")
        spacing = (abs(geotransform[1]), abs(geotransform[5]))

        # Use ITK's transformix_deformation_field (matches reference code)
        logger.info("  Generating deformation field using transformix...")

        try:
            # Generate deformation field using ITK's built-in function
            # This is the correct approach matching the reference ASC2_rectification.py
            deformation_field = itk.transformix_deformation_field(
                fixed_image,
                transform_parameters
            )

            # Convert to numpy array
            deformation_array = itk.array_from_image(deformation_field)

            # ITK convention: [:, :, 0] is along columns, [:, :, 1] is along rows,
            # in the ITK image's physical units. Convert to pixels.
            itk_spacing = np.array(fixed_image.GetSpacing(), dtype=np.float64)
            u_col = deformation_array[:, :, 0] / itk_spacing[0]
            u_row = deformation_array[:, :, 1] / itk_spacing[1]
            del deformation_array

            # Ground motion (moving -> fixed epoch) is the negative of u
            motion_col = -u_col
            motion_row = -u_row

            # Pixel offsets -> map offsets via the geotransform
            gt = geotransform
            displacement_x = (motion_col * gt[1] + motion_row * gt[2]).astype(np.float32)
            displacement_y = (motion_col * gt[4] + motion_row * gt[5]).astype(np.float32)

            logger.info(f"  East displacement range: [{np.min(displacement_x):.3f}, {np.max(displacement_x):.3f}]")
            logger.info(f"  North displacement range: [{np.min(displacement_y):.3f}, {np.max(displacement_y):.3f}]")

            return {
                'displacement_x': displacement_x,
                'displacement_y': displacement_y,
                'spacing': spacing
            }

        except Exception as e:
            logger.error(f"Failed to compute displacement: {e}")
            raise

    @staticmethod
    def apply_mask(displacement_x, displacement_y, mask):
        """
        Apply mask to displacement fields.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement field
        displacement_y : numpy.ndarray
            Y displacement field
        mask : numpy.ndarray
            Binary mask (1=valid, 0=no-data)

        Returns
        -------
        tuple
            (masked_disp_x, masked_disp_y)
        """
        masked_x = displacement_x.copy()
        masked_y = displacement_y.copy()

        masked_x[mask == 0] = np.nan
        masked_y[mask == 0] = np.nan

        return masked_x, masked_y

    @staticmethod
    def get_statistics(displacement_x, displacement_y, mask=None):
        """
        Compute statistics for displacement fields.

        Parameters
        ----------
        displacement_x : numpy.ndarray
            X displacement field
        displacement_y : numpy.ndarray
            Y displacement field
        mask : numpy.ndarray, optional
            Binary mask for valid regions

        Returns
        -------
        dict
            Statistics dictionary
        """
        if mask is not None:
            valid_mask = mask > 0
            dx = displacement_x[valid_mask]
            dy = displacement_y[valid_mask]
        else:
            dx = displacement_x.flatten()
            dy = displacement_y.flatten()

        # Remove NaN values
        dx = dx[~np.isnan(dx)]
        dy = dy[~np.isnan(dy)]

        magnitude = np.sqrt(dx**2 + dy**2)

        stats = {
            'x_mean': float(np.mean(dx)),
            'x_std': float(np.std(dx)),
            'x_min': float(np.min(dx)),
            'x_max': float(np.max(dx)),
            'y_mean': float(np.mean(dy)),
            'y_std': float(np.std(dy)),
            'y_min': float(np.min(dy)),
            'y_max': float(np.max(dy)),
            'magnitude_mean': float(np.mean(magnitude)),
            'magnitude_std': float(np.std(magnitude)),
            'magnitude_min': float(np.min(magnitude)),
            'magnitude_max': float(np.max(magnitude)),
            'n_valid_pixels': len(dx)
        }

        return stats
