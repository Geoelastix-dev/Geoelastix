"""GDAL-compliant no-data value handling."""

import numpy as np
import logging

logger = logging.getLogger("geoelastix.io")


class NoDataHandler:
    """
    Handle no-data values in raster datasets following GDAL/ArcGIS standards.

    This class provides methods to detect, create masks, and handle no-data values
    consistently across the registration pipeline.
    """

    @staticmethod
    def get_nodata_value(dataset):
        """
        Get no-data value from GDAL dataset.

        Parameters
        ----------
        dataset : gdal.Dataset
            GDAL dataset

        Returns
        -------
        float or None
            No-data value, or None if not defined
        """
        band = dataset.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        return nodata

    @staticmethod
    def create_mask(array, nodata_value=None, tolerance=1e-6):
        """
        Create binary mask for valid data.

        Parameters
        ----------
        array : numpy.ndarray
            Input raster array
        nodata_value : float, optional
            No-data value. If None, attempts to auto-detect.
        tolerance : float, optional
            Tolerance for floating-point comparison. Default: 1e-6

        Returns
        -------
        numpy.ndarray
            Binary mask (1 = valid data, 0 = no-data) with dtype uint8
        """
        mask = np.ones_like(array, dtype=np.uint8)

        # Handle NaN values
        if np.issubdtype(array.dtype, np.floating):
            nan_mask = np.isnan(array)
            mask[nan_mask] = 0

        # Handle explicit no-data value
        if nodata_value is not None:
            if np.isnan(nodata_value):
                # Already handled above
                pass
            elif np.issubdtype(array.dtype, np.floating):
                # Compare in the array's own precision, with a tolerance relative
                # to the no-data magnitude (e.g. -3.4028235e+38 for Float32)
                nodata_cast = np.asarray(nodata_value).astype(array.dtype)
                tol = tolerance * max(1.0, abs(float(nodata_cast)))
                nodata_mask = np.abs(array - nodata_cast) <= tol
                mask[nodata_mask] = 0
            else:
                # Integer comparison
                nodata_mask = (array == nodata_value)
                mask[nodata_mask] = 0

        return mask

    @staticmethod
    def fill_nodata(array, mask):
        """
        Replace no-data pixels with the value of the nearest valid pixel.

        Elastix must never see no-data values (e.g. -3.4e38): the image pyramid
        smooths them into the valid region and corrupts the registration. The
        masks still exclude these pixels from the metric; the fill only keeps
        smoothing and interpolation near the edges well behaved.

        Parameters
        ----------
        array : numpy.ndarray
            Input raster array
        mask : numpy.ndarray
            Binary mask (1 = valid data, 0 = no-data)

        Returns
        -------
        numpy.ndarray
            Float32 copy of the array with no-data pixels filled
        """
        from scipy import ndimage

        filled = array.astype(np.float32, copy=True)
        invalid = mask == 0

        if not invalid.any():
            return filled
        if invalid.all():
            filled[:] = 0.0
            return filled

        indices = ndimage.distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        filled[invalid] = filled[indices[0][invalid], indices[1][invalid]]
        return filled

    @staticmethod
    def auto_detect_nodata(array, check_edges=True):
        """
        Attempt to automatically detect no-data value.

        This looks for common no-data values and patterns at array edges.

        Parameters
        ----------
        array : numpy.ndarray
            Input raster array
        check_edges : bool, optional
            Check edge values for no-data. Default: True

        Returns
        -------
        float or None
            Detected no-data value, or None if not detected
        """
        # Common no-data values
        common_nodata = [-9999, -32768, -3.4028235e+38, 0]

        # Check for NaN
        if np.any(np.isnan(array)):
            return np.nan

        # Check common values
        for nodata in common_nodata:
            count = np.sum(np.abs(array - nodata) < 1e-6)
            if count > 0:
                # Found potential no-data value
                # Verify it's reasonable (appears at edges or is minimum)
                if check_edges:
                    edge_values = np.concatenate([
                        array[0, :], array[-1, :],
                        array[:, 0], array[:, -1]
                    ])
                    edge_min = np.min(edge_values)
                    if np.abs(edge_min - nodata) < 1e-6:
                        logger.info(f"Auto-detected no-data value: {nodata}")
                        return nodata

        # Check if minimum value might be no-data
        min_val = np.min(array)
        if check_edges:
            edge_values = np.concatenate([
                array[0, :], array[-1, :],
                array[:, 0], array[:, -1]
            ])
            edge_min = np.min(edge_values)
            if edge_min == min_val:
                # Minimum occurs at edge - likely no-data
                logger.warning(
                    f"Assuming edge minimum value as no-data: {min_val}"
                )
                return min_val

        return None

    @staticmethod
    def apply_mask(array, mask, nodata_value=-9999):
        """
        Apply mask to array, setting no-data regions.

        Parameters
        ----------
        array : numpy.ndarray
            Input array
        mask : numpy.ndarray
            Binary mask (1 = valid, 0 = no-data)
        nodata_value : float, optional
            Value to set for no-data regions. Default: -9999

        Returns
        -------
        numpy.ndarray
            Masked array
        """
        result = array.copy()
        result[mask == 0] = nodata_value
        return result

    @staticmethod
    def combine_masks(*masks):
        """
        Combine multiple masks using logical AND.

        Parameters
        ----------
        *masks : numpy.ndarray
            One or more binary masks

        Returns
        -------
        numpy.ndarray
            Combined mask (valid only where all inputs are valid)
        """
        if len(masks) == 0:
            raise ValueError("At least one mask must be provided")

        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.logical_and(combined, mask).astype(np.uint8)

        return combined

    @staticmethod
    def get_valid_data_percentage(mask):
        """
        Calculate percentage of valid data in mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask

        Returns
        -------
        float
            Percentage of valid data (0-100)
        """
        total_pixels = mask.size
        valid_pixels = np.sum(mask)
        percentage = (valid_pixels / total_pixels) * 100
        return percentage

    @staticmethod
    def replace_nodata(array, nodata_value, replacement=0):
        """
        Replace no-data values with a specified value.

        Parameters
        ----------
        array : numpy.ndarray
            Input array
        nodata_value : float
            No-data value to replace
        replacement : float, optional
            Replacement value. Default: 0

        Returns
        -------
        numpy.ndarray
            Array with no-data replaced
        """
        result = array.copy()

        if np.isnan(nodata_value):
            result[np.isnan(result)] = replacement
        elif np.issubdtype(array.dtype, np.floating):
            result[np.abs(result - nodata_value) < 1e-6] = replacement
        else:
            result[result == nodata_value] = replacement

        return result
