"""Registration masks: restrict the similarity metric to textured terrain."""

import logging
import numpy as np
from scipy import ndimage

logger = logging.getLogger("geoelastix.registration")


class RegistrationMask:
    """
    Build masks of the pixels elastix samples the similarity metric from.

    NCC is computed over all sampled pixels at once, so featureless or changed
    ground (water, floodplain, parking lots, construction) degrades the fit
    everywhere, not only where it occurs. Excluding it keeps the metric on
    terrain whose shape constrains the displacement.
    """

    @staticmethod
    def terrain_mask(dem, valid, pixel_size, min_slope=5.0, window=30.0,
                     min_width=0.0, min_area=0.0):
        """
        Mask of textured terrain: mean slope in a moving window >= ``min_slope``.

        Averaging over a window keeps gentle parts of hillslopes (e.g. landslide
        toes) that are surrounded by steeper ground, while large flat areas
        (water, floodplains, built-up flats) are excluded. Narrow strips (road
        embankments, levees, river banks) and small isolated patches pass the
        slope test but carry little texture and are often man-made change;
        ``min_width`` and ``min_area`` remove them.

        Parameters
        ----------
        dem : numpy.ndarray
            Elevation array
        valid : numpy.ndarray
            Valid data mask (1 = valid)
        pixel_size : float
            Pixel size in the same units as the elevations
        min_slope : float, optional
            Minimum window-mean slope in degrees. Default: 5.0
        window : float, optional
            Window size in map units. Default: 30.0
        min_width : float, optional
            Remove parts narrower than this (morphological opening with a disk
            of diameter ``min_width``), in map units. The slope window widens
            features, so a strip of true width w is about w + ``window`` wide
            in the mask. Default: 0 (keep all)
        min_area : float, optional
            Remove connected pieces smaller than this, in map units squared.
            Default: 0 (keep all)

        Returns
        -------
        numpy.ndarray
            uint8 mask (1 = use for registration)
        """
        valid = valid > 0
        z = np.where(valid, dem, np.nan).astype(np.float64)
        if not valid.all():
            idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                                 return_indices=True)
            z = z[tuple(idx)]

        gy, gx = np.gradient(z, pixel_size)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))

        size = max(1, int(round(window / pixel_size)))
        mean_slope = ndimage.uniform_filter(slope, size=size, mode='nearest')

        mask = (mean_slope >= min_slope) & valid

        if min_width > 0:
            # Opening with a disk, via distance transforms (exact, fast for large radii)
            radius = 0.5 * min_width / pixel_size
            core = ndimage.distance_transform_edt(mask) > radius
            mask = mask & (ndimage.distance_transform_edt(~core) <= radius)

        if min_area > 0:
            labels, n = ndimage.label(mask)
            if n:
                sizes = np.bincount(labels.ravel())[1:] * pixel_size ** 2
                mask = np.concatenate(([False], sizes >= min_area))[labels]

        mask = mask.astype(np.uint8)
        logger.info(f"  Terrain mask (mean slope >= {min_slope} deg over {window} map units, "
                    f"width >= {min_width}, area >= {min_area}): "
                    f"{100 * mask.sum() / max(valid.sum(), 1):.1f}% of valid pixels kept")
        return mask
