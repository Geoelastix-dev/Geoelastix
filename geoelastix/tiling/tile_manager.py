"""Tile manager for processing large raster datasets."""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger("geoelastix.tiling")


@dataclass
class Tile:
    """
    Represents a single tile in a tiled dataset.

    Attributes
    ----------
    index : tuple
        (row, col) index of tile in grid
    row_start : int
        Starting row in full image
    row_end : int
        Ending row in full image (exclusive)
    col_start : int
        Starting column in full image
    col_end : int
        Ending column in full image (exclusive)
    height : int
        Tile height in pixels
    width : int
        Tile width in pixels
    """
    index: Tuple[int, int]
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    height: int
    width: int

    def get_slice(self):
        """Get numpy slice for this tile."""
        return (slice(self.row_start, self.row_end),
                slice(self.col_start, self.col_end))


class TileManager:
    """
    Manage tiling strategy for large raster datasets.

    Automatically divides large images into manageable tiles with overlap
    for seamless processing and reconstruction.
    """

    def __init__(self, image_shape, tile_size=5000, overlap=100):
        """
        Initialize tile manager.

        Parameters
        ----------
        image_shape : tuple
            (height, width) of full image
        tile_size : int, optional
            Target tile size in pixels. Default: 5000
        overlap : int, optional
            Overlap between adjacent tiles in pixels. Default: 100
        """
        self.image_shape = image_shape
        self.height, self.width = image_shape
        self.tile_size = tile_size
        self.overlap = overlap

        logger.info(f"Initializing TileManager for image {self.height}x{self.width}")
        logger.info(f"  Tile size: {tile_size}")
        logger.info(f"  Overlap: {overlap}")

        # Determine if tiling is needed
        self.needs_tiling = self._check_needs_tiling()

        if self.needs_tiling:
            self.tiles = self._create_tiles()
            logger.info(f"  Created {len(self.tiles)} tiles")
        else:
            self.tiles = []
            logger.info("  No tiling needed - image size below threshold")

    def _check_needs_tiling(self):
        """Check if image needs to be tiled."""
        return max(self.height, self.width) > self.tile_size

    def _create_tiles(self):
        """
        Create tile layout with overlap.

        Returns
        -------
        list of Tile
            List of tile objects
        """
        tiles = []

        # Calculate number of tiles in each dimension
        # Use stride = tile_size - overlap for adjacent tiles
        stride = self.tile_size - self.overlap

        # Calculate tile grid dimensions
        n_rows = int(np.ceil((self.height - self.overlap) / stride))
        n_cols = int(np.ceil((self.width - self.overlap) / stride))

        logger.info(f"  Tile grid: {n_rows} x {n_cols}")

        # Create tiles
        for i in range(n_rows):
            for j in range(n_cols):
                # Calculate tile boundaries
                row_start = i * stride
                col_start = j * stride

                # Ensure we don't exceed image boundaries
                row_end = min(row_start + self.tile_size, self.height)
                col_end = min(col_start + self.tile_size, self.width)

                # Adjust start position if we're at the edge
                # to ensure tiles are tile_size (except possibly the last ones)
                if i == n_rows - 1 and row_end - row_start < self.tile_size:
                    row_start = max(0, row_end - self.tile_size)
                if j == n_cols - 1 and col_end - col_start < self.tile_size:
                    col_start = max(0, col_end - self.tile_size)

                tile = Tile(
                    index=(i, j),
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    height=row_end - row_start,
                    width=col_end - col_start
                )

                tiles.append(tile)

        return tiles

    def extract_tile(self, array, tile):
        """
        Extract a tile from a full array.

        Parameters
        ----------
        array : numpy.ndarray
            Full image array
        tile : Tile
            Tile specification

        Returns
        -------
        numpy.ndarray
            Extracted tile array
        """
        return array[tile.get_slice()].copy()

    def extract_tile_pair(self, fixed_array, moving_array, tile):
        """
        Extract corresponding tiles from fixed and moving arrays.

        Parameters
        ----------
        fixed_array : numpy.ndarray
            Fixed image array
        moving_array : numpy.ndarray
            Moving image array
        tile : Tile
            Tile specification

        Returns
        -------
        tuple
            (fixed_tile, moving_tile)
        """
        fixed_tile = self.extract_tile(fixed_array, tile)
        moving_tile = self.extract_tile(moving_array, tile)
        return fixed_tile, moving_tile

    def get_core_slice(self, tile):
        """
        Get the part of a tile to keep when mosaicking results.

        Half of the overlap is trimmed on every side that borders another
        tile, removing the poorly constrained registration edges. The cores
        of all tiles cover the full image without gaps.

        Parameters
        ----------
        tile : Tile
            Tile specification

        Returns
        -------
        tuple of slice
            (row_slice, col_slice) in full-image coordinates
        """
        half = self.overlap // 2

        row_start = tile.row_start + half if tile.row_start > 0 else 0
        row_end = tile.row_end - half if tile.row_end < self.height else self.height
        col_start = tile.col_start + half if tile.col_start > 0 else 0
        col_end = tile.col_end - half if tile.col_end < self.width else self.width

        return (slice(row_start, row_end), slice(col_start, col_end))

    def get_tile_info(self, tile):
        """
        Get information about a tile.

        Parameters
        ----------
        tile : Tile
            Tile object

        Returns
        -------
        dict
            Tile information
        """
        return {
            'index': tile.index,
            'position': (tile.row_start, tile.col_start),
            'size': (tile.height, tile.width),
            'bounds': {
                'row_start': tile.row_start,
                'row_end': tile.row_end,
                'col_start': tile.col_start,
                'col_end': tile.col_end
            }
        }

    def reconstruct_from_tiles(self, tile_arrays, blend_method='average'):
        """
        Reconstruct full array from processed tiles.

        Parameters
        ----------
        tile_arrays : list of tuple
            List of (tile, array) pairs
        blend_method : str, optional
            Method for blending overlapping regions:
            - 'average': Average overlapping values
            - 'feather': Weighted average with feathering
            - 'first': Keep first tile values
            Default: 'average'

        Returns
        -------
        numpy.ndarray
            Reconstructed full array
        """
        logger.info(f"Reconstructing full array from {len(tile_arrays)} tiles...")
        logger.info(f"  Blend method: {blend_method}")

        # Get data type from first tile
        first_tile_array = tile_arrays[0][1]
        dtype = first_tile_array.dtype

        # Initialize output arrays
        reconstructed = np.zeros(self.image_shape, dtype=dtype)

        if blend_method == 'average':
            # Track number of contributions to each pixel for averaging
            weight_sum = np.zeros(self.image_shape, dtype=np.float32)

            for tile, tile_array in tile_arrays:
                # Place tile in full array
                reconstructed[tile.get_slice()] += tile_array
                weight_sum[tile.get_slice()] += 1.0

            # Average overlapping regions
            valid_mask = weight_sum > 0
            reconstructed[valid_mask] = reconstructed[valid_mask] / weight_sum[valid_mask]

        elif blend_method == 'feather':
            # Weighted blending with feathering in overlap regions
            from .edge_handler import EdgeHandler
            edge_handler = EdgeHandler(self.overlap)

            weight_sum = np.zeros(self.image_shape, dtype=np.float32)

            for tile, tile_array in tile_arrays:
                # Create weight mask for this tile
                weight_mask = edge_handler.create_feather_mask(
                    tile.height, tile.width, self.overlap
                )

                # Apply weighted contribution
                tile_slice = tile.get_slice()
                reconstructed[tile_slice] += tile_array * weight_mask
                weight_sum[tile_slice] += weight_mask

            # Normalize by weight sum
            valid_mask = weight_sum > 0
            reconstructed[valid_mask] = reconstructed[valid_mask] / weight_sum[valid_mask]

        elif blend_method == 'first':
            # Simply use first tile values (no blending)
            for tile, tile_array in tile_arrays:
                # Only write where not already filled
                tile_slice = tile.get_slice()
                mask = reconstructed[tile_slice] == 0
                reconstructed[tile_slice][mask] = tile_array[mask]

        else:
            raise ValueError(f"Unknown blend method: {blend_method}")

        logger.info("  Reconstruction complete")

        return reconstructed

    def get_tile_by_index(self, row_idx, col_idx):
        """
        Get tile by grid index.

        Parameters
        ----------
        row_idx : int
            Row index in tile grid
        col_idx : int
            Column index in tile grid

        Returns
        -------
        Tile or None
            Tile object if found, None otherwise
        """
        for tile in self.tiles:
            if tile.index == (row_idx, col_idx):
                return tile
        return None

    def save_tile_layout(self, output_path):
        """
        Save tile layout information to file.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        layout_info = {
            'image_shape': self.image_shape,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'needs_tiling': self.needs_tiling,
            'n_tiles': len(self.tiles),
            'tiles': [
                {
                    'index': tile.index,
                    'row_start': tile.row_start,
                    'row_end': tile.row_end,
                    'col_start': tile.col_start,
                    'col_end': tile.col_end,
                    'height': tile.height,
                    'width': tile.width
                }
                for tile in self.tiles
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(layout_info, f, indent=2)

        logger.info(f"Saved tile layout to: {output_path}")

    @staticmethod
    def load_tile_layout(input_path):
        """
        Load tile layout from file.

        Parameters
        ----------
        input_path : str or Path
            Input file path

        Returns
        -------
        TileManager
            Reconstructed TileManager instance
        """
        import json

        with open(input_path, 'r') as f:
            layout_info = json.load(f)

        tile_manager = TileManager(
            image_shape=tuple(layout_info['image_shape']),
            tile_size=layout_info['tile_size'],
            overlap=layout_info['overlap']
        )

        logger.info(f"Loaded tile layout from: {input_path}")

        return tile_manager
