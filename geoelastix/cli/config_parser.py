"""Configuration file parser for GeoElastix."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("geoelastix.cli")


class ConfigParser:
    """
    Parse and validate YAML configuration files.

    Provides default values and validation for all configuration options.
    """

    DEFAULT_CONFIG = {
        'job': {
            'id': 'geoelastix_job',
            'description': None,
            'output_dir': './output'
        },
        'input': {
            'fixed_image': None,
            'moving_image': None,
        },
        'registration': {
            'method': 'NA',
            'metric': 'NCC',
            'parameter_file': None,
            'grid_spacing': None,
            'grid_schedule': None,
            'bending_weight': 0.0,
            'mask': {
                'terrain': False,
                'min_slope': 5.0,
                'window': 30.0,
                'min_width': 0.0,
                'min_area': 0.0,
                'file': None,
                'mask_outputs': True,
            },
            'log_to_console': False,
        },
        'processing': {
            'tile_threshold': 5000,
            'tile_overlap': 100,
            'min_valid_fraction': 0.05,
            'num_threads': 4,
        },
        'output': {
            'formats': ['geotiff', 'asc'],
            'generate_magnitude': True,
            'compression': 'LZW',
        },
        'visualization': {
            'generate_plots': True,
            'plot_types': ['quiver', 'contour_x', 'contour_y', 'contour_z', 'contour_magnitude'],
            'dpi': 300,
            'colormap': 'viridis',
            'quiver_subsample': 50,
        },
        'reporting': {
            'generate_pdf': True,
            'include_metrics': True,
            'include_parameters': True,
            'include_visualizations': True,
        },
        'quality': {
            'rmse_warning_threshold': 10.0,
            'min_overlap_percent': 70.0,
            'max_nodata_percent': 30.0,
            'min_dice': 0.7,
            'min_correlation': 0.5,
            'min_mi': 0.1,
        },
        'logging': {
            'level': 'INFO',
            'save_to_file': True,
            'log_to_console': True,
        }
    }

    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file

        Returns
        -------
        dict
            Configuration dictionary

        Raises
        ------
        FileNotFoundError
            If config file not found
        yaml.YAMLError
            If config file is invalid YAML
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info("Configuration loaded successfully")

        return config

    @staticmethod
    def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user config with defaults.

        Parameters
        ----------
        config : dict
            User configuration

        Returns
        -------
        dict
            Merged configuration with defaults
        """
        import copy

        merged = copy.deepcopy(ConfigParser.DEFAULT_CONFIG)

        def update_nested(target, source):
            """Recursively update nested dictionaries."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_nested(target[key], value)
                else:
                    target[key] = value

        update_nested(merged, config)

        return merged

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration.

        Parameters
        ----------
        config : dict
            Configuration to validate

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        logger.info("Validating configuration...")

        errors = []

        # Check required fields
        if 'input' not in config:
            errors.append("Missing 'input' section")
        else:
            if 'fixed_image' not in config['input'] or config['input']['fixed_image'] is None:
                errors.append("Missing required field: input.fixed_image")
            if 'moving_image' not in config['input'] or config['input']['moving_image'] is None:
                errors.append("Missing required field: input.moving_image")

            # Check if files exist
            if config['input'].get('fixed_image'):
                fixed_path = Path(config['input']['fixed_image'])
                if not fixed_path.exists():
                    errors.append(f"Fixed image not found: {fixed_path}")

            if config['input'].get('moving_image'):
                moving_path = Path(config['input']['moving_image'])
                if not moving_path.exists():
                    errors.append(f"Moving image not found: {moving_path}")

        # Validate registration method
        if 'registration' in config:
            method = config['registration'].get('method', 'NA').upper()
            valid_methods = ['NA', 'AF', 'RG', 'TS']
            if method not in valid_methods:
                errors.append(f"Invalid registration method: {method}. Must be one of {valid_methods}")

            # Check custom parameter file if provided
            param_file = config['registration'].get('parameter_file')
            if param_file is not None:
                param_path = Path(param_file)
                if not param_path.exists():
                    errors.append(f"Parameter file not found: {param_path}")

            grid_spacing = config['registration'].get('grid_spacing')
            if grid_spacing is not None and not grid_spacing > 0:
                errors.append("grid_spacing must be positive (map units)")

            grid_schedule = config['registration'].get('grid_schedule')
            if grid_schedule is not None and (
                    not isinstance(grid_schedule, (list, tuple))
                    or not all(s > 0 for s in grid_schedule)):
                errors.append("grid_schedule must be a list of positive factors, e.g. [8, 4, 2, 1]")

            bending_weight = config['registration'].get('bending_weight') or 0.0
            if bending_weight < 0:
                errors.append("bending_weight must be non-negative")

            mask_cfg = config['registration'].get('mask') or {}
            if mask_cfg.get('window', 30.0) <= 0:
                errors.append("mask.window must be positive (map units)")
            if mask_cfg.get('min_width', 0.0) < 0 or mask_cfg.get('min_area', 0.0) < 0:
                errors.append("mask.min_width and mask.min_area must be non-negative")
            if mask_cfg.get('file') and not Path(mask_cfg['file']).exists():
                errors.append(f"Registration mask file not found: {mask_cfg['file']}")

        # Validate output formats
        if 'output' in config:
            formats = config['output'].get('formats', [])
            valid_formats = ['geotiff', 'asc']
            for fmt in formats:
                if fmt.lower() not in valid_formats:
                    errors.append(f"Invalid output format: {fmt}. Must be one of {valid_formats}")

        # Validate plot types
        if 'visualization' in config:
            plot_types = config['visualization'].get('plot_types', [])
            valid_types = ['quiver', 'contour_x', 'contour_y', 'contour_z', 'contour_magnitude']
            for ptype in plot_types:
                if ptype not in valid_types:
                    errors.append(f"Invalid plot type: {ptype}. Must be one of {valid_types}")

        # Validate numeric thresholds
        if 'processing' in config:
            tile_threshold = config['processing'].get('tile_threshold', 5000)
            if tile_threshold <= 0:
                errors.append("tile_threshold must be positive")

            tile_overlap = config['processing'].get('tile_overlap', 100)
            if tile_overlap < 0:
                errors.append("tile_overlap must be non-negative")
            elif tile_overlap >= tile_threshold:
                errors.append("tile_overlap must be smaller than tile_threshold")

        # Validate logging level
        if 'logging' in config:
            log_level = config['logging'].get('level', 'INFO').upper()
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level not in valid_levels:
                errors.append(f"Invalid log level: {log_level}. Must be one of {valid_levels}")

            # Validate log_to_console is boolean
            log_to_console = config['logging'].get('log_to_console', False)
            if not isinstance(log_to_console, bool):
                errors.append(f"log_to_console must be a boolean (true/false), got: {log_to_console}")

        # Report errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✓ Configuration validated successfully")
        return True

    @staticmethod
    def load_and_validate(config_path: Path) -> Dict[str, Any]:
        """
        Load, merge with defaults, and validate configuration.

        Parameters
        ----------
        config_path : Path
            Path to configuration file

        Returns
        -------
        dict
            Validated configuration
        """
        # Load config
        config = ConfigParser.load_config(config_path)

        # Merge with defaults
        config = ConfigParser.merge_with_defaults(config)

        # Validate
        ConfigParser.validate_config(config)

        return config

    @staticmethod
    def create_default_config(output_path: Path):
        """
        Create a default configuration file template.

        Parameters
        ----------
        output_path : Path
            Output path for config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create template with comments
        template = """# GeoElastix Configuration File
# Template for landslide monitoring with multi-temporal DEMs

job:
  id: "my_landslide_analysis"          # Job identifier
  description: "2021_vs_2022"          # Optional description
  output_dir: "./output"                # Output directory

input:
  fixed_image: "./data/dem_new.tif"    # Reference DEM (newer)
  moving_image: "./data/dem_old.tif"   # Moving DEM (older)

registration:
  method: "NA"                          # NA, AF, RG, or TS
  metric: "NCC"                         # NCC or MI
  parameter_file: null                  # Optional custom parameter file
  grid_spacing: null                    # B-spline control point spacing in map units
                                        # (null = value in parameter file, in pixels)
  grid_schedule: null                   # Grid spacing factor per resolution, e.g. [8, 4, 2, 1]
  bending_weight: 0.0                   # Bending-energy penalty weight (0 = none; larger = smoother)
  mask:                                 # Pixels the similarity metric is sampled from
    terrain: false                      # Exclude flat/featureless ground (water, floodplain)
    min_slope: 5.0                      # Minimum window-mean slope (degrees)
    window: 30.0                        # Slope averaging window (map units)
    min_width: 0.0                      # Drop parts narrower than this, e.g. embankments (map units)
    min_area: 0.0                       # Drop isolated pieces smaller than this (map units^2)
    file: null                          # Optional raster mask on any grid (nonzero = use)
    mask_outputs: true                  # Set displacements outside the mask to no-data
  log_to_console: false                 # Show elastix registration output (default: false)

processing:
  tile_threshold: 5000                  # Tile size; tiling enabled if dimension > this
  tile_overlap: 100                     # Pixel overlap between tiles (>= 2x grid spacing)
  min_valid_fraction: 0.05              # Skip tiles with less valid data than this
  num_threads: 4                        # Number of parallel threads

output:
  formats: ["geotiff", "asc"]          # Output formats
  generate_magnitude: true              # Generate horizontal magnitude
  compression: "LZW"                    # GeoTIFF compression

visualization:
  generate_plots: true                  # Generate visualization plots
  plot_types:                           # Types of plots to generate
    - "quiver"                          # Vector field
    - "contour_x"                       # X displacement contour
    - "contour_y"                       # Y displacement contour
    - "contour_z"                       # Z displacement contour
    - "contour_magnitude"               # Magnitude contour
  dpi: 300                              # Plot resolution
  colormap: "viridis"                   # Matplotlib colormap
  quiver_subsample: 50                  # Subsample factor for quiver

reporting:
  generate_pdf: true                    # Generate PDF report
  include_metrics: true                 # Include quality metrics
  include_parameters: true              # Include registration parameters
  include_visualizations: true          # Include plots in report

quality:
  rmse_warning_threshold: 10.0          # Warning if RMSE > this (map units)
  min_overlap_percent: 70.0             # Warning if overlap < this
  max_nodata_percent: 30.0              # Warning if no-data > this
  min_dice: 0.7                         # Warning if Dice < this
  min_correlation: 0.5                  # Warning if correlation < this
  min_mi: 0.1                           # Warning if MI < this

logging:
  level: "INFO"                         # DEBUG, INFO, WARNING, ERROR
  save_to_file: true                    # Save log to file
  log_to_console: true                  # Output logs to console (default: true)
"""

        with open(output_path, 'w') as f:
            f.write(template)

        logger.info(f"Default configuration template saved to: {output_path}")

    @staticmethod
    def print_config(config: Dict[str, Any]):
        """
        Print configuration in readable format.

        Parameters
        ----------
        config : dict
            Configuration to print
        """
        import json

        logger.info("Configuration:")
        logger.info(json.dumps(config, indent=2, default=str))
