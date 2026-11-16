"""Main command-line interface for GeoElastix."""

import sys
import argparse
import logging
from pathlib import Path

from ..utils import setup_logging
from ..__version__ import __version__
from .config_parser import ConfigParser
from .workflow import WorkflowOrchestrator
from ..registration import ParameterManager


def create_parser():
    """
    Create argument parser for GeoElastix CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='geoelastix',
        description='GeoElastix: Geospatial Image Registration for Landslide Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register using configuration file
  geoelastix register --config config.yaml

  # Quick registration with minimal options
  geoelastix register --fixed dem_new.tif --moving dem_old.tif --job-id analysis1

  # List available registration methods
  geoelastix list-methods

  # Show parameter file details
  geoelastix show-params NA

  # Generate default configuration template
  geoelastix create-config --output my_config.yaml

For more information, visit: https://github.com/yzh211/geoelastix
        """
    )

    parser.add_argument('--version', action='version', version=f'GeoElastix {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register command
    register_parser = subparsers.add_parser(
        'register',
        help='Perform image registration and displacement analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    register_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )

    register_parser.add_argument(
        '--fixed', '-f',
        type=str,
        help='Path to fixed (reference) image'
    )

    register_parser.add_argument(
        '--moving', '-m',
        type=str,
        help='Path to moving image'
    )

    register_parser.add_argument(
        '--job-id',
        type=str,
        default='geoelastix_job',
        help='Job identifier'
    )

    register_parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory'
    )

    register_parser.add_argument(
        '--method',
        type=str,
        choices=['NA', 'AF', 'RG', 'TS'],
        default='NA',
        help='Registration method (NA=Non-Affine, AF=Affine, RG=Rigid, TS=Translation)'
    )

    register_parser.add_argument(
        '--grid-spacing',
        type=float,
        default=None,
        help='B-spline control point spacing in map units (default: value in parameter file)'
    )

    register_parser.add_argument(
        '--grid-schedule',
        type=float,
        nargs='+',
        default=None,
        help='Grid spacing factor per resolution, coarsest first (e.g. 8 4 2 1)'
    )

    register_parser.add_argument(
        '--bending-weight',
        type=float,
        default=0.0,
        help='Bending-energy penalty weight (default: 0 = none; larger = smoother)'
    )

    register_parser.add_argument(
        '--terrain-mask',
        action='store_true',
        help='Sample the similarity metric only on textured terrain (excludes water, flats)'
    )

    register_parser.add_argument(
        '--min-slope',
        type=float,
        default=5.0,
        help='Terrain mask: minimum window-mean slope in degrees (default: 5)'
    )

    register_parser.add_argument(
        '--min-width',
        type=float,
        default=0.0,
        help='Terrain mask: drop parts narrower than this, in map units (default: 0)'
    )

    register_parser.add_argument(
        '--min-area',
        type=float,
        default=0.0,
        help='Terrain mask: drop isolated pieces smaller than this, in map units^2 (default: 0)'
    )

    register_parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='Registration mask raster (nonzero = use)'
    )

    register_parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    register_parser.add_argument(
        '--log-to-console',
        action='store_true',
        default=False,
        help='Output logs to console (default: False)'
    )

    # List methods command
    list_parser = subparsers.add_parser(
        'list-methods',
        help='List available registration methods'
    )

    # Show parameters command
    show_params_parser = subparsers.add_parser(
        'show-params',
        help='Show details of a parameter file'
    )

    show_params_parser.add_argument(
        'method',
        type=str,
        choices=['NA', 'AF', 'RG', 'TS'],
        help='Registration method'
    )

    # Create config command
    create_config_parser = subparsers.add_parser(
        'create-config',
        help='Create a default configuration template'
    )

    create_config_parser.add_argument(
        '--output', '-o',
        type=str,
        default='config_template.yaml',
        help='Output path for configuration template'
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        'validate-config',
        help='Validate a configuration file'
    )

    validate_parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file'
    )

    return parser


def cmd_register(args):
    """
    Execute register command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments
    """
    logger = setup_logging(log_level=args.log_level, log_to_console=args.log_to_console)

    logger.info("="*70)
    logger.info(f"GEOELASTIX v{__version__}")
    logger.info("Geospatial Image Registration for Landslide Monitoring")
    logger.info("="*70)

    try:
        # Load configuration
        if args.config:
            # Load from config file
            logger.info(f"\nLoading configuration from: {args.config}")
            config = ConfigParser.load_and_validate(Path(args.config))
        else:
            # Build config from command-line arguments
            if not args.fixed or not args.moving:
                logger.error("Error: --fixed and --moving are required when not using --config")
                sys.exit(1)

            logger.info("\nBuilding configuration from command-line arguments")
            config = ConfigParser.merge_with_defaults({
                'job': {
                    'id': args.job_id,
                    'output_dir': args.output
                },
                'input': {
                    'fixed_image': args.fixed,
                    'moving_image': args.moving
                },
                'registration': {
                    'method': args.method,
                    'grid_spacing': args.grid_spacing,
                    'grid_schedule': args.grid_schedule,
                    'bending_weight': args.bending_weight,
                    'mask': {
                        'terrain': args.terrain_mask,
                        'min_slope': args.min_slope,
                        'min_width': args.min_width,
                        'min_area': args.min_area,
                        'file': args.mask
                    }
                },
                'logging': {
                    'level': args.log_level,
                    'log_to_console': args.log_to_console
                }
            })

            # Validate
            ConfigParser.validate_config(config)

        # Run workflow
        logger.info("\nStarting workflow...")
        orchestrator = WorkflowOrchestrator(config)
        results = orchestrator.run()

        logger.info("\n✓ GeoElastix completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"\n✗ Registration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def cmd_list_methods(args):
    """
    Execute list-methods command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments
    """
    methods = ParameterManager.list_available_methods()

    print("\nAvailable Registration Methods:")
    print("="*70)

    for code, description in methods.items():
        print(f"\n  {code}: {description}")

    print("\n" + "="*70)
    print("\nUse 'geoelastix show-params <method>' to see parameter details")
    return 0


def cmd_show_params(args):
    """
    Execute show-params command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments
    """
    try:
        param_file = ParameterManager.get_default_parameter_file(args.method)

        print(f"\nParameter file for {args.method}:")
        print("="*70)
        print(f"File: {param_file}")
        print("="*70)
        print()

        # Read and display file contents
        with open(param_file, 'r') as f:
            print(f.read())

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_create_config(args):
    """
    Execute create-config command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments
    """
    try:
        output_path = Path(args.output)
        ConfigParser.create_default_config(output_path)
        print(f"\n✓ Configuration template created: {output_path}")
        print("\nEdit this file with your specific settings, then run:")
        print(f"  geoelastix register --config {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_validate_config(args):
    """
    Execute validate-config command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments
    """
    logger = setup_logging(log_level='INFO')

    try:
        config_path = Path(args.config)
        config = ConfigParser.load_and_validate(config_path)

        print(f"\n✓ Configuration is valid: {config_path}")
        print("\nConfiguration summary:")
        print("  Fixed image:", config['input']['fixed_image'])
        print("  Moving image:", config['input']['moving_image'])
        print("  Method:", config['registration']['method'])
        print("  Output dir:", config['job']['output_dir'])

        return 0

    except Exception as e:
        print(f"\n✗ Configuration validation failed:")
        print(f"  {e}")
        return 1


def main():
    """
    Main entry point for GeoElastix CLI.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = create_parser()

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Execute command
    if args.command == 'register':
        return cmd_register(args)
    elif args.command == 'list-methods':
        return cmd_list_methods(args)
    elif args.command == 'show-params':
        return cmd_show_params(args)
    elif args.command == 'create-config':
        return cmd_create_config(args)
    elif args.command == 'validate-config':
        return cmd_validate_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
