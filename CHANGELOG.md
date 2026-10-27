# Changelog

All notable changes to GeoElastix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Horizontal displacement units and sign**: `displacement_x`/`displacement_y` were
  written in pixels and pointed from the fixed (newer) to the moving (older) epoch,
  with Y along image rows (southward). They are now East/North ground motion from
  the moving to the fixed epoch, in map units (requires fixed = newer DEM).
  Quiver, direction and rose plots updated accordingly.
- **No-data handling**: no-data values (e.g. -3.4e38) were passed to elastix and
  smeared into valid data by the image pyramid. They are now filled before
  registration; outputs are masked where the fixed DEM or the warped moving DEM is
  invalid. No-data detection compares in the raster's own precision.
- **Grid alignment**: a moving raster on a different grid (extent, pixel size, CRS)
  is warped onto the fixed grid instead of being silently misaligned.
- NaN pixels are written as the no-data value in GeoTIFF/ASC outputs.
- Warping onto the fixed grid never reads overview pyramids (`.ovr`), which can be
  misaligned with the full-resolution data (seen with ArcGIS overviews: a spurious
  metre-scale shift).

### Added

- Tiled processing in the workflow (`processing.tile_threshold`, `tile_overlap`,
  `min_valid_fraction`); overlaps are trimmed when mosaicking.
- `registration.grid_spacing` / `--grid-spacing`: B-spline control point spacing
  in map units.
- Registration mask (`registration.mask`, `--terrain-mask`, `--min-slope`, `--mask`):
  sample the similarity metric only on textured terrain (window-mean slope above a
  threshold) and/or a user mask raster. NCC is a global metric, so water,
  floodplains and built-up flats otherwise degrade the fit everywhere.
  Displacements outside the mask are set to no-data by default
  (`mask.mask_outputs`); the mask is written to `registered/registration_mask.tif`.
- `registration.bending_weight` / `--bending-weight`: bending-energy penalty
  (TransformBendingEnergyPenalty) for smoother deformation fields.
- `registration.grid_schedule` / `--grid-schedule`: B-spline GridSpacingSchedule.
- Regression tests for sign/units, no-data, grid warping, tiling, the terrain
  mask and the regularisation options.

## [0.1.0] - 2024-11-20

### Added - Phase 5: Documentation & Packaging

- **Comprehensive Documentation Suite**
  - 7 markdown files with 20,000+ words of documentation
  - Installation Guide: Platform-specific setup instructions
  - Quick Start Guide: Get running in 5 minutes
  - User Guide: Complete usage documentation with examples
  - API Reference: Full Python library documentation with code examples
  - Methodology: Technical details of algorithms and mathematical foundations
  - Troubleshooting: Solutions to common problems
  - Contributing Guidelines: Development process and standards

- **Professional README**
  - Badges for license, Python version, status
  - Comprehensive feature overview
  - Quick start examples (CLI, config, Python API)
  - Registration methods comparison table
  - Output structure documentation
  - System requirements and performance benchmarks
  - Use cases and workflow description
  - FAQ section with common questions
  - Complete links and contact information

- **CI/CD Infrastructure**
  - GitHub Actions workflows:
    - Test workflow: Python 3.8-3.11 on Linux/Windows/macOS
    - Lint workflow: flake8, black, isort, mypy
    - Documentation validation workflow
  - Automated testing matrix
  - Code coverage reporting with Codecov
  - Markdown link checking
  - Example config validation

- **Example Workflows**
  - batch_processing.py: Time series analysis script
  - python_api_example.py: Complete API usage example
  - Enhanced examples README with multiple use cases
  - Integration examples for QGIS, GPS validation
  - Method comparison workflows

- **Contributing Infrastructure**
  - CONTRIBUTING.md with development guidelines
  - Code of conduct
  - Development setup instructions
  - Coding standards (PEP 8, docstrings, type hints)
  - Testing requirements and coverage goals
  - Pull request process
  - Issue templates
  - Commit message conventions
  - Release process documentation

- **Project Metadata**
  - Updated .gitignore for comprehensive file exclusions
  - GitHub workflow configurations
  - Markdown link check configuration

### Added - Phase 4: CLI & Integration

- **Configuration Parser**
  - `ConfigParser`: YAML configuration file parsing and validation
    - Default configuration with all options
    - Merge user config with defaults
    - Comprehensive validation (required fields, file existence, valid options)
    - Template generation for new users
    - Pretty-print and JSON export

- **Workflow Orchestrator**
  - `WorkflowOrchestrator`: End-to-end workflow execution
    - 8-step automated workflow from input to output
    - Progress logging at each step
    - Automatic output directory structure
    - Integration of all modules (I/O, registration, displacement, validation, visualization)
    - Error handling and recovery
    - Timing and performance tracking
    - Support for both tiled and non-tiled processing

- **Command-Line Interface**
  - Main CLI with argparse framework
  - Multiple commands:
    - `register`: Perform registration and displacement analysis
    - `list-methods`: Show available registration methods
    - `show-params`: Display parameter file contents
    - `create-config`: Generate configuration template
    - `validate-config`: Validate configuration file
  - Flexible input methods:
    - Configuration file (YAML)
    - Command-line arguments
    - Mix of both (CLI overrides config)
  - Rich help messages and examples

- **Entry Point**
  - Console script: `geoelastix` command
  - Version display: `geoelastix --version`
  - Comprehensive help: `geoelastix --help`

### Added - Phase 3: Validation & Visualization

- **Quality Metrics Module**
  - `QualityMetrics`: Comprehensive registration quality assessment
    - RMSE (Root Mean Square Error) calculation
    - Mutual Information (MI) and Normalized MI
    - Dice coefficient for mask overlap
    - Coverage statistics (overlap percentages)
    - Pearson correlation coefficient
    - All metrics support masking for valid regions only

- **Quality Checker**
  - `QualityChecker`: Automated quality assessment with thresholds
    - Configurable warning and error thresholds
    - RMSE, overlap, no-data, Dice, correlation, and MI checks
    - Displacement statistics validation
    - Quality report generation (text format)
    - JSON export for machine-readable metrics

- **Visualization Modules**
  - `QuiverPlot`: Vector field visualization for horizontal displacement
    - Standard quiver plots with magnitude coloring
    - Quiver overlay on background images
    - Magnitude and direction side-by-side plots
    - Rose diagrams for displacement direction distribution
    - Configurable subsampling, colormaps, and scaling

  - `ContourPlot`: Contour map visualization
    - Filled contour plots for displacement fields
    - Multi-panel layout (X, Y, Z, magnitude)
    - Hillshade overlay for 3D effect
    - Difference maps between datasets
    - Symmetric colormaps centered at zero
    - Customizable levels, colors, and labels

- **Report Generator**
  - `ReportGenerator`: Automated report creation
    - Text-based comprehensive reports
    - JSON metrics export
    - Summary figures with multiple plots
    - Job information and metadata tracking
    - Quality assessment integration
    - Displacement statistics summaries

### Added - Phase 2: Displacement & Tiling

- **Displacement Calculation Modules**
  - `HorizontalDisplacement`: Extract X, Y displacement from elastix deformation fields
    - Support for BSpline, Affine, Rigid, and Translation transforms
    - Automatic deformation field extraction
    - Displacement statistics and masking
  - `VerticalDisplacement`: Compute Z displacement from elevation difference
    - Direct elevation comparison (Fixed - Registered)
    - Uplift/subsidence classification
    - Gradient computation for rapid change detection
  - `DisplacementMagnitude`: Calculate displacement magnitudes
    - Horizontal magnitude (√(x² + y²))
    - 3D magnitude (√(x² + y² + z²))
    - Direction calculation and compass naming
    - Strain rate computation
    - Vector field preparation for visualization

- **Tiling System for Large Datasets**
  - `TileManager`: Automatic tiling for images > 5000x5000 pixels
    - Intelligent tile layout with overlap
    - Tile extraction and reconstruction
    - Multiple blend methods (average, feather, first)
    - Tile layout save/load functionality
  - `EdgeHandler`: Seamless tile blending
    - Feathering masks for smooth transitions
    - Distance-based and Gaussian weight masks
    - Edge artifact detection
    - Overlap region blending

- **Tests**
  - Comprehensive displacement module tests
  - Vertical displacement with synthetic subsidence data
  - Magnitude and direction calculation tests
  - Tiling functionality tests

### Added - Phase 1: Core Foundation

- **Project Structure**
  - Modular package architecture
  - Apache 2.0 license
  - Conda environment with GDAL support
  - Git repository initialization

- **I/O Module**
  - `RasterIO`: GeoTIFF and ASC file reading/writing
  - `NoDataHandler`: GDAL-compliant no-data value handling
  - `CRSManager`: CRS detection, validation, and comparison
  - ITK image conversion utilities

- **Registration Module**
  - `ElastixWrapper`: Simplified ITK-Elastix interface
  - `TwoPassRegistration`: Automated two-pass registration strategy
  - `ParameterManager`: Parameter file management

- **Parameter Files**
  - NA_NCC_ASGD.txt - Non-Affine (default for landslide monitoring)
  - AF_MI_ASGD.txt - Affine transform
  - RG_MI_ASGD.txt - Rigid/Euler transform
  - TS_MI_ASGD.txt - Translation only

- **Utilities**
  - Professional logging configuration
  - Timestamp generation
  - Directory management helpers
  - File size formatting

- **Examples**
  - Wrigley landslide dataset
  - Configuration file template
  - I/O module test script

- **Documentation**
  - README with quick start guide
  - Wrigley example documentation
  - Installation instructions

## [0.0.0] - 2024-11-20

### Added
- Initial project scaffolding
- Git repository setup
