"""End-to-end workflow orchestration."""

import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import itk

from ..io import RasterIO, NoDataHandler, CRSManager
from ..registration import (ElastixWrapper, TwoPassRegistration, ParameterManager,
                            RegistrationMask)
from ..displacement import HorizontalDisplacement, VerticalDisplacement, DisplacementMagnitude
from ..tiling import TileManager
from ..validation import QualityMetrics, QualityChecker
from ..visualization import QuiverPlot, ContourPlot, ReportGenerator
from ..utils import setup_logging, generate_timestamp, get_output_directory, ensure_dir

logger = logging.getLogger("geoelastix.workflow")


class WorkflowOrchestrator:
    """
    Orchestrate the complete GeoElastix workflow.

    Integrates all modules to perform end-to-end registration and displacement analysis.
    """

    def __init__(self, config):
        """
        Initialize workflow orchestrator.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None

        logger.info("="*70)
        logger.info("GEOELASTIX WORKFLOW INITIALIZED")
        logger.info("="*70)

    def run(self):
        """
        Execute the complete workflow.

        Returns
        -------
        dict
            Results dictionary with all outputs
        """
        self.start_time = datetime.now()
        logger.info(f"Workflow started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Setup
            self._setup_output_directory()

            # Step 2: Load input data
            self._load_input_data()

            # Step 3: Perform registration
            self._perform_registration()

            # Step 4: Calculate displacement
            self._calculate_displacement()

            # Step 5: Compute quality metrics
            self._compute_quality_metrics()

            # Step 6: Generate visualizations
            self._generate_visualizations()

            # Step 7: Generate reports
            self._generate_reports()

            # Step 8: Save outputs
            self._save_outputs()

            self.end_time = datetime.now()
            elapsed = (self.end_time - self.start_time).total_seconds()

            logger.info("="*70)
            logger.info(f"✓ WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info(f"  Total time: {elapsed:.1f} seconds")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info("="*70)

            return self.results

        except Exception as e:
            self.end_time = datetime.now()
            logger.error(f"✗ Workflow failed: {e}")
            raise

    def _setup_output_directory(self):
        """Set up output directory structure."""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Setup Output Directory")
        logger.info("="*70)

        # Get output directory with timestamp
        base_dir = self.config['job']['output_dir']
        job_id = self.config['job']['id']
        description = self.config['job'].get('description')

        self.output_dir = get_output_directory(base_dir, job_id, description)
        ensure_dir(self.output_dir)

        # Create subdirectories
        self.displacement_dir = ensure_dir(self.output_dir / 'displacement')
        self.registered_dir = ensure_dir(self.output_dir / 'registered')
        self.visualization_dir = ensure_dir(self.output_dir / 'visualization')
        self.logs_dir = ensure_dir(self.output_dir / 'logs')
        self.params_dir = ensure_dir(self.output_dir / 'parameters')

        logger.info(f"✓ Output directory: {self.output_dir}")

        # Set up logging to file
        if self.config['logging']['save_to_file']:
            log_file = self.logs_dir / 'geoelastix.log'
            # Reconfigure logging with file
            setup_logging(
                log_level=self.config['logging']['level'],
                log_file=log_file,
                log_to_console=self.config['logging']['log_to_console']
            )

    def _load_input_data(self):
        """Load and validate input raster data."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Load Input Data")
        logger.info("="*70)

        # Load fixed image
        logger.info("\nLoading fixed image...")
        fixed_path = Path(self.config['input']['fixed_image'])
        fixed_data = RasterIO.read_raster(fixed_path)

        self.fixed_array = fixed_data['array']
        self.fixed_mask = fixed_data['mask']
        self.fixed_nodata = fixed_data['nodata']
        self.fixed_crs = fixed_data['crs']
        self.fixed_geotransform = fixed_data['geotransform']
        self.image_shape = fixed_data['shape']

        # Load moving image
        logger.info("\nLoading moving image...")
        moving_path = Path(self.config['input']['moving_image'])
        # Warped onto the fixed grid if extent/pixel size/CRS differ
        moving_data = RasterIO.read_raster(moving_path, reference=fixed_data)

        self.moving_array = moving_data['array']
        self.moving_mask = moving_data['mask']
        self.moving_nodata = moving_data['nodata']
        self.moving_crs = moving_data['crs']

        # Validate CRS compatibility
        logger.info("\nValidating CRS...")
        from osgeo import gdal
        fixed_ds = gdal.Open(str(fixed_path))
        moving_ds = gdal.Open(str(moving_path))
        CRSManager.validate_crs_pair(fixed_ds, moving_ds)
        fixed_ds = None
        moving_ds = None

        # Get pixel spacing
        self.pixel_spacing = CRSManager.get_pixel_size(gdal.Open(str(fixed_path)))
        logger.info(f"  Pixel spacing: {self.pixel_spacing[0]:.6f} x {self.pixel_spacing[1]:.6f}")

        self._build_registration_mask(fixed_data)

        logger.info("\n✓ Input data loaded successfully")

    def _build_registration_mask(self, fixed_data):
        """Pixels the similarity metric is sampled from (fixed image valid data by default)."""
        mask_cfg = self.config['registration'].get('mask') or {}
        self.registration_mask = self.fixed_mask.astype(np.uint8)

        if mask_cfg.get('terrain'):
            logger.info("\nBuilding terrain registration mask...")
            self.registration_mask &= RegistrationMask.terrain_mask(
                self.fixed_array, self.fixed_mask, self.pixel_spacing[0],
                min_slope=mask_cfg.get('min_slope', 5.0),
                window=mask_cfg.get('window', 30.0)
            )

        if mask_cfg.get('file'):
            logger.info(f"\nLoading registration mask: {mask_cfg['file']}")
            user = RasterIO.read_raster(Path(mask_cfg['file']), reference=fixed_data)
            use = np.isfinite(user['array']) & (user['array'] != 0) & (user['mask'] > 0)
            self.registration_mask &= use.astype(np.uint8)

        self.mask_outputs = bool(mask_cfg.get('mask_outputs', True))

    def _perform_registration(self):
        """Perform image registration."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Perform Registration")
        logger.info("="*70)

        # Get parameter object
        method = self.config['registration']['method']
        custom_param = self.config['registration'].get('parameter_file')

        logger.info(f"\nRegistration method: {method}")
        self.parameter_object = ParameterManager.create_parameter_object(
            method=method, custom_file=custom_param
        )

        # Images are registered with ITK spacing [1, 1], so elastix "physical
        # units" are pixels. Convert a grid spacing given in map units.
        grid_spacing = self.config['registration'].get('grid_spacing')
        if grid_spacing is not None:
            grid_spacing_px = float(grid_spacing) / self.pixel_spacing[0]
            ParameterManager.set_parameter(
                self.parameter_object, 'FinalGridSpacingInPhysicalUnits',
                f"{grid_spacing_px:.4f}"
            )
            logger.info(
                f"  B-spline grid spacing: {grid_spacing} map units "
                f"= {grid_spacing_px:.2f} pixels"
            )

        grid_schedule = self.config['registration'].get('grid_schedule')
        if grid_schedule:
            ParameterManager.set_grid_schedule(self.parameter_object, grid_schedule)
            logger.info(f"  Grid spacing schedule: {grid_schedule}")

        bending_weight = self.config['registration'].get('bending_weight') or 0.0
        if bending_weight > 0:
            ParameterManager.set_bending_weight(self.parameter_object, bending_weight)
            logger.info(f"  Bending-energy penalty weight: {bending_weight}")

        height, width = self.image_shape
        self.displacement_x = np.full(self.image_shape, np.nan, dtype=np.float32)
        self.displacement_y = np.full(self.image_shape, np.nan, dtype=np.float32)
        self.registered_array = np.full(self.image_shape, np.nan, dtype=np.float32)
        self.valid_mask = np.zeros(self.image_shape, dtype=np.uint8)

        tile_size = self.config['processing']['tile_threshold']
        overlap = self.config['processing']['tile_overlap']
        tile_manager = TileManager(self.image_shape, tile_size=tile_size, overlap=overlap)

        if not tile_manager.needs_tiling:
            self._register_block((slice(0, height), slice(0, width)),
                                 (slice(0, height), slice(0, width)))
        else:
            n_tiles = len(tile_manager.tiles)
            for k, tile in enumerate(tile_manager.tiles, start=1):
                logger.info(f"\nTile {k}/{n_tiles} {tile.index}: "
                            f"rows {tile.row_start}-{tile.row_end}, "
                            f"cols {tile.col_start}-{tile.col_end}")
                self._register_block(tile.get_slice(),
                                     tile_manager.get_core_slice(tile))

        logger.info("\n✓ Registration completed")

    def _register_block(self, block, core):
        """
        Register one block (whole image or tile) and store its core region.

        Parameters
        ----------
        block : tuple of slice
            Rows/cols of the block in the full image
        core : tuple of slice
            Rows/cols of the block's output that are kept (overlap trimmed)
        """
        min_valid = self.config['processing'].get('min_valid_fraction', 0.05)

        fixed_mask = self.fixed_mask[block]
        moving_mask = self.moving_mask[block]
        reg_mask = self.registration_mask[block]
        if reg_mask.mean() < min_valid or moving_mask.mean() < min_valid:
            logger.info("  Skipped - not enough valid data")
            return

        # Elastix must not see no-data values; fill them, masks exclude them
        fixed_itk = RasterIO.array_to_itk_image(
            NoDataHandler.fill_nodata(self.fixed_array[block], fixed_mask)
        )
        moving_itk = RasterIO.array_to_itk_image(
            NoDataHandler.fill_nodata(self.moving_array[block], moving_mask)
        )
        # The metric is sampled only inside the registration mask
        fixed_mask_itk = RasterIO.array_to_itk_image(reg_mask, is_mask=True)
        moving_mask_itk = RasterIO.array_to_itk_image(moving_mask, is_mask=True)

        # Perform two-pass registration
        two_pass = TwoPassRegistration(self.parameter_object)
        reg_results = two_pass.register(
            fixed_itk, moving_itk,
            fixed_mask=fixed_mask_itk,
            moving_mask=moving_mask_itk,
            log_to_console=self.config['registration']['log_to_console']
        )
        transform_parameters = reg_results['transform_parameters']
        registered = RasterIO.itk_image_to_array(reg_results['registered_image'])

        # Pixels whose source location is outside the valid moving data
        warped_mask = itk.transformix_filter(
            RasterIO.array_to_itk_image(moving_mask.astype(np.float32)),
            transform_parameters
        )
        valid = (fixed_mask > 0) & (itk.array_from_image(warped_mask) > 0.999)
        if self.mask_outputs:
            # Displacements outside the mask are B-spline interpolation, not measurement
            valid &= reg_mask > 0

        horiz = HorizontalDisplacement.compute_from_transform(
            fixed_itk, transform_parameters,
            geotransform=self.fixed_geotransform
        )

        rows, cols = core
        local = (slice(rows.start - block[0].start, rows.stop - block[0].start),
                 slice(cols.start - block[1].start, cols.stop - block[1].start))
        keep = valid[local]

        for target, source in ((self.displacement_x, horiz['displacement_x']),
                               (self.displacement_y, horiz['displacement_y']),
                               (self.registered_array, registered)):
            target[core] = np.where(keep, source[local], np.nan)
        self.valid_mask[core] = keep

        # Keep the last transform for reporting
        self.transform_parameters = transform_parameters

    def _calculate_displacement(self):
        """Calculate displacement fields."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Calculate Displacement")
        logger.info("="*70)

        # Horizontal displacement (X, Y) was computed per block in step 3
        horiz_stats = HorizontalDisplacement.get_statistics(
            self.displacement_x, self.displacement_y,
            mask=self.valid_mask
        )

        # Vertical displacement (Z) = newer elevation - older elevation of the
        # same (displaced) ground point
        logger.info("\nCalculating vertical displacement...")
        vert_result = VerticalDisplacement.compute_from_arrays(
            self.fixed_array, self.registered_array,
            mask=self.valid_mask
        )

        self.displacement_z = vert_result['displacement_z']
        vert_stats = vert_result['statistics']

        # Magnitude
        logger.info("\nCalculating displacement magnitude...")
        mag_result = DisplacementMagnitude.compute_horizontal_magnitude(
            self.displacement_x, self.displacement_y
        )

        self.displacement_magnitude = mag_result['magnitude']
        self.displacement_direction = mag_result['direction']
        mag_stats = mag_result['statistics']

        # Combine statistics
        self.displacement_stats = {**horiz_stats, **vert_stats, **mag_stats}

        logger.info("\n✓ Displacement calculated")

    def _compute_quality_metrics(self):
        """Compute quality metrics."""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Compute Quality Metrics")
        logger.info("="*70)

        # Compute all metrics
        self.metrics = QualityMetrics.compute_all_metrics(
            self.fixed_array,
            np.nan_to_num(self.registered_array, nan=0.0),
            fixed_mask=self.fixed_mask,
            registered_mask=self.valid_mask,
            image_shape=self.image_shape
        )

        # Quality checks
        quality_thresholds = self.config['quality']
        checker = QualityChecker(thresholds=quality_thresholds)

        self.quality_checks = checker.check_metrics(self.metrics)
        self.quality_checks_disp = checker.check_displacement(self.displacement_stats)

        logger.info("\n✓ Quality metrics computed")

    def _generate_visualizations(self):
        """Generate visualization plots."""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Generate Visualizations")
        logger.info("="*70)

        if not self.config['visualization']['generate_plots']:
            logger.info("Visualization disabled in configuration")
            return

        plot_types = self.config['visualization']['plot_types']
        dpi = self.config['visualization']['dpi']
        colormap = self.config['visualization']['colormap']
        subsample = self.config['visualization']['quiver_subsample']

        self.plot_files = []

        # Quiver plot
        if 'quiver' in plot_types:
            logger.info("\nGenerating quiver plot...")
            fig = QuiverPlot.create_quiver_plot(
                self.displacement_x, self.displacement_y,
                subsample=subsample, dpi=dpi, colormap=colormap,
                mask=self.fixed_mask,
                pixel_spacing=self.pixel_spacing,
                geotransform=self.fixed_geotransform
            )
            output_path = self.visualization_dir / 'horizontal_quiver.png'
            QuiverPlot.save_plot(fig, output_path)
            self.plot_files.append(output_path)

        # Contour plots
        contour_types = ['contour_x', 'contour_y', 'contour_z', 'contour_magnitude']
        data_map = {
            'contour_x': (self.displacement_x, 'X Displacement (East-West)', 'RdBu_r'),
            'contour_y': (self.displacement_y, 'Y Displacement (North-South)', 'RdBu_r'),
            'contour_z': (self.displacement_z, 'Z Displacement (Vertical)', 'RdBu_r'),
            'contour_magnitude': (self.displacement_magnitude, 'Horizontal Magnitude', 'viridis'),
        }

        for ctype in contour_types:
            if ctype in plot_types:
                data, title, cmap = data_map[ctype]
                logger.info(f"\nGenerating {ctype} contour plot...")

                fig = ContourPlot.create_contour_plot(
                    data, title=title, dpi=dpi, colormap=cmap,
                    mask=self.fixed_mask,
                    center_zero=(ctype != 'contour_magnitude'),
                    pixel_spacing=self.pixel_spacing,
                    geotransform=self.fixed_geotransform
                )

                filename = f"{ctype.replace('contour_', 'displacement_')}_contour.png"
                output_path = self.visualization_dir / filename
                ContourPlot.save_plot(fig, output_path)
                self.plot_files.append(output_path)

        logger.info(f"\n✓ Generated {len(self.plot_files)} visualizations")

    def _generate_reports(self):
        """Generate reports."""
        logger.info("\n" + "="*70)
        logger.info("STEP 7: Generate Reports")
        logger.info("="*70)

        report_gen = ReportGenerator(self.output_dir)

        # Job information
        job_info = {
            'id': self.config['job']['id'],
            'description': self.config['job'].get('description'),
            'timestamp': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'fixed_image': str(self.config['input']['fixed_image']),
            'moving_image': str(self.config['input']['moving_image']),
            'method': self.config['registration']['method'],
            'metric': self.config['registration']['metric'],
        }

        # Generate text report
        logger.info("\nGenerating text report...")
        report_path = report_gen.generate_text_report(
            job_info, self.metrics, self.displacement_stats,
            self.quality_checks
        )

        # Save metrics to JSON
        logger.info("\nSaving metrics to JSON...")
        json_path = report_gen.save_metrics_json(
            self.metrics, self.displacement_stats,
            self.quality_checks, output_filename='metrics.json'
        )

        logger.info("\n✓ Reports generated")

    def _save_outputs(self):
        """Save output files."""
        logger.info("\n" + "="*70)
        logger.info("STEP 8: Save Outputs")
        logger.info("="*70)

        output_formats = self.config['output']['formats']
        compression = self.config['output']['compression']

        # Save registered image
        logger.info("\nSaving registered image...")
        if 'geotiff' in output_formats:
            RasterIO.write_geotiff(
                self.registered_dir / 'registered_moving.tif',
                self.registered_array,
                self.fixed_geotransform,
                self.fixed_crs,
                nodata_value=self.fixed_nodata,
                compression=compression
            )

        if 'geotiff' in output_formats:
            RasterIO.write_geotiff(
                self.registered_dir / 'registration_mask.tif',
                self.registration_mask, self.fixed_geotransform, self.fixed_crs,
                nodata_value=255, compression=compression
            )

        # Save displacement fields
        logger.info("\nSaving displacement fields...")
        displacement_data = {
            'displacement_x': self.displacement_x,
            'displacement_y': self.displacement_y,
            'displacement_z': self.displacement_z,
        }

        if self.config['output']['generate_magnitude']:
            displacement_data['displacement_magnitude'] = self.displacement_magnitude

        for name, data in displacement_data.items():
            if 'geotiff' in output_formats:
                RasterIO.write_geotiff(
                    self.displacement_dir / f'{name}.tif',
                    data, self.fixed_geotransform, self.fixed_crs,
                    nodata_value=-9999, compression=compression
                )

            if 'asc' in output_formats:
                RasterIO.write_asc(
                    self.displacement_dir / f'{name}.asc',
                    data, self.fixed_geotransform, nodata_value=-9999
                )

        logger.info(f"\n✓ Outputs saved to: {self.output_dir}")
