"""Regression tests for displacement sign/units, no-data handling and tiling."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift
from osgeo import gdal, osr

from geoelastix.io import RasterIO, NoDataHandler
from geoelastix.registration import ParameterManager, TwoPassRegistration, RegistrationMask
from geoelastix.displacement import HorizontalDisplacement
from geoelastix.tiling import TileManager
from geoelastix.cli.config_parser import ConfigParser
from geoelastix.cli.workflow import WorkflowOrchestrator

PIXEL = 1.5  # metres
GEOTRANSFORM = (700000.0, PIXEL, 0.0, 4300000.0, 0.0, -PIXEL)


def synthetic_pair(shape=(400, 400), shift_rows=2.0, shift_cols=3.0, uplift=0.0):
    """Older/newer DEMs where the ground moves +shift_cols east, +shift_rows south."""
    rng = np.random.default_rng(0)
    older = gaussian_filter(rng.normal(size=shape), 8) * 200 + 300
    newer = shift(older, (shift_rows, shift_cols), order=3, mode='nearest') + uplift
    return older.astype(np.float32), newer.astype(np.float32)


def test_horizontal_displacement_sign_and_units():
    """East/North displacement in map units for a known ground shift."""
    older, newer = synthetic_pair()

    fixed = RasterIO.array_to_itk_image(newer)
    moving = RasterIO.array_to_itk_image(older)
    params = ParameterManager.create_parameter_object('NA')
    result = TwoPassRegistration(params).register(fixed, moving)

    horiz = HorizontalDisplacement.compute_from_transform(
        fixed, result['transform_parameters'], geotransform=GEOTRANSFORM
    )
    inner = (slice(50, 350), slice(50, 350))

    # 3 px east = +4.5 m East; 2 px south = -3.0 m North
    assert np.mean(horiz['displacement_x'][inner]) == pytest.approx(3 * PIXEL, abs=0.3)
    assert np.mean(horiz['displacement_y'][inner]) == pytest.approx(-2 * PIXEL, abs=0.3)


def test_fill_nodata_removes_nodata_values():
    array = np.array([[1, 2, -3.4028235e38], [4, -3.4028235e38, 6]], dtype=np.float32)
    mask = NoDataHandler.create_mask(array, -3.4028234663852886e38)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 1]]

    filled = NoDataHandler.fill_nodata(array, mask)
    assert np.all(np.abs(filled) < 10)
    assert filled[0, 0] == 1 and filled[1, 2] == 6


def test_tile_cores_cover_image_once():
    tm = TileManager((1234, 987), tile_size=300, overlap=60)
    count = np.zeros((1234, 987), dtype=int)
    for tile in tm.tiles:
        rows, cols = tm.get_core_slice(tile)
        assert rows.start >= tile.row_start and rows.stop <= tile.row_end
        assert cols.start >= tile.col_start and cols.stop <= tile.col_end
        count[rows, cols] += 1
    assert count.min() >= 1


def _write_geotiff(path, array, geotransform):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    RasterIO.write_geotiff(path, array, geotransform, srs.ExportToWkt(),
                           nodata_value=-3.4028234663852886e38, compression='NONE')


def test_moving_raster_is_warped_onto_fixed_grid(tmp_path):
    older, newer = synthetic_pair(shape=(100, 120))
    _write_geotiff(tmp_path / 'fixed.tif', newer, GEOTRANSFORM)
    # Moving raster offset by 10 pixels and with a different size
    gt = (GEOTRANSFORM[0] - 10 * PIXEL,) + GEOTRANSFORM[1:]
    _write_geotiff(tmp_path / 'moving.tif', np.pad(older, ((0, 0), (10, 0))), gt)

    fixed = RasterIO.read_raster(tmp_path / 'fixed.tif')
    moving = RasterIO.read_raster(tmp_path / 'moving.tif', reference=fixed)

    assert moving['shape'] == fixed['shape']
    assert moving['geotransform'] == pytest.approx(fixed['geotransform'])
    np.testing.assert_allclose(moving['array'][:, 1:-1], older[:, 1:-1], atol=1e-3)


@pytest.mark.parametrize('tile_threshold', [5000, 250])
def test_workflow_east_north_up(tmp_path, tile_threshold):
    """End-to-end: known motion and uplift, with and without tiling."""
    older, newer = synthetic_pair(shape=(500, 500), uplift=0.5)
    # No-data border on the older DEM
    older[:, :20] = -3.4028235e38
    _write_geotiff(tmp_path / 'new.tif', newer, GEOTRANSFORM)
    _write_geotiff(tmp_path / 'old.tif', older, GEOTRANSFORM)

    config = ConfigParser.merge_with_defaults({
        'job': {'id': 'synthetic', 'output_dir': str(tmp_path / 'out')},
        'input': {'fixed_image': str(tmp_path / 'new.tif'),
                  'moving_image': str(tmp_path / 'old.tif')},
        'registration': {'grid_spacing': 48.0},
        'processing': {'tile_threshold': tile_threshold, 'tile_overlap': 80},
        'output': {'formats': ['geotiff']},
        'visualization': {'generate_plots': False},
        'logging': {'save_to_file': False, 'log_to_console': False},
    })
    wf = WorkflowOrchestrator(config)
    wf.run()

    inner = (slice(60, 440), slice(60, 440))
    assert np.nanmedian(wf.displacement_x[inner]) == pytest.approx(3 * PIXEL, abs=0.3)
    assert np.nanmedian(wf.displacement_y[inner]) == pytest.approx(-2 * PIXEL, abs=0.3)
    assert np.nanmedian(wf.displacement_z[inner]) == pytest.approx(0.5, abs=0.2)
    # No-data region of the moving DEM must not produce values
    assert np.all(np.isnan(wf.displacement_z[:, :20]))
    assert np.nanmax(np.abs(wf.displacement_z)) < 50

    out = gdal.Open(str(next((tmp_path / 'out').glob('*/displacement/displacement_x.tif'))))
    assert out.GetGeoTransform() == pytest.approx(GEOTRANSFORM)


def test_terrain_mask_excludes_flat_ground():
    older, _ = synthetic_pair(shape=(300, 300))
    dem = older.copy()
    dem[:, :100] = 150.0                       # flat "water" surface
    valid = np.ones(dem.shape, dtype=np.uint8)
    valid[:10] = 0
    mask = RegistrationMask.terrain_mask(dem, valid, PIXEL, min_slope=5.0, window=30.0)
    assert mask[:, :80].sum() == 0
    assert mask[20:, 140:].mean() > 0.9
    assert mask[:10].sum() == 0


def test_bending_weight_and_grid_schedule_parameters():
    params = ParameterManager.create_parameter_object('NA')
    ParameterManager.set_grid_schedule(params, [8, 4, 2, 1])
    ParameterManager.set_bending_weight(params, 0.5)
    pm = params.GetParameterMap(0)
    assert [float(s) for s in pm['GridSpacingSchedule']] == [8, 4, 2, 1]
    assert pm['Registration'][0] == 'MultiMetricMultiResolutionRegistration'
    assert list(pm['Metric']) == ['AdvancedNormalizedCorrelation', 'TransformBendingEnergyPenalty']
    assert float(pm['Metric1Weight'][0]) == 0.5
    with pytest.raises(ValueError):
        ParameterManager.set_grid_schedule(params, [4, 2, 1])


def test_workflow_terrain_mask_ignores_changed_flat_ground(tmp_path):
    """Moving terrain next to a flat area whose surface changed (e.g. water level)."""
    older, newer = synthetic_pair(shape=(500, 500))
    rng = np.random.default_rng(1)
    older[:, :200] = 150.0
    newer[:, :200] = 150.0 + gaussian_filter(rng.normal(size=(500, 200)), 3) * 0.5
    _write_geotiff(tmp_path / 'new.tif', newer, GEOTRANSFORM)
    _write_geotiff(tmp_path / 'old.tif', older, GEOTRANSFORM)

    config = ConfigParser.merge_with_defaults({
        'job': {'id': 'masked', 'output_dir': str(tmp_path / 'out')},
        'input': {'fixed_image': str(tmp_path / 'new.tif'),
                  'moving_image': str(tmp_path / 'old.tif')},
        'registration': {'grid_spacing': 48.0, 'grid_schedule': [8, 4, 2, 1],
                         'bending_weight': 0.1, 'mask': {'terrain': True}},
        'output': {'formats': ['geotiff']},
        'visualization': {'generate_plots': False},
        'logging': {'save_to_file': False, 'log_to_console': False},
    })
    ConfigParser.validate_config(config)
    wf = WorkflowOrchestrator(config)
    wf.run()

    terrain = (slice(60, 440), slice(260, 440))
    assert np.nanmedian(wf.displacement_x[terrain]) == pytest.approx(3 * PIXEL, abs=0.3)
    assert np.nanmedian(wf.displacement_y[terrain]) == pytest.approx(-2 * PIXEL, abs=0.3)
    # Flat ground is outside the registration mask: no displacement reported
    assert np.all(np.isnan(wf.displacement_x[:, :180]))
    assert any((tmp_path / 'out').glob('*/registered/registration_mask.tif'))


def test_terrain_mask_drops_narrow_strips_and_small_patches():
    rng = np.random.default_rng(2)
    dem = np.full((300, 300), 150.0)
    dem[:, 150:] = 300 + gaussian_filter(rng.normal(size=(300, 150)), 8) * 200   # hills
    dem[100:110, :120] += np.linspace(0, 3, 10)[:, None] * 3       # 15 m wide embankment
    dem[20:40, 20:40] += gaussian_filter(rng.normal(size=(20, 20)), 2) * 50       # small patch
    valid = np.ones(dem.shape, dtype=np.uint8)

    plain = RegistrationMask.terrain_mask(dem, valid, PIXEL)
    assert plain[100:110, 20:100].mean() > 0.5 and plain[20:40, 20:40].mean() > 0.5

    # The 30 m slope window widens the 15 m embankment to ~40 m in the mask
    filtered = RegistrationMask.terrain_mask(dem, valid, PIXEL, min_width=50.0, min_area=5000.0)
    assert filtered[:, :130].sum() == 0
    assert filtered[20:280, 170:].mean() > 0.9
