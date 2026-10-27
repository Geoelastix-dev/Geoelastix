"""Parameter file management for ITK-Elastix."""

import logging
from pathlib import Path
import itk

logger = logging.getLogger("geoelastix.registration")


class ParameterManager:
    """
    Manage ITK-Elastix parameter files.

    Handles loading, validation, and modification of elastix parameter files.
    """

    # Default parameter files included with GeoElastix
    DEFAULT_PARAMETERS = {
        'NA': 'NA_NCC_ASGD.txt',        # Non-Affine (default for landslide)
        'AF': 'AF_MI_ASGD.txt',          # Affine
        'RG': 'RG_MI_ASGD.txt',          # Rigid/Euler
        'TS': 'TS_MI_ASGD.txt',          # Translation
    }

    @staticmethod
    def get_default_parameter_file(method='NA'):
        """
        Get path to default parameter file.

        Parameters
        ----------
        method : str, optional
            Registration method ('NA', 'AF', 'RG', 'TS'). Default: 'NA'

        Returns
        -------
        Path
            Path to parameter file

        Raises
        ------
        ValueError
            If method is not recognized
        FileNotFoundError
            If parameter file not found
        """
        method = method.upper()

        if method not in ParameterManager.DEFAULT_PARAMETERS:
            raise ValueError(
                f"Unknown registration method: {method}. "
                f"Available methods: {list(ParameterManager.DEFAULT_PARAMETERS.keys())}"
            )

        # Get parameter file path relative to this module
        param_filename = ParameterManager.DEFAULT_PARAMETERS[method]
        module_dir = Path(__file__).parent.parent.parent
        param_path = module_dir / 'parameters' / param_filename

        if not param_path.exists():
            raise FileNotFoundError(
                f"Parameter file not found: {param_path}"
            )

        logger.info(f"Using default parameter file: {param_filename}")
        return param_path

    @staticmethod
    def load_parameter_file(file_path):
        """
        Load elastix parameter file.

        Parameters
        ----------
        file_path : str or Path
            Path to parameter file

        Returns
        -------
        itk.ParameterObject
            ITK-Elastix parameter object

        Raises
        ------
        FileNotFoundError
            If file not found
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {file_path}")

        logger.info(f"Loading parameter file: {file_path.name}")

        # Create parameter object
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(str(file_path))

        return parameter_object

    @staticmethod
    def create_parameter_object(method='NA', custom_file=None):
        """
        Create parameter object from method name or custom file.

        Parameters
        ----------
        method : str, optional
            Registration method ('NA', 'AF', 'RG', 'TS'). Default: 'NA'
        custom_file : str or Path, optional
            Path to custom parameter file. If provided, method is ignored.

        Returns
        -------
        itk.ParameterObject
            ITK-Elastix parameter object
        """
        if custom_file is not None:
            logger.info(f"Using custom parameter file: {custom_file}")
            return ParameterManager.load_parameter_file(custom_file)
        else:
            param_file = ParameterManager.get_default_parameter_file(method)
            return ParameterManager.load_parameter_file(param_file)

    @staticmethod
    def get_parameter_map(parameter_object, index=0):
        """
        Get parameter map from parameter object.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        index : int, optional
            Index of parameter map (for multi-resolution). Default: 0

        Returns
        -------
        dict
            Parameter map as dictionary
        """
        return parameter_object.GetParameterMap(index)

    @staticmethod
    def set_parameter(parameter_object, key, value, index=0):
        """
        Set a parameter value in parameter object.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        key : str
            Parameter key
        value : str or list
            Parameter value(s)
        index : int, optional
            Parameter map index. Default: 0

        Returns
        -------
        itk.ParameterObject
            Modified parameter object
        """
        param_map = parameter_object.GetParameterMap(index)

        if isinstance(value, (list, tuple)):
            param_map[key] = [str(v) for v in value]
        else:
            param_map[key] = [str(value)]

        parameter_object.SetParameterMap(index, param_map)
        logger.debug(f"Set parameter {key} = {value}")

        return parameter_object

    @staticmethod
    def set_grid_schedule(parameter_object, schedule, index=0):
        """
        Set the B-spline GridSpacingSchedule (one factor per resolution).

        The final grid spacing is multiplied by these factors, e.g. [8, 4, 2, 1]
        starts coarse and refines the grid with the image pyramid.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        schedule : list of float
            Grid spacing factor for each resolution, coarsest first
        index : int, optional
            Parameter map index. Default: 0
        """
        n_res = int(ParameterManager.get_parameter(
            parameter_object, 'NumberOfResolutions', index)[0])
        if len(schedule) != n_res:
            raise ValueError(f"grid_schedule needs {n_res} values (NumberOfResolutions), "
                             f"got {len(schedule)}")
        return ParameterManager.set_parameter(
            parameter_object, 'GridSpacingSchedule', [float(s) for s in schedule], index)

    @staticmethod
    def set_bending_weight(parameter_object, weight, index=0):
        """
        Add a bending-energy penalty to the similarity metric.

        Switches to MultiMetricMultiResolutionRegistration with the existing
        metric (weight 1) plus TransformBendingEnergyPenalty (``weight``).
        Larger weights give smoother deformation fields. A weight of 0 leaves
        the parameters unchanged.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        weight : float
            Bending-energy weight relative to the similarity metric
        index : int, optional
            Parameter map index. Default: 0
        """
        if not weight:
            return parameter_object
        param_map = parameter_object.GetParameterMap(index)
        metric = param_map['Metric'][0]
        param_map['Registration'] = ['MultiMetricMultiResolutionRegistration']
        param_map['Metric'] = [metric, 'TransformBendingEnergyPenalty']
        param_map['Metric0Weight'] = ['1.0']
        param_map['Metric1Weight'] = [str(float(weight))]
        parameter_object.SetParameterMap(index, param_map)
        logger.debug(f"Added bending-energy penalty, weight {weight}")
        return parameter_object

    @staticmethod
    def get_parameter(parameter_object, key, index=0):
        """
        Get parameter value from parameter object.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        key : str
            Parameter key
        index : int, optional
            Parameter map index. Default: 0

        Returns
        -------
        list or None
            Parameter value(s), or None if not found
        """
        param_map = parameter_object.GetParameterMap(index)
        return param_map.get(key, None)

    @staticmethod
    def write_parameter_file(parameter_object, file_path, index=0):
        """
        Write parameter object to file.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        file_path : str or Path
            Output file path
        index : int, optional
            Parameter map index to write. Default: 0
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        parameter_object.WriteParameterFile(
            parameter_object.GetParameterMap(index),
            str(file_path)
        )

        logger.info(f"Wrote parameter file: {file_path}")

    @staticmethod
    def print_parameters(parameter_object, index=0):
        """
        Print parameter values for debugging.

        Parameters
        ----------
        parameter_object : itk.ParameterObject
            Parameter object
        index : int, optional
            Parameter map index. Default: 0
        """
        param_map = parameter_object.GetParameterMap(index)

        logger.info("Elastix Parameters:")
        for key, value in param_map.items():
            logger.info(f"  {key}: {value}")

    @staticmethod
    def list_available_methods():
        """
        List available registration methods.

        Returns
        -------
        dict
            Dictionary of method codes and descriptions
        """
        methods = {
            'NA': 'Non-Affine (deformable registration) - Default for landslide monitoring',
            'AF': 'Affine Transform (rotation, translation, scaling, shearing)',
            'RG': 'Rigid/Euler Transform (rotation and translation only)',
            'TS': 'Translation (translation only, no rotation or scaling)',
        }
        return methods
