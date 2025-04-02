"""BMI wrapper for interfacing dHBV 2.0 with NOAA-OWP NextGen framework.

Motivated by LSTM BMI of Austin Raney, Jonathan Frame.
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union, Iterable

import numpy as np
from numpy.typing import NDArray
import torch
import yaml
from bmipy import Bmi
# from dMG.conf import config
# from dMG.core.data import take_sample_test
from dMG import ModelHandler, import_data_sampler
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
# from ruamel.yaml import YAML
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# -------------------------------------------- #
# Dynamic input variables (CSDMS standard names)
# -------------------------------------------- #
_dynamic_input_vars = [
    ('atmosphere_water__liquid_equivalent_precipitation_rate', 'mm d-1'),
    ('land_surface_air__temperature', 'degC'),
    ('land_surface_water__potential_evaporation_volume_flux', 'mm d-1'),
]

# ------------------------------------------- #
# Static input variables (CSDMS standard names)
# ------------------------------------------- #
_static_input_vars = [
    ('basin__area', 'km2'),
    ('land_surface_water__Hargreaves_potential_evaporation_volume_flux', 'mm d-1'),
    ('free_land_surface_water', 'mm d-1'),
    ('soil_clay__attr', 'percent'),
    ('soil_gravel__attr', 'percent'),
    ('soil_sand__attr', 'percent'),
    ('soil_silt__attr', 'percent'),
    ('land_vegetation__normalized_diff_vegetation_index', '-'),
    ('soil_active-layer__porosity', '-'),
    ('soil_clay__grid', 'km2'),
    ('soil_sand__grid', 'km2'),
    ('soil_silt__grid', 'km2'),
    ('soil_clay__volume_fraction', 'percent'),
    ('soil_gravel__volume_fraction', 'percent'),
    ('soil_sand__volume_fraction', 'percent'),
    ('soil_silt__volume_fraction', 'percent'),
    ('ratio__mean_potential_evapotranspiration__mean_precipitation', '-'),
    ('land_surface_water__glacier_fraction', 'percent'),
    ('atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate', 'mm d-1'),
    ('atmosphere_water__daily_mean_of_temperature', 'degC'),
    ('basin__mean_of_elevation', 'm'),
    ('basin__mean_of_slope', 'm km-1'),
    ('bedrock__permeability', 'm2'),
    ('p_seasonality', '-'),
    ('land_surface_water__potential_evaporation_volume_flux_seasonality', '-'),
    ('land_surface_water__snow_fraction', 'percent'),
    ('atmosphere_water__precipitation_falling_as_snow_fraction', 'percent'),
    ('land_surface_water__permafrost_fraction', '-'),
]

# ------------------------------------- #
# Output variables (CSDMS standard names)
# ------------------------------------- #
_output_vars = [
    ('land_surface_water__runoff_volume_flux', 'm3 s-1'),
]

# ---------------------------------------------- #
# Internal variable names <-> CSDMS standard names
# ---------------------------------------------- #
_var_name_internal_map = {
    # ----------- Dynamic inputs -----------
    'P': 'atmosphere_water__liquid_equivalent_precipitation_rate',
    'Temp': 'land_surface_air__temperature',
    'PET': 'land_surface_water__potential_evaporation_volume_flux',
    # ----------- Static inputs -----------
    'uparea': 'basin__area',
    'ETPOT_Hargr': 'land_surface_water__Hargreaves_potential_evaporation_volume_flux',
    'FW': 'free_land_surface_water',
    'HWSD_clay': 'soil_clay__attr',
    'HWSD_gravel': 'soil_gravel__attr',
    'HWSD_sand': 'soil_sand__attr',
    'HWSD_silt': 'soil_silt__attr',
    'NDVI': 'land_vegetation__normalized_diff_vegetation_index',
    'Porosity': 'soil_active-layer__porosity',
    'SoilGrids1km_clay': 'soil_clay__grid',
    'SoilGrids1km_sand': 'soil_sand__grid',
    'SoilGrids1km_silt': 'soil_silt__grid',
    'T_clay': 'soil_clay__volume_fraction',
    'T_gravel': 'soil_gravel__volume_fraction',
    'T_sand': 'soil_sand__volume_fraction',
    'T_silt': 'soil_silt__volume_fraction',
    'aridity': 'ratio__mean_potential_evapotranspiration__mean_precipitation',
    'glaciers': 'land_surface_water__glacier_fraction',
    'meanP': 'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate',
    'meanTa': 'atmosphere_water__daily_mean_of_temperature',
    'meanelevation': 'basin__mean_of_elevation',
    'meanslope': 'basin__mean_of_slope',
    'permeability': 'bedrock__permeability',
    'seasonality_P': 'p_seasonality',
    'seasonality_PET': 'land_surface_water__potential_evaporation_volume_flux_seasonality',
    'snow_fraction': 'land_surface_water__snow_fraction',
    'snowfall_fraction': 'atmosphere_water__precipitation_falling_as_snow_fraction',
    'permafrost': 'land_surface_water__permafrost_fraction',
    # ----------- Outputs -----------
    'flow_sim': 'land_surface_water__runoff_volume_flux',
}

_var_name_external_map = {v: k for k, v in _var_name_internal_map.items()}


def map_to_external(name: str):
    """Return the external name (exposed via BMI) for a given internal name."""
    return _var_name_internal_map[name]


def map_to_internal(name: str):
    """Return the internal name for a given external name (exposed via BMI)."""
    return _var_name_external_map[name]




################################################################################
################################################################################


# MAIN BMI >>>>


################################################################################
################################################################################


def bmi_array(arr: list[float]) -> NDArray:
    """Trivial wrapper function to ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype="float64")


class deltaModelBmi(Bmi):
    """
    dHBV 2.0UH BMI: NextGen-compatible, differentiable, physics-informed ML
    model for hydrologic forecasting. (Song et al., 2024)

    Note: This dHBV 2.0UH BMI can only run forward inference. To train,
        see dMG package (https://github.com/mhpi/generic_deltaModel).
    """
    _att_map = {
        'model_name':         'dHVB 2.0UH for NextGen',
        'version':            '2.0',
        'author_name':        'Leo Lonzarich',
        'time_step_size':     86400,
        'time_units':         'seconds',
        # 'time_step_type':     '',
        # 'grid_type':          'scalar',
        # 'step_method':        '',
    }
    
    def __init__(
            self,
            config_path: Optional[str] = None,
            verbose=False,
        ) -> None:
        """Create a BMI dHBV 2.0UH model ready for initialization.

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        verbose
            Enables debug print statements if True.
        """
        super().__init__()
        self._model = None
        self._initialized = False
        self.verbose = verbose

        self._var_loc = 'node'
        self._var_grid_id = 0

        self._start_time = 0.0
        self._end_time = np.finfo('d').max
        self._time_units = 's'

        self.config_bmi = None
        self.config_model = None

        # Timing BMI computations
        t_start = time.time()
        self.bmi_process_time = 0

        # Read BMI and model configuration files.
        if config_path is not None:
            if not Path(config_path).is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path, 'r') as f:
                self.config_bmi = yaml.safe_load(f)
            
            try:
                with open(self.config_bmi.get('config_model'), 'r') as f:
                    self.config_model = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration: {e}")
        
            self.sampler = import_data_sampler(self.config_model['data_sampler'])(self.config_model)

                    
        # Initialize variables.
        self._dynamic_var = self._set_vars(_dynamic_input_vars, bmi_array([0.0]))
        self._static_var = self._set_vars(_static_input_vars, bmi_array([0.0]))
        self._output_vars = self._set_vars(_output_vars, bmi_array([0.0]))

        # Track total BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI init took {time.time() - t_start} s")

    @staticmethod
    def _set_vars(
        vars: list[tuple[str, str]],
        var_value: NDArray,
    ) -> dict[str, dict[str, Union[NDArray, str]]]:
        """Set the values of the given variables."""
        var_dict = {}
        for item in vars:
            var_dict[item[0]] = {'value': var_value.copy(), 'units': item[1]}
        return var_dict
    
    def initialize(self, config_path: Optional[str] = None) -> None:
        """(Control function) Initialize the BMI model.

        This BMI operates in two modes:
            (Necessesitated by the fact that dhBV 2.0's internal NN must forward
            on all data at once. <-- Forwarding on each timestep one-by-one with
            saving/loading hidden states would slash LSTM performance. However,
            feeding in hidden states day-by-day leeds to great efficiency losses
            vs simply feeding all data at once due to carrying gradients at each
            step.)

            1) Feed all input dataBMI before
                'bmi.initialize()'. Then internal model is forwarded on all data
                and generates predictions during '.initialize()'.
            
            2) Run '.initialize()', then pass data day by day as normal during
                'bmi.update()'. If forwarding period is sufficiently small (say,
                <100 days), then forwarding LSTM on individual days with saved
                states is reasonable.

        To this end, a configuration file can be specified either during
        `bmi.__init__()`, or during `.initialize()`. If running BMI as type (1),
        config must be passed in the former, otherwise passed in the latter for (2).

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        """
        t_start = time.time()

        # Read BMI configuration file if provided.
        if config_path is not None:
            if not Path(config_path).is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path, 'r') as f:
                self.config_bmi = yaml.safe_load(f)
        
        if self.cfg_bmi is None:
            raise ValueError("No configuration file given. A config path" \
                             "must be passed at time of bmi init() or" \
                             "initialize() call.")

        # Load model configuration.
        if self.config_model is None:
            try:
                with open(self.config_bmi.get('config_model'), 'r') as f:
                    self.config_model = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration: {e}")

            self.sampler = import_data_sampler(self.config_model['data_sampler'])(self.config_model)


        # Load static vars from BMI config into internal storage.
        for var_name, var_value in self.config_bmi.get('static_vars', {}).items():
            if var_name in self._static_var:
                self._static_var[var_name]['value'] = bmi_array(var_value)
            else:
                log.warning(f"Static variable '{var_name}' not recognized. Skipping.")

        # # Set simulation parameters.
        self.current_time = self.config_bmi.get('start_time', 0.0)
        # self._time_step_size = self.config_bmi.get('time_step_size', 86400)  # Default to 1 day in seconds.
        # self._end_time = self.config_bmi.get('end_time', np.finfo('d').max)\

        # Load a trained model.
        try:
            self._model = self._load_trained_model(self.config_model)
            # self._model = ModelHandler(self.config_model).to(self.config_model['device'])
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")

        # Forward model on all data if specified.
        if self.config_bmi.get('forward_init', False):
            predictions = self.run_forward()

            # Process and store predictions.
            self._process_predictions(predictions)

        # Track total BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI initialize [ctrl fn] took {time.time() - t_start} s | Total runtime: {self.bmi_process_time} s")

    def update(self) -> None:
        """(Control function) Advance model state by one time step."""
        t_start = time.time()
        self.current_time += self._time_step_size 
        
        # Forward model on individual timesteps if not initialized with forward_init.
        if not self.config_bmi.get('forward_init', False):
            predictions = self.run_forward()

            # Process and store predictions.
            self._process_predictions(predictions)

        # Track total BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI update [ctrl fn] took {time.time() - t_start} s | Total runtime: {self.bmi_process_time} s")
    
    def run_forward(self):
        """Forward model and save outputs to return on update call."""
        data_dict = self._format_inputs()

        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config_model['predict']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)
        
        batch_predictions = []
        # Forward through basins in batches.
        with torch.no_grad():
            for i in range(len(batch_start)):
                dataset_sample = self.sampler.get_validation_sample(
                    data_dict,
                    batch_start[i],
                    batch_end[i],
                )

                # Forward dPLHydro model
                self.prediction = self._model.forward(dataset_sample, eval=True)

                # For single hydrology model.
                model_name = self.config_model['dpl_model']['phy_model']['model'][0]
                prediction = {
                    key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()
                }
                batch_predictions.append(prediction)
        
        return self._batch_data(batch_predictions)

        # preds = torch.cat([d['flow_sim'] for d in batched_preds_list], dim=1)
        # preds = preds.numpy()

        # # Scale and check output
        # self.scale_output()

    def _process_predictions(self, predictions):
        """Process model predictions and store them in output variables."""
        for var_name, prediction in predictions.items():
            if var_name in self._output_vars:
                self._output_vars[var_name]['value'] = prediction.cpu().numpy()
            else:
                log.warning(f"Output variable '{var_name}' not recognized. Skipping.")

    def _format_inputs(self):
        """Format dynamic and static inputs for the model."""
        inputs = {}
        for var_name, var_info in self._dynamic_var.items():
            inputs[var_name] = var_info['value']
        for var_name, var_info in self._static_var.items():
            inputs[var_name] = var_info['value']
        return inputs
    
    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge list of batch data dictionaries into a single dictionary."""
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data
        
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e
        
    def update_frac(self, time_frac: float) -> None:
        """
        Update model by a fraction of a time step.
        
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        if self.verbose:
            print("Warning: This model is trained to make predictions on one day timesteps.")
        time_step = self.get_time_step()
        self._time_step_size = self._time_step_size * time_frac
        self.update()
        self._time_step_size = time_step

    def update_until(self, end_time: float) -> None:
        """(Control function) Update model until a particular time.

        Note: Models should be trained standalone with dPLHydro_PMI first before
        forward predictions with this BMI.

        Parameters
        ----------
        end_time : float
            Time to run model until.
        """
        t_start = time.time()

        n_steps = (end_time - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

        # Keep running total of BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI update_until [ctrl fn] took {time.time() - t_start} s | Total runtime: {self.bmi_process_time} s")

    def finalize(self) -> None:
        """(Control function) Finalize model."""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
        self._initialized = False
        if self.verbose:
            log.info("BMI model finalized.")

    def array_to_tensor(self) -> None:
        """
        Converts input values into Torch tensor object to be read by model. 
        """  
        raise NotImplementedError("array_to_tensor")
    
    def tensor_to_array(self) -> None:
        """
        Converts model output Torch tensor into date + gradient arrays to be
        passed out of BMI for backpropagation, loss, optimizer tuning.
        """  
        raise NotImplementedError("tensor_to_array")
    
    def get_tensor_slice(self):
        """
        Get tensor of input data for a single timestep.
        """
        # sample_dict = take_sample_test(self.bmi_config, self.dataset_dict)
        # self.input_tensor = torch.Tensor()
    
        raise NotImplementedError("get_tensor_slice")

    def get_var_type(self, var_name):
        """
        Data type of variable.

        Parameters
        ----------g
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_standard_name):
        """Get units of variable.

        Parameters
        ----------
        var_standard_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return self._var_units_map[var_standard_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]

    def get_var_grid(self, var_name):
        """Grid id for a variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Grid id.
        """
        # for grid_id, var_name_list in self._grids.items():
        #     if var_name in var_name_list:
        #         return grid_id
        raise NotImplementedError("get_var_grid")

    def get_grid_rank(self, grid_id: int):
        """Rank of grid.

        Parameters
        ----------
        grid_id
            Identifier of a grid.

        Returns
        -------
        int
            Rank of grid.
        """
        if grid_id == 0:
            return 1
        raise ValueError(f"Unsupported grid rank: {grid_id!s}. only support 0")

    def get_grid_size(self, grid_id):
        """Size of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        # return int(np.prod(self._model.shape))
        raise NotImplementedError("get_grid_size")

    def get_value_ptr(self, var_standard_name: str, model:str) -> np.ndarray:
        """Reference to values.

        Parameters
        ----------
        var_standard_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        if model == 'nn':
            if var_standard_name not in self._nn_values.keys():
                raise ValueError(f"No known variable in BMI model: {var_standard_name}")
            return self._nn_values[var_standard_name]

        elif model == 'pm':
            if var_standard_name not in self._pm_values.keys():
                raise ValueError(f"No known variable in BMI model: {var_standard_name}")
            return self._pm_values[var_standard_name]
        
        else:
            raise ValueError("Valid model type (nn or pm) must be specified.")

    def get_value(self, var_name, dest):
        """Copy of values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        array_like
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name).flatten()
        return dest

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.

        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value(self, var_name, values: np.ndarray, model:str):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        values : array_like
            Array of new values.
        """
        if not isinstance(values, (np.ndarray, list, tuple)):
            values = np.array([values])

        val = self.get_value_ptr(var_name, model=model)

        # val = values.reshape(val.shape)
        val[:] = values

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        # var_name = self._grids[grid_id][0]
        # shape[:] = self.get_value_ptr(var_name).shape
        # return shape
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        # spacing[:] = self._model.spacing
        # return spacing
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        # origin[:] = self._model.origin
        # return origin
        raise NotImplementedError("get_grid_origin")

    def get_grid_type(self, grid_id):
        """Type of grid."""
        # return self._grid_type[grid_id]
        raise NotImplementedError("get_grid_type")

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        return self._current_time

    def get_time_step(self):
        return self._time_step_size

    def get_time_units(self):
        return self._time_units

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")

    def initialize_config(self, config_path: str) -> Dict:
        """
        Check that config_path is valid path and convert config into a
        dictionary object.
        """
        config_path = Path(config_path).resolve()
        
        if not config_path:
            raise RuntimeError("No BMI configuration path provided.")
        elif not config_path.is_file():
            raise RuntimeError(f"BMI configuration not found at path {config_path}.")
        else:
            with config_path.open('r') as f:
                self.config = yaml.safe_load(f)
    

        # USE BELOW FOR HYDRA + OMEGACONF:
        # try:
        #     config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
        #         cfg, resolve=True
        #     )
        #     config = Config(**config_dict)
        # except ValidationError as e:
        #     log.exception(e)
        #     raise e
        # return config, config_dict

    # def init_var_dicts(self):
    #     """
    #     Create lookup tables for CSDMS variables and init variable arrays.
    #     """
    #     # Make lookup tables for variable name (Peckham et al.).
    #     self._var_name_map_long_first = {
    #         long_name:self._var_name_units_map[long_name][0] for \
    #         long_name in self._var_name_units_map.keys()
    #         }
    #     self._var_name_map_short_first = {
    #         self._var_name_units_map[long_name][0]:long_name for \
    #         long_name in self._var_name_units_map.keys()}
    #     self._var_units_map = {
    #         long_name:self._var_name_units_map[long_name][1] for \
    #         long_name in self._var_name_units_map.keys()
    #     }

    #     # Initialize inputs and outputs.
    #     for var in self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn']:
    #         standard_name = self._var_name_map_short_first[var]
    #         self._nn_values[standard_name] = []
    #         # setattr(self, var, 0)

    #     for var in self.config['observations']['var_t_hydro_model'] + self.config['observations']['var_c_hydro_model']:
    #         standard_name = self._var_name_map_short_first[var]
    #         self._pm_values[standard_name] = []
    #         # setattr(self, var, 0)

    # def scale_output(self) -> None:
    #     """
    #     Scale and return more meaningful output from wrapped model.
    #     """
    #     models = self.config['hydro_models'][0]

    #     # TODO: still have to finish finding and undoing scaling applied before
    #     # model run. (See some checks used in bmi_lstm.py.)

    #     # Strip unnecessary time and variable dims. This gives 1D array of flow
    #     # at each basin.
    #     # TODO: setup properly for multiple models later.
    #     self.streamflow_cms = self.preds[models]['flow_sim'].squeeze()

    # def _get_batch_sample(self, config: Dict, dataset_dictionary: Dict[str, torch.Tensor], 
    #                     i_s: int, i_e: int) -> Dict[str, torch.Tensor]:
    #     """
    #     Take sample of data for testing batch.
    #     """
    #     dataset_sample = {}
    #     for key, value in dataset_dictionary.items():
    #         if value.ndim == 3:
    #             # TODO: I don't think we actually need this.
    #             # Remove the warmup period for all except airtemp_memory and hydro inputs.
    #             if key in ['airT_mem_temp_model', 'x_phy', 'inputs_nn_scaled']:
    #                 warm_up = 0
    #             else:
    #                 warm_up = config['warm_up']
    #             dataset_sample[key] = value[warm_up:, i_s:i_e, :].to(config['device'])
    #         elif value.ndim == 2:
    #             dataset_sample[key] = value[i_s:i_e, :].to(config['device'])
    #         else:
    #             raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")
    #     return dataset_sample

    # def _values_to_dict(self) -> None:
    #     """
    #     Take CSDMS Standard Name-mapped forcings + attributes and construct data
    #     dictionary for NN and physics model.
    #     """
    #     # n_basins = self.config['batch_basins']
    #     n_basins = 671
    #     rho = self.config['rho']

    #     # Initialize dict arrays.
    #     # NOTE: used to have rho+1 here but this is no longer necessary?
    #     x_nn = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_t_nn'])))
    #     c_nn = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_c_nn'])))
    #     x_phy = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_t_hydro_model'])))
    #     c_hydro_model = np.zeros((n_basins, len(self.config['observations']['var_c_hydro_model'])))

    #     for i, var in enumerate(self.config['observations']['var_t_nn']):
    #         standard_name = self._var_name_map_short_first[var]
    #         # NOTE: Using _values is a bit hacky. Should use get_values I think.    
    #         x_nn[:, :, i] = np.array([self._nn_values[standard_name]])
        
    #     for i, var in enumerate(self.config['observations']['var_c_nn']):
    #         standard_name = self._var_name_map_short_first[var]
    #         c_nn[:, :, i] = np.array([self._nn_values[standard_name]])

    #     for i, var in enumerate(self.config['observations']['var_t_hydro_model']):
    #         standard_name = self._var_name_map_short_first[var]
    #         x_phy[:, :, i] = np.array([self._pm_values[standard_name]])

    #     for i, var in enumerate(self.config['observations']['var_c_hydro_model']):
    #         standard_name = self._var_name_map_short_first[var]
    #         c_hydro_model[:, i] = np.array([self._pm_values[standard_name]])
        
    #     self.dataset_dict = {
    #         'inputs_nn_scaled': np.concatenate((x_nn, c_nn), axis=2), #[np.newaxis,:,:],
    #         'x_phy': x_phy, #[np.newaxis,:,:],
    #         'c_hydro_model': c_hydro_model
    #     }
    #     print(self.dataset_dict['inputs_nn_scaled'].shape)

    #     # Convert to torch tensors:
    #     for key in self.dataset_dict.keys():
    #         if type(self.dataset_dict[key]) == np.ndarray:
    #             self.dataset_dict[key] = torch.from_numpy(self.dataset_dict[key]).float() #.to(self.config['device'])

    # def get_csdms_name(self, var_name):
    #     """
    #     Get CSDMS Standard Name from variable name.
    #     """
    #     return self._var_name_map_long_first[var_name]
    