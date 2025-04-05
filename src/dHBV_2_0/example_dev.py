# Need these for BMI
from bmipy import Bmi
import time
# Import data_tools
# Basic utilities
import numpy as np
import pandas as pd
from pathlib import Path 
# Configuration file functionality
import yaml
# LSTM here is based on PyTorch
import torch

# Here is the LSTM model we want to run
# import nextgen_cuda_lstm
import lstm.nextgen_cuda_lstm as nextgen_cuda_lstm   # (SDP)
import os

# These are not used (SDP)
### from torch import nn
### import sys

#------------------------------------------------------------------------
USE_PATH = True  # (SDP)
if not(USE_PATH):
    import os

class bmi_LSTM(Bmi):

    def __init__(self):
        """Create a Bmi LSTM model that is ready for initialization."""
        super(bmi_LSTM, self).__init__()
        self._name = "LSTM for Next Generation NWM"
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._var_grid_type = "scalar"
        self._start_time = 0
        self._end_time = np.finfo("d").max
        self._time_units = "hour"  # (SDP)
        self._time_step_size = 1.0 # (SDP)

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    # Note: not currently in use
    _att_map = {
        'model_name':         'LSTM for Next Generation NWM',
        'version':            '1.0',
        'author_name':        'Jonathan Martin Frame' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = [
        'land_surface_radiation~incoming~longwave__energy_flux',
        'land_surface_air__pressure',
        'atmosphere_air_water~vapor__relative_saturation',
        'atmosphere_water__liquid_equivalent_precipitation_rate',  ### SDP, 08/30/22
        ##### 'atmosphere_water__time_integral_of_precipitation_mass_flux',  #### SDP
        'land_surface_radiation~incoming~shortwave__energy_flux',
        'land_surface_air__temperature',
        'land_surface_wind__x_component_of_velocity',
        'land_surface_wind__y_component_of_velocity']
    # (Next line didn't fix ngen pointer error)
    # _input_var_names = []

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['land_surface_water__runoff_depth', 
                         'land_surface_water__runoff_volume_flux']
    # (Next line didn't fix ngen pointer error)
    # _output_var_names = ['land_surface_water__runoff_volume_flux']
                         
    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    # This is going to get long, 
    #     since the input variable names could come from any forcing...
    #------------------------------------------------------
    #_var_name_map_long_first = {
    _var_name_units_map = {
        'land_surface_water__runoff_volume_flux':['streamflow_cms','m3 s-1'],
        'land_surface_water__runoff_depth':['streamflow_m','m'],
        #--------------   Dynamic inputs --------------------------------
        #NJF Let the model assume equivalence of `kg m-2` == `mm h-1` since we can't convert
        #mass flux automatically from the ngen framework
        'atmosphere_water__liquid_equivalent_precipitation_rate':['APCP_surface','mm h-1'],
        'land_surface_radiation~incoming~longwave__energy_flux':['DLWRF_surface','W m-2'],
        'land_surface_radiation~incoming~shortwave__energy_flux':['DSWRF_surface','W m-2'],
        'atmosphere_air_water~vapor__relative_saturation':['SPFH_2maboveground','kg kg-1'],
        'land_surface_air__pressure':['PRES_surface','Pa'],
        'land_surface_air__temperature':['TMP_2maboveground','degK'],
        'land_surface_wind__x_component_of_velocity':['UGRD_10maboveground','m s-1'],
        'land_surface_wind__y_component_of_velocity':['VGRD_10maboveground','m s-1'],
        #--------------   STATIC Attributes -----------------------------
        'basin__mean_of_elevation':['elev_mean','m'],
        'basin__mean_of_slope':['slope_mean','m km-1'],
        }

    _static_attributes_list = ['elev_mean','slope_mean']
    
    def __getattribute__(self, item):
        """
        Customize instance attribute access.

        For those items that correspond to BMI input or output variables (which should be in numpy arrays) and have
        values that are just a single-element array, deviate from the standard behavior and return the single array
        element.  Fall back to the default behavior in any other case.

        This supports having a BMI variable be backed by a numpy array, while also allowing the attribute to be used as
        just a scalar, as it is in many places for this type.

        Parameters
        ----------
        item
            The name of the attribute item to get.

        Returns
        -------
        The value of the named item.
        """
        # Have these work explicitly (or else loops)
        if item == '_input_var_names' or item == '_output_var_names':
            return super(bmi_LSTM, self).__getattribute__(item)

        # By default, for things other than BMI variables, use normal behavior
        if item not in super(bmi_LSTM, self).__getattribute__('_input_var_names') and item not in super(bmi_LSTM, self).__getattribute__('_output_var_names'):
            return super(bmi_LSTM, self).__getattribute__(item)

        # Return the single scalar value from any ndarray of size 1
        value = super(bmi_LSTM, self).__getattribute__(item)
        if isinstance(value, np.ndarray) and value.size == 1:
            return value[0]
        else:
            return value

    def __setattr__(self, key, value):
        """
        Customized instance attribute mutator functionality.

        For those attribute with keys indicating they are a BMI input or output variable (which should be in numpy
        arrays), wrap any scalar ``value`` as a one-element numpy array and use that in a nested call to the superclass
        implementation of this function.  In any other cases, just pass the given ``key`` and ``value`` to a nested
        call.

        This supports automatically having a BMI variable be backed by a numpy array, even if it is initialized using a
        scalar, while otherwise maintaining standard behavior.

        Parameters
        ----------
        key
        value

        Returns
        -------

        """
        # Have these work explicitly (or else loops)
        if key == '_input_var_names' or key == '_output_var_names':
            super(bmi_LSTM, self).__setattr__(key, value)

        # Pass thru if value is already an array
        if isinstance(value, np.ndarray):
            super(bmi_LSTM, self).__setattr__(key, value)
        # Override to put scalars into ndarray for BMI input/output variables
        elif key in self._input_var_names or key in self._output_var_names:
            super(bmi_LSTM, self).__setattr__(key, np.array([value]))
        # By default, use normal behavior
        else:
            super(bmi_LSTM, self).__setattr__(key, value)

    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file=None ):
        #NJF ensure this is a Path type so the follow open works as expected
        #When used with NGen, the bmi_cfg_file is just a string...

        bmi_cfg_file = Path(bmi_cfg_file)
        # ----- Create some lookup tables from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for \
                                         long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for \
                                          long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for \
                                          long_name in self._var_name_units_map.keys()}
        
        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        for var_name in list(self._var_name_units_map.keys()):
            # ---------- All the variables are single values ------------------#
            # ---------- so just set to zero for now.        ------------------#
            self._values[var_name] = 0.0
            setattr( self, var_name, 0.0 )
        
        # -------------- Read in the BMI configuration -------------------------#
        # This will direct all the next moves.
        if bmi_cfg_file is not None:
            #----------------------------------------------------------
            # Note: bmi_cfg_file should have type 'str', vs. being a
            #       Path object. So apply Path in initialize(). (SDP)
            #----------------------------------------------------------
            ### with bmi_cfg_file.open('r') as fp:    # (orig)
            with open(bmi_cfg_file,'r') as fp:    # (SDP)
                cfg = yaml.safe_load(fp)
            self.cfg_bmi = self._parse_config(cfg)
        else:
            print("Error: No configuration provided, nothing to do...")
        
        # Number of inidividual ensemble members
        self.N_ENS = len(self.cfg_bmi['train_cfg_file'])

        # Note: these need to be initialized here as scale_output() called in update()
        self.lstm_output = {i_ens:0.0 for i_ens in range(self.N_ENS)}
        self.streamflow_cms = {i_ens:0.0 for i_ens in range(self.N_ENS)}
        self.streamflow_fms = {i_ens:0.0 for i_ens in range(self.N_ENS)}
        self.surface_runoff_mm = {i_ens:0.0 for i_ens in range(self.N_ENS)}

        # Gather verbosity lvl from bmi-config for stdout printing, etc.
        self.verbose = self.cfg_bmi['verbose']
        if self.verbose == 0:
            print("Will not print anything except errors because verbosity set to", self.verbose)
        if self.verbose == 1:
            print("Will print warnings and errors because verbosity set to", self.verbose)
        if self.verbose > 1:
            print("Will print warnings, errors and random information because verbosity set to", self.verbose)
        print("self.verbose", self.verbose)

        # ------------- Load in the configuration file for the specific LSTM --#
        # This will include all the details about how the model was trained
        # Inputs, outputs, hyper-parameters, scalers, weights, etc. etc.
        self.get_training_configurations()
        self.get_scaler_values()
        
        # ------------- Initialize an ENSEMBLE OF LSTM models ------------------------------#
        self.lstm = {}
        self.h_t = {}
        self.c_t = {}

        for i_ens in range(self.N_ENS):
            self.lstm[i_ens] = nextgen_cuda_lstm.Nextgen_CudaLSTM(input_size=self.input_size[i_ens],
                                                        hidden_layer_size=self.hidden_layer_size[i_ens],
                                                        output_size=self.output_size[i_ens],
                                                        batch_size=1, 
                                                        seq_length=1)

            # ------------ Load in the trained weights ----------------------------#
            # Save the default model weights. We need to make sure we have the same keys.
            default_state_dict = self.lstm[i_ens].state_dict()

            # Trained model weights from Neuralhydrology.
            if (USE_PATH):  # (SDP)

                if self.verbose > 0:
                    print(self.cfg_train[i_ens]['run_dir'])

                trained_model_file = self.cfg_train[i_ens]['run_dir'] / 'model_epoch{}.pt'.format(str(self.cfg_train[i_ens]['epochs']).zfill(3))
            else:
                str1 = self.cfg_train[i_ens]['run_dir'] + '/' + 'model_epoch{}.pt'
                trained_model_file = str1.format(str(self.cfg_train[i_ens]['epochs']).zfill(3))

            trained_state_dict = torch.load(trained_model_file, map_location=torch.device('cpu'))

            # Changing the name of the head weights, since different in NH
            trained_state_dict['head.weight'] = trained_state_dict.pop('head.net.0.weight')
            trained_state_dict['head.bias'] = trained_state_dict.pop('head.net.0.bias')
            trained_state_dict = {x:trained_state_dict[x] for x in default_state_dict.keys()}

            # Load in the trained weights.
            self.lstm[i_ens].load_state_dict(trained_state_dict)

            # ------------- Initialize the values for the input to the LSTM  -----#
            # jmframe(jan 27): If we assume all models have the same inputs, this only needs to happen once.
            if i_ens == 0:
                self.set_static_attributes()
                self.initialize_forcings()
            
            if self.cfg_bmi['initial_state'] == 'zero':
                self.h_t[i_ens] = torch.zeros(1, self.batch_size, self.hidden_layer_size[i_ens]).float()
                self.c_t[i_ens] = torch.zeros(1, self.batch_size, self.hidden_layer_size[i_ens]).float()

        # ------------- Start a simulation time  -----------------------------#
        # jmframe: Since the simulation time here doesn't really matter. 
        #          Just use seconds and set the time to zero
        #          But add some logic maybe, so to be able to start at some time
        self.t = self._start_time

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s    
        self.output_factor_cms =  (1/1000) * (self.cfg_bmi['area_sqkm'] * 1000*1000) * (1/3600)

    #------------------------------------------------------------ 
    def update(self):
        with torch.no_grad():

            self.create_scaled_input_tensor()

            for i_ens in range(self.N_ENS):

                self.lstm_output[i_ens], self.h_t[i_ens], self.c_t[i_ens] = self.lstm[i_ens].forward(self.input_tensor[i_ens], self.h_t[i_ens], self.c_t[i_ens])
                
                self.scale_output(i_ens)

            self.ensemble_output()
                
            #self.t += self._time_step_size
            self.t += self.get_time_step()

    #------------------------------------------------------------ 
    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        if self.verbose > 0:
            print("Warning: This version of the LSTM is designed to make predictions on one hour timesteps.")
        time_step = self.get_time_step()
        self._time_step_size = time_frac * self._time_step_size
        self.update()
        self._time_step_size = time_step

    #------------------------------------------------------------ 
    def update_until(self, then):
        """Update model until a particular time.
        Parameters
        ----------
        then : float
            Time to run model until.
        """
        if self.verbose > 0:
            print("then", then)
            print("self.get_current_time()", self.get_current_time())
            print("self.get_time_step()", self.get_time_step())
        n_steps = (then - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

    #------------------------------------------------------------    
    def finalize( self ):
        """Finalize model."""
        self._model = None
    
    #------------------------------------------------------------
    #------------------------------------------------------------
    # LSTM: SETUP Functions
    #------------------------------------------------------------
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def get_training_configurations(self):

        self.cfg_train = {}
        self.input_size = {}
        self.hidden_layer_size = {}
        self.output_size = {}
        self.all_lstm_inputs = {}
        self.train_data_scaler = {}

        for i_ens in range(self.N_ENS):

            if self.cfg_bmi['train_cfg_file'][i_ens] is not None:
                if (USE_PATH):  # (SDP)
                    with self.cfg_bmi['train_cfg_file'][i_ens].open('r') as fp:
                        cfg = yaml.safe_load(fp)
                    self.cfg_train[i_ens] = self._parse_config(cfg)
                else:
                    with open(self.cfg_bmi['train_cfg_file'][i_ens],'r') as fp:  # (SDP)
                        cfg = yaml.safe_load(fp)
                    self.cfg_train[i_ens] = self._parse_config(cfg)

            # Including a list of the model input names.
            if self.verbose > 0:
                print("Setting the LSTM arcitecture based on the last run ensemble configuration")
                print(self.cfg_train[i_ens])
            # Collect the LSTM model architecture details from the configuration file 
            self.input_size[i_ens]        = len(self.cfg_train[i_ens]['dynamic_inputs']) + len(self.cfg_train[i_ens]['static_attributes'])
            self.hidden_layer_size[i_ens] = self.cfg_train[i_ens]['hidden_size']
            self.output_size[i_ens]       = len(self.cfg_train[i_ens]['target_variables']) 

            self.all_lstm_inputs[i_ens] = []
            self.all_lstm_inputs[i_ens].extend(self.cfg_train[i_ens]['dynamic_inputs'])
            self.all_lstm_inputs[i_ens].extend(self.cfg_train[i_ens]['static_attributes'])

            # WARNING: This implimentation of the LSTM can only handle a batch size of 1
            # No need to included different batch sizes
            self.batch_size        = 1 

            scaler_file = os.path.join(self.cfg_train[i_ens]['run_dir'], 'train_data', 'train_data_scaler.yml')

            with open(scaler_file, 'r') as f:
                scaler_data = yaml.safe_load(f)

            self.train_data_scaler[i_ens] = scaler_data

            # Scaler data from the training set. This is used to normalize the data (input and output).
            if self.verbose > 1:
                print(f"ensemble member {i_ens}")
                print(self.cfg_train[i_ens]['run_dir'])
                print(self.cfg_train[i_ens]['run_dir'])

    #------------------------------------------------------------
    def get_scaler_values(self):

        """Mean and standard deviation for the inputs and LSTM outputs"""

        self.input_mean = {}
        self.input_std = {}
        self.out_mean = {}
        self.out_std = {}

        for i_ens in range(self.N_ENS):

            self.out_mean[i_ens] = self.train_data_scaler[i_ens]['xarray_feature_center']['data_vars'][self.cfg_train[i_ens]['target_variables'][0]]['data']
            self.out_std[i_ens] = self.train_data_scaler[i_ens]['xarray_feature_scale']['data_vars'][self.cfg_train[i_ens]['target_variables'][0]]['data']

            self.input_mean[i_ens] = []
            self.input_mean[i_ens].extend([self.train_data_scaler[i_ens]['xarray_feature_center']['data_vars'][x]['data'] for x in self.cfg_train[i_ens]['dynamic_inputs']])
            self.input_mean[i_ens].extend([self.train_data_scaler[i_ens]['attribute_means'][x] for x in self.cfg_train[i_ens]['static_attributes']])
            self.input_mean[i_ens] = np.array(self.input_mean[i_ens])

            self.input_std[i_ens] = []
            self.input_std[i_ens].extend([self.train_data_scaler[i_ens]['xarray_feature_scale']['data_vars'][x]['data'] for x in self.cfg_train[i_ens]['dynamic_inputs']])
            self.input_std[i_ens].extend([self.train_data_scaler[i_ens]['attribute_stds'][x] for x in self.cfg_train[i_ens]['static_attributes']])
            self.input_std[i_ens] = np.array(self.input_std[i_ens])
            if self.verbose > 1:
                print('###########################')
                print('input_mean')
                print(self.input_mean[i_ens])
                print('input_std')
                print(self.input_std[i_ens])
                print('out_mean')
                print(self.out_mean[i_ens])
                print('out_std')
                print(self.out_std[i_ens])

    #------------------------------------------------------------ 
    def create_scaled_input_tensor(self):

        self.input_list = {}
        self.input_array = {}
        self.input_array_scaled = {}
        self.input_tensor = {}

        #------------------------------------------------------------
        # Note:  A BMI-enabled model should not use long var names
        #        internally (i.e. saved into self); it should just
        #        use convenient short names.  For the BMI functions
        #        that require a long var name, it should be mapped
        #        to the model's short name before taking action.
        #------------------------------------------------------------        
        # TODO: Choose to store values in dictionary or not.

        #--------------------------------------------------------------        
        # Note:  The code in this block is more verbose, but makes it
        #        much easier to test and debug and helped find a bug
        #        in the lines above (long vs. short names.) 
        #--------------------------------------------------------------
        for i_ens in range(self.N_ENS):
            if self.verbose > 1:
                print('Creating scaled input tensor...')
            n_inputs = len(self.all_lstm_inputs[i_ens])
            self.input_list[i_ens] = []  #############
            DEBUG = False
            for k in range(n_inputs):
                short_name = self.all_lstm_inputs[i_ens][k]
                long_name  = self._var_name_map_short_first[ short_name ]
                # vals = self.get_value( self, long_name )
                vals = getattr( self, short_name )  ####################

                self.input_list[i_ens].append( vals )
                if self.verbose > 1:         
                    print('  short_name =', short_name )
                    print('  long_name  =', long_name )
                    array = getattr( self, short_name )
                    ## array = self.get_value( long_name )  
                    print('  type       =', type(vals) )
                    print('  vals       =', vals )

            #--------------------------------------------------------
            # W/o setting dtype here, it was "object_", and crashed
            #--------------------------------------------------------
            ## self.input_array = np.array( self.input_list )
            self.input_array[i_ens] = np.array( self.input_list[i_ens], dtype='float64' )  # SDP
            if self.verbose > 0:
                print('Normalizing the tensor...')
                print('  input_mean =', self.input_mean[i_ens] )
                print('  input_std  =', self.input_std[i_ens]  )
                print()
            # Center and scale the input values for use in torch
            self.input_array_scaled[i_ens] = (self.input_array[i_ens] - self.input_mean[i_ens]) / self.input_std[i_ens]
            if self.verbose > 1:
                print('### input_list =', self.input_list[i_ens])
                print('### input_array =', self.input_array[i_ens])
                print('### dtype(input_array) =', self.input_array[i_ens].dtype )
                print('### type(input_array_scaled) =', type(self.input_array_scaled[i_ens]))
                print('### dtype(input_array_scaled) =', self.input_array_scaled.dtype[i_ens] )
                print()
            self.input_tensor[i_ens] = torch.tensor(self.input_array_scaled[i_ens])

    #------------------------------------------------------------ 
    def scale_output(self, i_ens):

        if self.verbose > 1:
            print("model output:", self.lstm_output[i_ens][0,0,0].numpy().tolist())

        if self.cfg_train[i_ens]['target_variables'][0] in ['qobs_mm_per_hour', 'QObs(mm/hr)', 'QObs(mm/h)']:
            self.surface_runoff_mm[i_ens] = (self.lstm_output[i_ens][0,0,0].numpy().tolist() * self.out_std[i_ens] + self.out_mean[i_ens])

        elif self.cfg_train[i_ens]['target_variables'][0] in ['QObs(mm/d)']:
            self.surface_runoff_mm[i_ens] = (self.lstm_output[i_ens][0,0,0].numpy().tolist() * self.out_std[i_ens] + self.out_mean[i_ens]) * (1/24)
            
        self.surface_runoff_mm[i_ens] = max(self.surface_runoff_mm[i_ens],0.0)

        setattr(self, 'land_surface_water__runoff_depth', self.surface_runoff_mm[i_ens]/1000.0)
        self.streamflow_cms[i_ens] = self.surface_runoff_mm[i_ens] * self.output_factor_cms

        if self.verbose > 1:
            print("streamflow:", self.streamflow_cms[i_ens])


    #-------------------------------------------------------------------
    def ensemble_output(self):
        # Calculate mean surface runoff (mm) across ensemble members
        ens_mean_surface_runoff_mm = np.mean([self.surface_runoff_mm[i_ens] for i_ens in range(self.N_ENS)])

        # Set the land_surface_water__runoff_depth attribute (convert mm to m)
        setattr(self, 'land_surface_water__runoff_depth', ens_mean_surface_runoff_mm / 1000.0)

        # Calculate mean streamflow (cms) across ensemble members
        ens_mean_streamflow_cms = np.mean([self.streamflow_cms[i_ens] for i_ens in range(self.N_ENS)])

        # Set the land_surface_water__runoff_volume_flux attribute
        setattr(self, 'land_surface_water__runoff_volume_flux', ens_mean_streamflow_cms)

    #---------------------------------------------------------------------------- 
    def set_static_attributes(self):
        """ Get the static attributes from the configuration file
        """
        i_ens = 0
        for attribute in self._static_attributes_list:
            if attribute in self.cfg_train[i_ens]['static_attributes']:
                #------------------------------------------------------------
                # Note:  A BMI-enabled model should not use long var names
                #        internally (i.e. saved into self); it should just
                #        use convenient short names.  For the BMI functions
                #        that require a long var name, it should be mapped
                #        to the model's short name before taking action.
                #------------------------------------------------------------
                setattr(self, attribute, self.cfg_bmi[attribute])  # SDP

    #---------------------------------------------------------------------------- 
    def initialize_forcings(self):

        if self.verbose > 0:
            print('Initializing all forcings to 0...')
        i_ens = 0

        for forcing_name in self.cfg_train[i_ens]['dynamic_inputs']:
            if self.verbose > 1:
                print('  forcing_name =', forcing_name)
            #------------------------------------------------------------
            # Note:  A BMI-enabled model should not use long var names
            #        internally (i.e. saved into self); it should just
            #        use convenient short names.  For the BMI functions
            #        that require a long var name, it should be mapped
            #        to the model's short name before taking action.
            #------------------------------------------------------------
            setattr(self, forcing_name, 0)

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    # Note: not currently using _att_map{}
    # def get_attribute(self, att_name):
    
    #     try:
    #         return self._att_map[ att_name.lower() ]
    #     except:
    #         print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        #return self.get_attribute( 'model_name' )
        return self._name

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        """
        Copy values for the named variable into the provided destination array.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : np.ndarray
            A numpy array into which to copy the variable values.
        Returns
        -------
        np.ndarray
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name)

        if self.verbose > 1:
            print("self.verbose", self.verbose)
            print("get value dest", dest)

        return dest

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> np.ndarray:
        """
        Get reference to values.

        Get the backing reference - i.e., the backing numpy array - for the given variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        np.ndarray
            Value array.
        """
        # We actually need this function to return the backing array, so bypass override of __getattribute__ (that
        # extracts scalar) and use the base implementation
        return super(bmi_LSTM, self).__getattribute__(var_name)

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """

        #JG MW 03.01.23 - otherwise Bmi_py_Adaptor.hpp `get_analogous_cxx_type` fails
        return self.get_value_ptr(long_var_name).dtype.name
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # Note: all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id  

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
        # JG get_value_ptr is already an np.array
        return self.get_value_ptr(name).itemsize  

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        # Note: all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    #-------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return self._start_time

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return self._end_time


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.t

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self._time_step_size
        
    #-------------------------------------------------------------------
    def get_time_units( self ):

        # Note: get_attribute() is not a BMI v2 method
        return self._time_units

    #-------------------------------------------------------------------
    def set_value(self, var_name: str, values:np.ndarray):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """
    
        internal_array = self.get_value_ptr(var_name)
        internal_array[:] = values

        short_name = self._var_name_map_long_first[ var_name ]
        
        if (internal_array.ndim > 0):
            setattr( self, short_name, internal_array[0])
        else:
            setattr( self, short_name, internal_array )
            
        try: 
            #NJF From NGEN, `internal_array` is a singleton array
            setattr( self, var_name, internal_array[0] )
        
            # jmframe: this next line is basically a duplicate. 
            # I guess we should stick with the attribute names instead of a dictionary approach. 
            self._values[var_name] = internal_array[0]
        # JLG 03242022: this isn't really an "error" block as standalone considers value as scalar?
        except TypeError:
            setattr( self, var_name, internal_array )
        
            # jmframe: this next line is basically a duplicate. 
            # I guess we should stick with the attribute names instead of a dictionary approach. 
            self._values[var_name] = internal_array

    #------------------------------------------------------------ 
    def set_value_at_indices(self, var_name: str, inds: np.ndarray, src: np.ndarray):
        """
        Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        inds : np.ndarray
            Array of corresponding indices into which to copy the values within ``src``.
        src : np.ndarray
            Array of new values.
        """
        internal_array = self.get_value_ptr(var_name)
        for i in range(inds.shape[0]):
            internal_array[inds[i]] = src[i]

    #------------------------------------------------------------ 
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
        return self.get_var_itemsize(var_name)*len(self.get_value_ptr(var_name))

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name: str, dest:np.ndarray, indices:np.ndarray) -> np.ndarray:
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
        #NJF This must copy into dest!!!
        #Convert to np.array in case of singleton/non numpy type, then flatten
        original: np.ndarray = self.get_value_ptr(var_name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = original[value_index]
        return dest
 
    #   Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------ 
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
 
         # 0 is the only id we have
        if grid_id == 0: 
            return 1

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
       
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id=0):

        # 0 is the only id we have        
        if grid_id == 0:
            return 'scalar'

    #------------------------------------------------------------ 
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    #------------------------------------------------------------ 
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    #------------------------------------------------------------ 
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 


    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #-- Random utility functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 

    def _parse_config(self, cfg):
        for key, val in cfg.items():
            # Handle 'train_cfg_file' specifically to ensure it is always a list
            if key == 'train_cfg_file':
                if val is not None and val != "None":
                    if isinstance(val, list):
                        cfg[key] = [Path(element) if USE_PATH else element for element in val]
                    else:
                        cfg[key] = [Path(val)] if USE_PATH else [val]
                else:
                    cfg[key] = []

            # Convert all path strings to PosixPath objects for other keys
            elif any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if val is not None and val != "None":
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            if USE_PATH:
                                temp_list.append(Path(element))
                            else:
                                temp_list.append(element)  # (SDP)
                        cfg[key] = temp_list
                    else:
                        if USE_PATH:
                            cfg[key] = Path(val)
                        else:
                            cfg[key] = val  # (SDP)
                else:
                    cfg[key] = None

            # Convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Add more config parsing if necessary
        return cfg
    


def coerce_config(cfg: dict[str, typing.Any]):
    for key, val in cfg.items():
        # Handle 'train_cfg_file' specifically to ensure it is always a list
        if key == "train_cfg_file":
            if val is not None and val != "None":
                if isinstance(val, list):
                    cfg[key] = [Path(element) for element in val]
                else:
                    cfg[key] = [Path(val)]
            else:
                cfg[key] = []

        # Convert all path strings to PosixPath objects for other keys
        elif any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
            if val is not None and val != "None":
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # Convert Dates to pandas Datetime indexs
        elif key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")



def load_static_attributes(cfg: dict[str, Any], var_dict: dict[str, dict[str, Union[NDArray, str]]]) -> None:
    for external_name in self.names():
        internal_name = crosswalk_to_interal(external_name)
        value = cfg[internal_name]
        state.set_value(external_name, bmi_array([value]))