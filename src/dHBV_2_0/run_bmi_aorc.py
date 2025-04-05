import os
from pathlib import Path

import numpy as np
import pandas as pd

from bmi_dhbv import deltaModelBmi as Bmi

# os.chdir(os.path.expanduser('../dHBV_2_0/'))


### Select a basin from the sample data ###
basin_id = "cat-88306"
bmi_config_path = f'C:/Users/LeoLo/Desktop/noaa_owp/dHBV_2_0/bmi_config_files/bmi_config_{basin_id}_5yr.yaml'
### ----------------------------------- ###


# Load the USGS data 
# REPLACE THIS PATH WITH YOUR LOCAL FILE PATH:
forc_path = f'C:/Users/LeoLo/Desktop/noaa_owp/dHBV_2_0/data/aorc/juniata_river_basin/forcings_5yr_{basin_id}.npy'
attr_path = f'C:/Users/LeoLo/Desktop/noaa_owp/dHBV_2_0/data/aorc/juniata_river_basin/attributes_5yr_{basin_id}.npy'
# obs_path = f'/Users/LeoLo/Desktop/noaa_owp/dHBV_2_0/data/aorc/juniata_river_basin/obs_5yr_{basin_id}.npy'

forc = np.load(forc_path)
attr = np.load(attr_path)
# obs = np.load(obs_path)

# Create an instance of the dHBV 2.0 through BMI
model = Bmi(config_path=bmi_config_path)

### BMI initialization ###
model.initialize()

# streamflow_pred = np.zeros(forc.shape[0])

# for i in range(0, forc.shape[0]):
#     # Extract forcing/attribute data for the current time step
#     prcp = forc[i, :0, 0]
#     temp = forc[i, :0, 1]
#     pet = forc[i, :0, 2]

#     # # Check if any of the inputs are NaN
#     # if np.isnan([prcp, temp, pet]).any():
#     #     if model.verbose > 0:
#     #         print(f"Skipping timestep {i} due to NaN values in inputs.")
#     #     continue

#     model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', prcp)
#     model.set_value('land_surface_air__temperature', temp)
#     model.set_value('land_surface_water__potential_evaporation_volume_flux', pet)

#     ### BMI update ###
#     model.update()

#     # Retrieve and scale the runoff output
#     dest_array = np.zeros(1)
#     model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    
#     # streamflow_pred[i] = dest_array[0] * 1000  # Convert to mm/hr
#     streamflow_pred[i] = dest_array[0]


#  ### BMI finalization ###
# model.finalize()


# # # Calculate NSE for the model predictions
# # obs = obs.dropna()
# # sim = streamflow_pred.dropna()

# # denom = ((obs - obs.mean()) ** 2).sum()
# # num = ((sim - obs) ** 2).sum()
# # nse = 1 - num / denom
# # print(f"NSE: {nse:.2f}")
