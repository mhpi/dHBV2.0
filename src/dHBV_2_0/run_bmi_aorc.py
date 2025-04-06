import numpy as np
from dHBV_2_0.src.dHBV_2_0.bmi import DeltaModelBmi as Bmi


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

streamflow_pred = np.zeros(forc.shape[0])
nan_idx = []

# 1) Compile forcing data within BMI to do batch run.
for i in range(0, forc.shape[0]):
    # Extract forcing/attribute data for the current time step
    prcp = forc[i, :, 0]
    temp = forc[i, :, 1]
    pet = forc[i, :, 2]

    ## Check if any of the inputs are NaN
    if np.isnan([prcp, temp, pet]).any():
        # if model.verbose > 0:
        print(f"Skipping timestep {i} due to NaN values in inputs.")
        nan_idx.append(i)
        continue

    model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', prcp)
    model.set_value('land_surface_air__temperature', temp)
    model.set_value('land_surface_water__potential_evaporation_volume_flux', pet)


### BMI initialization ###
model.initialize()

# 2) DO pseudo model forward and return pre-predicted values at each timestep
for i in range(0, forc.shape[0]):
    if i in nan_idx:
        # Skip the update for this timestep
        continue

    ### BMI update ###
    model.update()

    # Retrieve and scale the runoff output
    dest_array = np.zeros(1)
    model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    
    streamflow_pred[i] = dest_array[0]  # Convert to mm/day -> mm/hr

 ### BMI finalization ###
model.finalize()

print("\n=/= -- Streamflow prediction completed -- =/=")
print(f"    Basin ID:              {basin_id}")
print(f"    Total Process Time:    {model.bmi_process_time:.4f} seconds")
print(f"    Mean streamflow:       {streamflow_pred.mean():.4f} mm/day")
print(f"    Max streamflow:        {streamflow_pred.max():.4f} mm/day")
print(f"    Min streamflow:        {streamflow_pred.min():.4f} mm/day")
print("=/= ------------------------------------ =/=")


# # Calculate NSE for the model predictions
# obs = obs.dropna()
# sim = streamflow_pred.dropna()

# denom = ((obs - obs.mean()) ** 2).sum()
# num = ((sim - obs) ** 2).sum()
# nse = 1 - num / denom
# print(f"NSE: {nse:.2f}")
