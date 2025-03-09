## TODO: needs to be updated to latest dHBV 2.0 dMG implementation.

import numpy as np
from pathlib import Path
import dHBV_2_0.src.dHBV_2_0.bmi_dhbv as bmi_dhbv   # Load module bmi_dm (bmi_dm.py) from dhbv_2_0 package.
import os, os.path
lstm_dir = os.path.expanduser('../dhbv_2_0/')
os.chdir( lstm_dir )
import pandas as pd

basin_id = "cat-88306"


# Load the USGS data 
# REPLACE THIS PATH WITH YOUR LOCAL FILE PATH:
file_path = f"/Users/jmframe/CAMELS_data_sample/hourly/usgs-streamflow/{basin_id}-usgs-hourly.csv"
df_runoff = pd.read_csv(file_path)
df_runoff = df_runoff.set_index("date")
df_runoff.index = pd.to_datetime(df_runoff.index)
df_runoff = df_runoff[["QObs(mm/h)"]].rename(columns={"QObs(mm/h)": "usgs_obs"})
df_runoff["model_pred"] = None

# REPLACE THIS PATH WITH YOUR LOCAL FILE PATH:
forcing_file_path = f"/Users/jmframe/CAMELS_data_sample/hourly/aorc_hourly/{basin_id}_1980_to_2024_agg_rounded.csv"
df_forcing = pd.read_csv(forcing_file_path)
df_forcing = df_forcing.set_index("time")
df_forcing.index = pd.to_datetime(df_forcing.index)
df_forcing = df_forcing[df_runoff.index[0]:df_runoff.index[-1]]

# Create an instance of the LSTM model with BMI
model_instance = bmi_lstm.bmi_LSTM()

# Initialize the model with a configuration file
model_instance.initialize(bmi_cfg_file=Path(f'../bmi_config_files/{basin_id}_nh_AORC_hourly_ensemble.yml'))

# Add ensemble columns to the runoff DataFrame
for i_ens in range(model_instance.N_ENS):
    df_runoff[f"ensemble_{i_ens+1}"] = None  # Initialize ensemble columns with None


# Iterate through the forcing DataFrame and calculate model predictions
print('Working, please wait...')
for i, (idx, row) in enumerate(df_forcing.iterrows()):
    # Extract forcing data for the current timestep
    precip = row["APCP_surface"]
    temp = row["TMP_2maboveground"]
    dlwrf = row["DLWRF_surface"]
    dswrf = row["DSWRF_surface"]
    pres = row["PRES_surface"]
    spfh = row["SPFH_2maboveground"]
    ugrd = row["UGRD_10maboveground"]
    vgrd = row["VGRD_10maboveground"]

    # Check if any of the inputs are NaN
    if np.isnan([precip, temp, dlwrf, dswrf, pres, spfh, ugrd, vgrd]).any():
        if model_instance.verbose > 0:
            print(f"Skipping timestep {idx} due to NaN values in inputs.")
        continue

    # Set the model forcings
    model_instance.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', precip)
    model_instance.set_value('land_surface_air__temperature', temp)
    model_instance.set_value('land_surface_radiation~incoming~longwave__energy_flux', dlwrf)
    model_instance.set_value('land_surface_radiation~incoming~shortwave__energy_flux', dswrf)
    model_instance.set_value('land_surface_air__pressure', pres)
    model_instance.set_value('atmosphere_air_water~vapor__relative_saturation', spfh)
    model_instance.set_value('land_surface_wind__x_component_of_velocity', ugrd)
    model_instance.set_value('land_surface_wind__y_component_of_velocity', vgrd)

    # Update the model
    model_instance.update()

    # Retrieve and scale the runoff output
    dest_array = np.zeros(1)
    model_instance.get_value('land_surface_water__runoff_depth', dest_array)
    land_surface_water__runoff_depth = dest_array[0] * 1000  # Convert to mm/hr

    # Add ensemble member values to the DataFrame
    for i_ens in range(model_instance.N_ENS):
        df_runoff.loc[idx, f"ensemble_{i_ens+1}"] = model_instance.surface_runoff_mm[i_ens]  # Add individual ensemble member values


    # Add the output to the DataFrame
    df_runoff.loc[idx, "model_pred"] = land_surface_water__runoff_depth

    if i > 10000:
        break


# Ensure "model_pred" is numeric
df_runoff["model_pred"] = pd.to_numeric(df_runoff["model_pred"], errors="coerce")

# Calculate NSE for the model predictions
obs = df_runoff["usgs_obs"].dropna()
sim = df_runoff["model_pred"].dropna()

# Align indices of observation and simulation for metric calculation
common_index = obs.index.intersection(sim.index)
obs = obs.loc[common_index].values
sim = sim.loc[common_index].values

denominator = ((obs - obs.mean()) ** 2).sum()
numerator = ((sim - obs) ** 2).sum()
nse = 1 - numerator / denominator
print(f"NSE: {nse:.2f}")