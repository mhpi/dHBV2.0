(Using from S.D. Peckham)

For use in NextGen, the lstm_py project folder should be copied into
the NextGen project folder, at ngen/extern/lstm_py.  Please see the
text file:  "How_to_Run_LSTM_in_NextGen.txt" in the docs folder for
much more information.

This subfolder, "ngen_files", contains files that should be copied into
the ngen repo tree (or project folder), into the indicated subfolders.
This will allow the new LSTM Python package to be run from within NextGen.

The "ngen_files/data/lstm" folder contains:
(1) "realization config" files for NextGen in "rc_files" folder
(2) catchment and nexus data GeoJSON files in "spatial" folder,
    including ones for HUCO1, CAMELS-test, etc.
(3) LSTM model config files in the "config" folder.
    Some of these are also in lstm_py/bmi_config_files.
(4) forcing data for testing HUC01, CAMELS-test, etc. in data/forcing
