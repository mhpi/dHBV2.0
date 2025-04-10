(Using from S.D. Peckham)

For use in NextGen, the dHBV_2_0 project folder should be copied into
the NextGen project folder, at ngen/extern/dhbv_2_0.  Please see the
text file:  "How_to_Run_LSTM_in_NextGen.txt" in the docs folder for
much more information.

This subfolder, "ngen_files", contains files that should be copied into
the ngen repo tree (or project folder), into the indicated subfolders.
This will allow the new LSTM Python package to be run from within NextGen.

The "ngen_files/data/dhbv_2_0" folder contains:
(1) "realization config" files for NextGen in "ngen_rc" folder
(2) catchment and nexus data GeoJSON files in "spatial" folder,
    including ones for HUCO1, CAMELS-test, etc.
(3) dHBV 2.0 model config files in the "config" folder.
(4) forcing data for testing AORC on the Juniata River Basin (JRB), etc. in data/forcing
