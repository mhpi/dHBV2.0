# δHBV2.0

First introduced in the preprint *“High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning”* by Song et al. (2025), δHBV2.0 is the latest-generation differentiable HBV model leveraging intelligent parameterization, big data, and highly-parallelized GPU compute with PyTorch to deliver CONUS-scale, high-resolution inference of HBV parameters and fluxes. See [publication](#publication) for details.

This repo is an operations-level module for use with NOAA-OWP’s Next Generation National Water Modeling Framework ([NextGen](https://github.com/NOAA-OWP/ngen)) and currently supports δHBV2.0 with unit hydrograph (UH) routing.

</br>

## Model Description

δHBV2.0UH is a differentiable model, characterized by the use of an LSTM and MLP to learn parameters for the differentiable physical model HBV2.0, which can fed along with weather forcings (precipitation, temperature, PET) to simulate hydrological states and fluxes (like streamflow) with high spatial resolution. Routing can be done with UH, or with NextGen's internal T-Route integration. In essence, δHBV 2.0UH is a differentiable modeling modality defined by

$$
    \theta_{d, m}^{1:t} = LSTM( x_m^{1:t}, A_m )
$$

$$
    \theta_{s, m} = MLP( A_m )
$$

$$
    Q_k^{1:t}, S_k^{1:t} = HBV(x_m^{1:t}, \theta_{d, m}^{1:t}, \theta_{s, m})
$$

where $\theta_{d, m}^{1:t}$ and $\theta_{d, m}^{1:t}$ are learned dynamic and static HBV parameters, $x_m^{1:t}$ are unit-basin-scale forcings, $A_m$ are unit basin attributes, respectively, $Q_b^{1:t}$ are HBV fluxes (e.g., streamflow), and $S_b^{1:t}$ are HBV states (e.g., snowpack) for unit basins $m\in [1, 2, \ldots, M]$ and coarse gage basins $k\in [1, 2, \ldots, K]$. All model parameters are spatially distinct, but can be learned with time-dependency if desired.

*Note that HBV here differs from the original NumPy version proposed by Beck et al. (2020), with modifications for multiscale training and PyTorch compatibility.*

</br>

## Model Development and Training

δHBV2.0UH is built on the differentiable modeling framework [δMG](https://github.com/mhpi/generic_deltamodel), a successor to [HydroDL](https://github.com/mhpi/hydroDL) serving as a testbed for accelerating operational deployments of differentiable model. As such, this module minimally contains the physical model HBV2.0, a BMI, and configuration/realization files to support NextGen operation (see [below](#package-organization)). At runtime, the physical model is ported to δMG and composed with neural networks to form the differentiable model δHBV2.0. *Note*: training code will be released in δMG at a later time, but we offer an [example script](https://github.com/mhpi/generic_deltamodel/blob/master/example/hydrology/example_dhbv_2_0.ipynb) demonstrating inference with δMG.

We also provide model training/validation/inference [examples](https://github.com/mhpi/generic_deltamodel/tree/master/example/hydrology) for precursor models δHBV 1.0 and δHBV 1.1p, which give more detail on differentiable model construction in practice.

</br>

## Package Organization

The entirety of this module is intended to be placed in NextGen's `extern/` directory, and contains the following components:

- The physical model HBV2.0;
- Model and data configuration files;
- δHBV2.0UH BMI to interface with NextGen. (This uses δMG's differentiable modeling pipeline as a backbone to build and forward the complete differentiable model: LSTM & MLP + HBV2.0);
- BMI configuration files;
- NextGen realization files;

</br>

## Operational Deployment

1. Install [NextGen in a Box](https://github.com/leoglonz/NGIAB-DM) or the [NextGen framework](https://github.com/leoglonz/ngen). Both of these are modifications from [CIROH-UA](https://github.com/CIROH-UA) distributions in order to support differentiable modeling.
2. If using NGIAB, compile with Docker [`docker/Dockerfile`](https://github.com/leoglonz/NGIAB-DM/blob/master/docker/Dockerfile). Instructions for using NGIAB can be found on the official [DocuHub](https://docs.ciroh.org/docs/products/Community%20Hydrologic%20Modeling%20Framework/nextgeninaboxDocker/workflow/).
3. Clone `dHBV2.0`,

    ```bash
    git clone https://github.com/mhpi/dHBV2.0.git
    ```

4. Move `dHBV2.0` to NextGen's `extern/` directory.
5. Download a demo subset of AORC forcings and Hydrofabric 2.2 basin attributes from [AWS](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/aorc_hydrofabric/ngen_demo.zip). Add this sample data to the `forcings/` directory in NextGen;
6. Start Nextgen model forward; e.g. `./build/ngen ./data/dhbv2/spatial/catchment_data_cat-88306.geojson 'cat-88306' ./data/dhbv2/spatial/nexus_data_nex-87405.geojson 'nex-87405 ./data/dhbv2/realizations/realization_cat-88306.json`.

</br>

## Publication

*Yalan Song, Tadd Bindas, Chaopeng Shen, Haoyu Ji, Wouter Johannes Maria Knoben, Leo Lonzarich, Martyn P. Clark, et al. "High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning." Water Resources Research (2025). <https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024WR038928>.*

</br>

## Issues

For questions, or to report bugs, concerns, etc., please reach out by posting an issue here or on the [𝛿MG repo](https://github.com/mhpi/generic_deltamodel/issues).
