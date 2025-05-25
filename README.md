# δHBV 2.0

First introduced in the preprint “High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning” by Song et al. (2024), δHBV 2.0 is the latest-generation differentiable HBV model leveraging intelligent parameterization, big data, and highly-parallelized GPU compute with PyTorch to deliver CONUS-scale, high-resolution inference of HBV parameters and fluxes. (See publication [below](#publication) for details.)

This repo is an operations-level package for use with NOAA-OWP’s Next Generation National Water Modeling Framework ([NextGen](https://github.com/NOAA-OWP/ngen)) and currently supports δHBV 2.0 with unit hydrograph (UH) routing.

</br>

## Model Description

δHBV 2.0UH is a differentiable model, characterized by the use of an LSTM to learn parameters for the differentiable physical model HBV, which can be forwarded with weather forcings (precipitation, temperature, PET) to predict hydrological states and fluxes like streamflow with high spatial resolution. Routing can be done with UH which uses parameters learned from a separate MLP neural network, or with NextGen's internal t-route integration, which uses parameters learned from a separate MLP neural network. In essence, δHBV 2.0UH is a differentiable modeling modality defined by

    HBV_params = LSTM( Forcings, Basin Attributes )
    Routing_params = MLP( Forcings, Basin Attributes )

    Fluxes/States e.g., Streamflow = HBV2.0( Forcings, HBV_params, Routing_params)

Learned parameters are spatially distinct, but can also be time-dependent if desired.

*Note that HBV here differs from the original NumPy version proposed by Beck et al. (2020), with modifications for multiscale modeling and PyTorch compatibility.*

</br>

## Model Development and Training

δHBV 2.0UH is built on the generic differentiable modeling framework [δMG](https://github.com/mhpi/generic_deltamodel), a successor package to [HydroDL](https://github.com/mhpi/hydroDL) serving as a model testbed intended to accelerate deployment in operational environments. Therefore, while this package includes the physical HBV model, utility code and neural networks are imported from δMG. Note that training codes will be released in δMG at a later time, but we offer an [example script](https://github.com/mhpi/generic_deltamodel/blob/master/example/hydrology/example_dhbv_2_0.ipynb) demonstrating forward inference on δMG's development backend.

We also provide model training/validation/inference [examples](https://github.com/mhpi/generic_deltamodel/tree/master/example/hydrology) for precursor models δHBV 1.0 and δHBV 1.1p, which give more detail on differentiable model construction in practice.

</br>

## Package Organization

The entirety of this package is intended to be placed in NextGen's `extern/` directory, and contains the following components:

- The physical model HBV 2.0 + UH;
- Model and data configuration files;
- δHBV 2.0UH BMI to interface with NextGen and forward within the framework. (Note, this uses δMG's differentiable modeling pipeline as a backbone to build and forward the complete differentiable model: LSTM & MLP + HBV 2.0);
- BMI configuration files;
- NextGen realization files;

</br>

## Operational Deployment

1. Install [NextGen in a Box](https://github.com/CIROH-UA/NGIAB-CloudInfra) (NGIAB) or the NextGen [prototype](https://github.com/NOAA-OWP/ngen) from NOAA-OWP;
2. If using NGIAB, compile with docker image.
3. Clone the `dHBV_2_0` package,

        ```bash
        git clone https://github.com/mhpi/dHBV2.0.git
        ```

4. Move `dHBV2.0` to NextGen's `extern/` directory.
5. Download a demo subset of AORC forcings and Hydrofabric 2.2 basin attributes [here](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/aorc_hydrofabric/ngen_demo.zip). Add this sample data to the `forcings/` directory in NextGen;
6. Begin Nextgen model forwarding; e.g. `./build/ngen ./data/dhbv_2_0/spatial/catchment_data_cat-88306.geojson 'cat-88306' ./data/dhbv_2_0/spatial/nexus_data_nex-87405.geojson 'nex-87405 ./data/dhbv_2_0/realizations/realization_cat-88306.json`.

</br>

## Publication

*Yalan Song, Tadd Bindas, Chaopeng Shen, Haoyu Ji, Wouter Johannes Maria Knoben, Leo Lonzarich, Martyn P. Clark, et al. "High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning." Water Resources Research (2025). <https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024WR038928>.*

</br>

## Issues

For questions, or to report bugs, concerns, etc., please reach out by posting an issue here or on the [dMG repo](https://github.com/mhpi/generic_deltamodel/issues).
