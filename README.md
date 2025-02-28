# δHBV 2.0


First introduced in the preprint “High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning” by Song et al. (2024), δHBV 2.0 is the latest-generation differentiable HBV model leveraging intelligent parameterization, big data, and highly-parallelized GPU compute with PyTorch to deliver CONUS-scale, high-resolution inference of HBV parameters and fluxes. (See publication [below](#publication) for details.)

This repo is an operations-level package for use with NOAA-OWP’s Next Generation National Water Modeling Framework ([NextGen](https://github.com/NOAA-OWP/ngen)) and currently supports δHBV 2.0 with unit hydrograph (UH) routing.

<br>

## Model Description:

δHBV 2.0UH is a differentiable model, characterized by the use of an LSTM to learn parameters for the differentiable physical model HBV, which is then forwarded with weather forcings (precipitation, temperature, PET) to predict hydraulic states and fluxes like streamflow with high spatial resolution. Routing is then done with UH, which uses parameters learned from a separate MLP neural network. In essence, δHBV 2.0UH is a differentiable modeling modality defined by

    HBV_params = LSTM( Forcings, Basin Attributes )
    Routing_params = MLP( Forcings, Basin Attributes )

    e.g., Streamflow = HBV2.0( Forcings, HBV_params, Routing_params)

Learned parameters are spatially distinct, but can also be time-dependent if desired.

*Note that HBV here differs from the original NumPy version proposed by Beck et al. (2020), with modifications for multiscale modeling and PyTorch compatibility.*

<br>

## Model Development and Training:

δHBV 2.0UH is built on the generic differentiable modeling framework [δMG](https://github.com/mhpi/generic_deltaModel), a successor package to [HydroDL](https://github.com/mhpi/hydroDL) serving as a model testbed intended to accelerate deployment in operational environments. Therefore, while this package includes HBV, utility code and neural networks are imported from δMG. Note that while training codes for this model will be released in δMG at a later time, we offer an [example script](https://github.com/mhpi/generic_deltaModel/blob/master/example/hydrology/example_dhbv_2_0.ipynb) demonstrating forward inference on δMG's development backend.

We also have model training/validation/inference [examples](https://github.com/mhpi/generic_deltaModel/tree/master/example/hydrology) for predecessor models δHBV 1.0 and δHBV 1.1p, which give more detail on differentiable model construction for parameter learning in practice.

<br>

## Package Organization:
This package is intended to be placed in NextGen's `extern/` directory, and contains the following components:
- The physical model HBV 2.0 + UH;
- Model and data configuration files;
- δHBV 2.0UH BMI to interface with NextGen and forward within the framework. (Note, this uses δMG's differentiable modeling pipeline as a backbone to build and forward the complete differentiable model: LSTM & MLP + HBV 2.0);
- BMI configuration files;
- NextGen realization files;

*Note: Currently the structure of this repository is modeled after NOAA-OWP's ngen-compatible [LSTM](https://github.com/NOAA-OWP/lstm) developed by Scott Peckham, Jonathan Frame et al.*

<br>

## Operational Deployment:
1. Install [NextGen in a Box](https://github.com/CIROH-UA/NGIAB-CloudInfra) (NGIAB) or the NextGen [prototype](https://github.com/NOAA-OWP/ngen) from NOAA-OWP;
3. If using NGIAB, compile with docker image.
4. Clone the `dHBV_2_0` package,
   ```bash
   git clone https://github.com/mhpi/dHBV_2_0.git
   ```
5. Move `dHBV_2_0` to NextGen's `extern/` directory.
6. Download a demo subset of AORC forcings and Hydrofabric 2.2 basin attributes [here](NEEDS_LINK). Add this sample data to the `forcings/` directory in NextGen;
7. Place the realization files in `config/` and begin Nextgen model forwarding with the `run.sh` script.

<br>

## Publication:

*Song, Yalan, Tadd Bindas, Chaopeng Shen, Haoyu Ji, Wouter Johannes Maria Knoben, Leo Lonzarich, Martyn P. Clark et al. "High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning." Authorea Preprints (2024). https://essopenarchive.org/doi/full/10.22541/essoar.172736277.74497104.*

<br>

## Issues:
For questions, concerns, bugs, etc., please reach out by posting an issue here or on the [δMG repo](https://github.com/mhpi/generic_deltaModel/issues).
