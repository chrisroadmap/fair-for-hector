# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run SSP scenarios with fair

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
fair.__version__

# %% [markdown]
# ## historical and future run
#
# Run two scenarios as a test case (not enough memory on laptop to run eight), then take this on to Hydra for scaling up.

# %%
f = FAIR(ch4_method='Thornhill2021')
f.define_time(1750, 2500, 1)
scenarios = ['ssp119', 'ssp126']#, 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']
f.define_scenarios(scenarios)

# %%
fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)
configs = df_configs.index  # label for the "config" axis
f.define_configs(configs)

# %%
species, properties = read_properties(filename='../data/fair-calibration/species_configs_properties_1.4.0.csv')
f.define_species(species, properties)

# %%
f.allocate()

# %%
f.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)

# %%
# I was lazy and didn't convert emissions to CSV, so use the old clunky method of importing from netCDF
# this is from calibration-1.4.0
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = 841
da = da_emissions.loc[dict(config="unspecified", scenario=scenarios)]
fe = da.expand_dims(dim=["config"], axis=(2))
f.emissions = fe.drop_vars(("config")) * np.ones((1, 1, output_ensemble_size, 1))

# %%
fill(
    f.forcing,
    f.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    f.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
f.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")

# fill(f_stoch.climate_configs['stochastic_run'], True)
# fill(f_stoch.climate_configs['use_seed'], True)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
f.run()

# %% [markdown]
# ## get output that Brian wants
#
# - emissions of all species
# - concentrations of all species
# - radiative forcing of all species
# - temperature
# - ECS of each config
#
# in addition I will supply
# - ocean heat content change
# - top of atmosphere energy imbalance

# %%
ds = xr.Dataset(
    data_vars = dict(
        emissions = (['timepoints', 'scenario', 'species'], f.emissions.sel(config=0).data),
        concentration = (['timebounds', 'scenario', 'config', 'species'], f.concentration.data),
        forcing = (['timebounds', 'scenario', 'config', 'species'], f.forcing.data),
        forcing_sum = (['timebounds', 'scenario', 'config'], f.forcing_sum.data),
        temperature_rel_1750 = (['timebounds', 'scenario', 'config'], f.temperature.sel(layer=0).data),
        ocean_heat_content_change = (['timebounds', 'scenario', 'config'], f.ocean_heat_content_change.data),
        toa_imbalance = (['timebounds', 'scenario', 'config'], f.toa_imbalance.data),
        ecs = (['config'], f.ebms.ecs.data),
    ),
    coords = dict(
        timepoints = f.timepoints,
        timebounds = f.timebounds,
        scenario = scenarios,
        config = df_configs.index,
        species = species
    )
)

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/ssp_fair2.2.2_cal1.4.0.nc')

# %%
