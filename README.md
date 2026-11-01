# Rainfall Nowcasting — SIH26071

Research project for SIH 2026: AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System.

## Objective

Develop a localized short-term rainfall nowcasting system using satellite rainfall observations and meteorological data, with a target forecast horizon of 1–6 hours.

**Status:** Research prototype — In progress.

## Current Research Components

### Model A — IMERG ConvLSTM
- ConvLSTM-based spatiotemporal model.
- Designed for gridded satellite rainfall observations.
- Trained checkpoints are included in `models/checkpoints/`.

### Model B — ERA5 Atmospheric Forecasting
- ConvLSTM-based atmospheric forecasting model.
- Uses seven ERA5 variables: `t2m`, `d2m`, `u10`, `v10`, `sp`, `msl`, `tcc`.
- Predicts future atmospheric-variable sequences.
- Does not directly predict rainfall intensity.

## Preliminary Evaluation

Model B checkpoint inference was tested on 20 validation sequences. Results varied across the seven variables when compared with a persistence baseline.

This is a preliminary test, not a full validation study. Full-dataset evaluation and forecast-lead analysis are pending.

## Repository Structure

- `colab_model_a/` — IMERG ConvLSTM implementation
- `colab_model_b/` — ERA5 atmospheric forecasting implementation
- `models/checkpoints/` — trained model checkpoints
- `frontend/` — Next.js dashboard prototype
- `mosdac_api/` — satellite data acquisition utilities
- `notebooks/` — research notebooks
- `src/` — supporting source code
- `tests/` — tests and validation utilities
- `data/` — local datasets; most data files are excluded from Git

## Technology

**ML:** Python, PyTorch, NumPy, xarray, h5py  
**Data access:** CDS API, Earthaccess  
**Frontend:** Next.js, React, TypeScript, MapLibre GL

## Setup

```bash
python -m venv .venv