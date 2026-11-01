# Model B — ERA5 Atmospheric Forecasting

Deep-learning atmospheric forecasting model for the rainfall nowcasting system.

## Data

Input:
- 12 ERA5 frames
- 30-minute resolution
- 6-hour history
- 7 atmospheric variables
- 20 × 30 spatial grid

Output:
- 12 future frames
- next 6 hours
- 7 atmospheric variables

## Variables

1. t2m
2. d2m
3. u10
4. v10
5. sp
6. msl
7. tcc

## Dataset split

TRAIN:
2021–2023

VALIDATION:
2024

FINAL TEST:
2025

2025 test data is intentionally NOT included in this package.

## Model

ConvLSTM encoder + autoregressive ConvLSTM decoder.

## Training

Run:

python src/training/train_era5.py
