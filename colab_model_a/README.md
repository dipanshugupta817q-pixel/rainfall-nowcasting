# Model A — IMERG ConvLSTM

## Dataset

Historical IMERG rainfall.

Training:
2008–2020

Validation:
2021–2024

Final test:
2025

The 2025 test split is intentionally NOT included
in this Colab training package.

## Input

12 rainfall frames
30-minute resolution

Total history:
6 hours

Grid:
20 × 30

## Output

12 future rainfall frames

Forecast horizon:
6 hours

Resolution:
30 minutes

## Transformation

Training uses:

log1p(rainfall)

Predictions are converted back using:

expm1(prediction)

## Model

Temporal ConvLSTM encoder-decoder.

## Training

Google Colab GPU.
