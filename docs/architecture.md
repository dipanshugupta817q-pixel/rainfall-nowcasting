# System Architecture

## SIH26071 Short-Term Rainfall Nowcasting

```text
                 HISTORICAL DATA
                       +
              NEAR-REAL-TIME DATA
                       │
                       ▼
              ┌─────────────────┐
              │  DATA INGESTION │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PREPROCESSING   │
              │                 │
              │ Missing Values  │
              │ Outliers        │
              │ Alignment       │
              │ Normalization   │
              └────────┬────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ SPATIO-TEMPORAL         │
          │ SEQUENCE GENERATION     │
          │                          │
          │ 12 Past Rainfall Frames │
          └────────────┬─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    ConvLSTM     │
              │     MODEL       │
              └────────┬────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ 1–6 HOUR RAINFALL        │
          │ FORECAST                 │
          └────────────┬─────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     INTENSITY      LOCATION     MOVEMENT
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PROBABILITY &   │
              │ CONFIDENCE      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FASTAPI        │
              │  BACKEND        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   DASHBOARD     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ FLOOD /         │
              │ INUNDATION      │
              │ MODULE          │
              └─────────────────┘
              
Core Data Flow
Rainfall Data
      ↓
Spatial Grid
      ↓
Time Synchronization
      ↓
Preprocessing
      ↓
Sequence Creation
      ↓
ConvLSTM
      ↓
Future Rainfall Maps
      ↓
Movement Analysis
      ↓
Nowcast API
      ↓
Dashboard

Initial Model Configuration

Input:

12 frames
30-minute interval
6-hour historical context

Output:

12 frames
30-minute interval
6-hour forecast horizon
Past 6 Hours
│
├── t-5:30
├── t-5:00
├── t-4:30
├── ...
├── t-1:00
├── t-0:30
└── t=0

        ↓

      ConvLSTM

        ↓

Next 6 Hours
│
├── t+0:30
├── t+1:00
├── t+1:30
├── ...
├── t+5:30
└── t+6:00