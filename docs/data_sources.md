\# Data Sources



\## 1. Historical Rainfall — NASA GPM IMERG Final Run



\### Dataset

NASA GPM Integrated Multi-satellitE Retrievals for GPM (IMERG)



\### Product

GPM IMERG Half-Hourly Final Run



\### Version

V07B



\### Temporal Coverage

2008-06-01 to 2020-09-30



\### Selected Months

June, July, August, September



\### Temporal Resolution

30 minutes



\### Spatial Resolution

0.1° approximately



\### Spatial Domain

Latitude:

28.05°N to 29.95°N



Longitude:

76.05°E to 78.95°E



\### Regional Grid

Latitude points: 20

Longitude points: 30



Each rainfall frame therefore has:



&#x20;   (time=1, latitude=20, longitude=30)



\### Variable



Primary variable:



&#x20;   precipitation



Unit:



&#x20;   mm/hr



\### Stored File Format



NPZ



Canonical historical filename:



&#x20;   YYYYMMDD\_HHMM.npz



Each file contains:



&#x20;   precipitation

&#x20;   lat

&#x20;   lon

&#x20;   time



\### Dataset Completeness



Expected frames:



&#x20;   76,128



Available expected frames:



&#x20;   76,128



Missing frames:



&#x20;   0



Duplicate timestamps:



&#x20;   0



Additional files:



&#x20;   48 frames from June 2022



The additional 2022 frames are outside the historical

training period and will not be included in the training index.



\### Preprocessing Already Performed



1\. Server-side spatial subset from NASA OPeNDAP.

2\. Regional extraction to 20 × 30 grid.

3\. Conversion to NumPy-compatible NPZ format.

4\. Precipitation dimensions standardized to:



&#x20;      (time, latitude, longitude)



5\. Missing/fill values represented appropriately.

6\. Historical timestamp coverage verified.

7\. Duplicate timestamp files removed from the active dataset.



\### Quality-Control Status



Historical IMERG dataset:



&#x20;   VERIFIED



Coverage:



&#x20;   COMPLETE



Temporal gaps:



&#x20;   NONE



Spatial grid:



&#x20;   CONSISTENT



Duplicate timestamps:



&#x20;   NONE



\---



\## 2. Future Weather / NWP Data



Planned source:



ERA5 / compatible numerical weather prediction data.



Potential variables:



\- 2 m temperature

\- 2 m dew point temperature

\- 10 m u-component wind

\- 10 m v-component wind

\- Mean sea level pressure

\- Surface pressure

\- Total precipitation

\- Cloud-related variables



These variables will be integrated after the rainfall-only

baseline has been established.



\---



\## 3. Future Satellite Data



Satellite observations will be integrated as additional

spatio-temporal channels after the baseline rainfall model.



\---



\## 4. Future Radar Data



Radar observations will be integrated where suitable

coverage and historical availability permit.



\---



\## 5. Data Integration Principle



All data sources must ultimately be transformed to a

common:



\- geographic grid

\- temporal resolution

\- coordinate convention

\- missing-value representation



before being passed to the multi-channel nowcasting model.

