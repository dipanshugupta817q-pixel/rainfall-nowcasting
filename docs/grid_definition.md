\# Canonical Geographic Grid



\## Purpose



The project uses one canonical geographic grid for all

spatio-temporal rainfall nowcasting data.



All future data sources such as weather, NWP, satellite,

and radar data must ultimately be aligned to this grid

before entering the model.



\---



\## Spatial Domain



\### Latitude



Minimum:



&#x20;   28.05°N



Maximum:



&#x20;   29.95°N



Number of grid points:



&#x20;   20



\### Longitude



Minimum:



&#x20;   76.05°E



Maximum:



&#x20;   78.95°E



Number of grid points:



&#x20;   30



\---



\## Grid Resolution



Approximately:



&#x20;   0.1° × 0.1°



The IMERG source grid is retained as the canonical

reference grid.



\---



\## Tensor Shape



Spatial rainfall frame:



&#x20;   (latitude, longitude)



&#x20;   (20, 30)



Single temporal frame including time dimension:



&#x20;   (time, latitude, longitude)



&#x20;   (1, 20, 30)



\---



\## Coordinate Convention



Latitude axis:



&#x20;   ascending



Longitude axis:



&#x20;   ascending



The same coordinate ordering must be maintained during

all preprocessing and model preparation.



\---



\## Reference Grid



The IMERG latitude and longitude arrays are treated as

the reference coordinates.



Future datasets must be transformed onto these

coordinates rather than changing the canonical grid.



\---



\## Data Alignment Rule



For every additional data source:



1\. Read its native latitude/longitude grid.

2\. Check coordinate orientation.

3\. Handle longitude convention if required.

4\. Reproject/regrid where necessary.

5\. Interpolate or aggregate to the canonical grid.

6\. Verify final shape is:



&#x20;      (20, 30)



7\. Verify geographic bounds.

8\. Verify that no unintended spatial shift has occurred.



\---



\## Model Input Convention



After temporal sequence construction, spatial tensors

will use:



&#x20;   (time, latitude, longitude, channels)



or, for framework-specific ConvLSTM input:



&#x20;   (batch, time, latitude, longitude, channels)



The final model tensor layout will be explicitly verified

before training.

