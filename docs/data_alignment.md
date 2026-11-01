\# Data Alignment and Resampling



\## Canonical Temporal Resolution



The rainfall nowcasting system uses:



&#x20;   30 minutes



as its canonical temporal resolution.



\---



\## IMERG



IMERG Final Run is already available at:



&#x20;   30-minute intervals



Therefore:



&#x20;   No temporal resampling is required.



The historical IMERG dataset is retained at its native

30-minute resolution.



\---



\## Canonical Spatial Grid



All model inputs will use:



&#x20;   Latitude  : 20 points

&#x20;   Longitude : 30 points



Spatial domain:



&#x20;   Latitude  : 28.05°N -> 29.95°N

&#x20;   Longitude : 76.05°E -> 78.95°E



\---



\## Future Data Sources



Additional data sources may have different temporal

resolutions.



Examples include:



\- Weather observations

\- ERA5

\- Numerical Weather Prediction data

\- Satellite observations

\- Radar observations



These sources must be transformed to the canonical

30-minute timeline before model integration.



\---



\## Temporal Resampling Rule



When a source has a temporal resolution coarser than

30 minutes:



&#x20;   Interpolation or temporal expansion may be required,

&#x20;   depending on the physical variable.



When a source has a finer temporal resolution:



&#x20;   Aggregation or temporal selection may be required.



The resampling method must depend on the physical

meaning of the variable.



\---



\## Important Constraint



Rainfall should not be blindly interpolated between

observations.



For precipitation variables, aggregation or

physically appropriate temporal transformation should

be preferred over arbitrary linear interpolation.



\---



\## Current Status



IMERG:



&#x20;   Native 30-minute resolution

&#x20;   No resampling required



Spatial grid:



&#x20;   Already matches canonical project grid



Temporal grid:



&#x20;   Already matches canonical project timeline

