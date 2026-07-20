# NASA POWER Climate Forcing Verification Package

Supporting data for the climate forcing parameters used in:
"A Physics-Informed Neural Operator for Thermal Ranking of Low-Cost Wall
Materials in Hot-Dry Climates" (manuscript under review, IJHMT,
ref. HMT-D-26-04702).

Purpose: independent verification of the two fixed climate forcing values
adopted in the manuscript, (i) peak solar irradiance of 900 W/m^2 and
(ii) diurnal outdoor temperature swing of Delta T = 12 K, against the
NASA POWER dataset for rural upper Sindh, Pakistan.

## Data source

NASA Prediction of Worldwide Energy Resources (POWER),
https://power.larc.nasa.gov, accessed via the Data Access Viewer
(DAV v2.6.13) in July 2026.

- Solar irradiance: CERES SYN1deg (native resolution 1 deg x 1 deg)
- Temperature: MERRA-2 reanalysis (native resolution 0.5 deg lat x
  0.625 deg lon)
- All timestamps in Local Solar Time (LST)
- Missing-data flag: -999

## Request settings (common to all files)

- Capability: Single Point, Standard Resolution
- User Community: Sustainable Buildings
- Parameters: ALLSKY_SFC_SW_DWN (All Sky Surface Shortwave Downward
  Irradiance, W m-2), T2M_RANGE (Temperature at 2 Meters Range, K)
- Output format: CSV

Note on user communities: POWER user communities act only as
parameter-menu filters and unit conventions over the same underlying
MERRA-2 and CERES SYN1deg sources; the retrieved variables are core
meteorological quantities available in all communities. Under the
Sustainable Buildings community, hourly irradiance is the hourly-mean
irradiance and daily irradiance is the daily-mean irradiance, both in
W m-2.

## Files

1. POWER_Point_Hourly_20250501_20250630_027d70N_068d86E_LST.csv
   Hourly ALLSKY_SFC_SW_DWN, Sukkur (27.70 N, 68.86 E),
   May 1 - Jun 30, 2025. 1464 records (61 days x 24 h), no missing
   values. Primary dataset for the peak-irradiance verification.

2. POWER_Point_Daily_20240501_20240630_027d70N_068d86E_LST.csv
   Daily ALLSKY_SFC_SW_DWN + T2M_RANGE, Sukkur,
   May 1 - Jun 30, 2024. 61 days, complete.

3. POWER_Point_Daily_20250501_20250630_027d70N_068d86E_LST.csv
   Daily ALLSKY_SFC_SW_DWN + T2M_RANGE, Sukkur,
   May 1 - Jun 30, 2025. 61 days, complete.

4. POWER_Point_Daily_20260501_20260630_027d70N_068d86E_LST.csv
   Daily ALLSKY_SFC_SW_DWN + T2M_RANGE, Sukkur,
   May 1 - Jun 30, 2026. 60 of 61 days valid (one day carries the
   -999 source-availability flag).

5. POWER_Point_Daily_20250501_20250630_027d56N_068d21E_LST.csv
   Daily ALLSKY_SFC_SW_DWN + T2M_RANGE, Larkana (27.56 N, 68.21 E),
   May 1 - Jun 30, 2025. 61 days, complete. Spatial sample: Larkana
   lies in a different MERRA-2 grid cell (cell elevation 52.74 m vs
   62.79 m at Sukkur). Note that Larkana and Sukkur share the same
   1-deg CERES solar cell, so the irradiance value is a ~100 km
   regional average covering both districts.

## Key results

Peak solar irradiance (File 1, Sukkur, hourly, May-Jun 2025):
- Median daily peak hourly-mean irradiance: 888 W/m^2
- Mean of daily peaks: 882 W/m^2; maximum: 1017 W/m^2
- Days with peak >= 900 W/m^2: 28 of 61 (46%)
- Daily peak occurs at LST hour 11-12, confirming correct local-time
  handling and the half-sine, noon-centred profile assumed in the
  manuscript's forcing model.
- Hourly means understate instantaneous peaks, so the adopted
  instantaneous 900 W/m^2 is supported conservatively.
- Corroboration from daily means (Files 2-4): May-Jun median
  daily-mean irradiance of 267-298 W/m^2 across 2024-2026 is
  consistent with half-sine profiles peaking near or above 900 W/m^2
  in all three years.

Diurnal temperature swing (T2M_RANGE, May-Jun):
- Sukkur 2024 (File 2): median 16.5 K, mean 16.2 K, range 7.1-20.6 K,
  58/61 days >= 12 K
- Sukkur 2025 (File 3): median 16.2 K, mean 16.0 K, range 6.6-21.2 K,
  54/61 days >= 12 K
- Sukkur 2026 (File 4): median 16.1 K, mean 16.3 K, range 10.8-23.3 K,
  56/60 days >= 12 K
- Larkana 2025 (File 5): median 16.0 K, mean 15.7 K, range
  6.7-20.6 K, 57/61 days >= 12 K
- Conclusion: across three consecutive years at Sukkur and an
  independent grid cell at Larkana, the typical May-Jun swing in
  rural upper Sindh is approximately 16 K (medians 16.0-16.5 K), with
  observed daily values spanning roughly 7-23 K. The manuscript's
  adopted Delta T = 12 K is a conservative value near the 10th
  percentile of this distribution, as stated in the manuscript
  (outdoor temperature forcing subsection).

## Consistency checks performed

1. Completeness: record counts match expectations in all files; the
   only -999 value is a single flagged day in File 4.
2. Physical sanity: hourly irradiance is exactly 0 during night hours
   (22:00-03:00 LST); daily irradiance peaks fall at solar noon.
3. Spatial: an independent MERRA-2 temperature cell (Larkana)
   reproduces the Sukkur swing distribution.
4. Interannual: three consecutive years (2024-2026) give May-Jun
   swing medians within 0.5 K of each other.

## Reproduction

Any file can be reproduced at
https://power.larc.nasa.gov/data-access-viewer with the settings above,
or via the POWER API using parameters ALLSKY_SFC_SW_DWN and T2M_RANGE,
community=SB, time-standard=LST, and the coordinates and date ranges
listed per file.
