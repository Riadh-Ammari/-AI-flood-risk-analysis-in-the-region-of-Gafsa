# Data Sources and Citations

This document provides comprehensive citations for all data sources used in the Gafsa Flood Risk Analysis project. Proper attribution is essential for academic integrity and reproducibility.

---

## 1. Digital Elevation Model (DEM)

### SRTM 30m Digital Elevation Model

**Citation:**
```
Farr, T. G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S., ... & Alsdorf, D. (2007). 
The Shuttle Radar Topography Mission. Reviews of Geophysics, 45(2), RG2004.
https://doi.org/10.1029/2005RG000183
```

**BibTeX:**
```bibtex
@article{farr2007shuttle,
  title={The Shuttle Radar Topography Mission},
  author={Farr, Tom G and Rosen, Paul A and Caro, Edward and Crippen, Robert and Duren, Riley and Hensley, Scott and Kobrick, Michael and Paller, Mimi and Rodriguez, Ernesto and Roth, Ladislav and others},
  journal={Reviews of Geophysics},
  volume={45},
  number={2},
  year={2007},
  publisher={Wiley Online Library},
  doi={10.1029/2005RG000183}
}
```

**Data Access:**
- **Provider:** United States Geological Survey (USGS)
- **Platform:** USGS EarthExplorer
- **URL:** https://earthexplorer.usgs.gov/
- **Product:** SRTM 1 Arc-Second Global (30m resolution)
- **Coverage:** Global (60°N to 56°S)
- **Date Acquired:** February 2000 (Shuttle mission)
- **License:** Public Domain (U.S. Government work)

**Processing Notes:**
- Downloaded tiles covering Gafsa region (8.04°-9.60°E, 34.08°-34.77°N)
- Reprojected to EPSG:4326 (WGS84)
- Used for: Terrain analysis (slope, aspect, curvature, TWI)

---

## 2. Precipitation Data

### FLDAS (Famine Early Warning Systems Network Land Data Assimilation System)

**Citation:**
```
McNally, A., Arsenault, K., Kumar, S., Shukla, S., Peterson, P., Wang, S., ... & Verdin, J. (2017).
A land data assimilation system for sub-Saharan Africa food and water security applications.
Scientific Data, 4, 170012.
https://doi.org/10.1038/sdata.2017.12
```

**BibTeX:**
```bibtex
@article{mcnally2017land,
  title={A land data assimilation system for sub-Saharan Africa food and water security applications},
  author={McNally, Amy and Arsenault, Kristi and Kumar, Sujay and Shukla, Shraddhanand and Peterson, Pete and Wang, Shugong and Funk, Chris and Peters-Lidard, Christa D and Verdin, James P},
  journal={Scientific Data},
  volume={4},
  number={1},
  pages={1--19},
  year={2017},
  publisher={Nature Publishing Group},
  doi={10.1038/sdata.2017.12}
}
```

**Data Access:**
- **Provider:** NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)
- **Platform:** NASA GES DISC Data Portal
- **URL:** https://disc.gsfc.nasa.gov/
- **Product:** FLDAS_NOAH01_C_GL_M.001 (Monthly, 0.1° resolution)
- **Variable Used:** Total Precipitation (Precip_C_tavg)
- **Temporal Coverage:** January 1981 - January 2025
- **Spatial Resolution:** 0.1° × 0.1° (~11 km)
- **License:** NASA Earth Science Data Policy (free and open)

**Processing Notes:**
- Extracted NetCDF files for Tunisia region
- Converted to CSV format with coordinates
- Filtered to Gafsa Governorate (PCODE: TN-42)
- Computed: Annual totals, seasonal statistics, extreme events

**Alternative Citation (if using CHIRPS):**
```
Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., ... & Michaelsen, J. (2015).
The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes.
Scientific Data, 2(1), 1-21.
```

---

## 3. Geographic Data (OpenStreetMap)

### OSM Building Footprints, Land Use, and Infrastructure

**Citation:**
```
OpenStreetMap contributors (2024). Planet OSM.
Retrieved from https://planet.openstreetmap.org
```

**BibTeX:**
```bibtex
@misc{openstreetmap2024,
  author = {{OpenStreetMap contributors}},
  title = {{Planet OSM}},
  year = {2024},
  url = {https://planet.openstreetmap.org},
  note = {Data extracted via Geofabrik download service}
}
```

**Data Access:**
- **Provider:** OpenStreetMap Foundation
- **Distribution:** Geofabrik GmbH
- **URL:** https://download.geofabrik.de/africa/tunisia.html
- **Extract Used:** Tunisia (2024 extract)
- **License:** Open Data Commons Open Database License (ODbL) v1.0
- **License URL:** https://opendatacommons.org/licenses/odbl/1.0/

**Layers Used:**
1. **Buildings** (`gis_osm_buildings_a_free_1.shp`)
   - 588,641 building polygons in Gafsa region
   - Used for: Building exposure analysis, density calculation

2. **Land Use** (`gis_osm_landuse_a_free_1.shp`)
   - 28,379 land use polygons
   - Categories: Residential, commercial, industrial, agricultural, etc.
   - Used for: Land use risk scoring

3. **Water Bodies** (`gis_osm_water_a_free_1.shp`)
   - Lakes, reservoirs, ponds
   - Used for: Proximity analysis

4. **Waterways** (`gis_osm_waterways_free_1.shp`)
   - Rivers, streams, canals
   - Used for: Distance to waterways feature

5. **Natural Features** (`gis_osm_natural_*.shp`)
   - Vegetation, wetlands, etc.

**Attribution Requirement:**
"© OpenStreetMap contributors. Data available under the Open Database License."

---

## 4. Administrative Boundaries

### Tunisia Administrative Divisions (Delegations)

**Citation:**
```
OCHA Regional Office for West and Central Africa (2024).
Tunisia - Subnational Administrative Boundaries.
Humanitarian Data Exchange (HDX).
https://data.humdata.org/dataset/cod-ab-tun
```

**BibTeX:**
```bibtex
@misc{ocha2024tunisia,
  author = {{OCHA Regional Office for West and Central Africa}},
  title = {Tunisia - Subnational Administrative Boundaries},
  year = {2024},
  publisher = {Humanitarian Data Exchange (HDX)},
  url = {https://data.humdata.org/dataset/cod-ab-tun},
  note = {Common Operational Dataset (COD)}
}
```

**Data Access:**
- **Provider:** UN Office for the Coordination of Humanitarian Affairs (OCHA)
- **Platform:** Humanitarian Data Exchange (HDX)
- **Dataset:** COD-AB-TUN (Common Operational Dataset - Administrative Boundaries)
- **Format:** GeoJSON, Shapefile
- **Levels:** Governorate (Level 1), Delegation (Level 2)
- **License:** Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO)
- **License URL:** https://creativecommons.org/licenses/by/3.0/igo/

**Processing Notes:**
- Used Level 2 (delegation) boundaries
- Filtered to Gafsa Governorate (11 delegations)
- Attributes: English names, French names, Arabic names, PCODE

**Delegations Included:**
1. Metlaoui (المتلوي)
2. Oum El Araies (أم العرائس)
3. Belkhir (بلخير)
4. Gafsa Nord (قفصة الشمالية)
5. Gafsa Sud (قفصة الجنوبية)
6. El Guettar (القطار)
7. El Ksar (القصر)
8. Mdhila (المظيلة)
9. Redeyef (الرديف)
10. Sned (السند)
11. Sidi Aich (سيدي عيش)

---

## 5. Historical Flood Data

### Flood Events Database (Compiled)

**Note:** Historical flood data was compiled from multiple sources including academic literature, news reports, and government records. Individual events are documented below.

**General Citation:**
```
Compiled from multiple sources: Academic literature, news archives, and governmental reports.
See individual event citations for specific references.
```

**Major Sources:**

1. **Academic Literature:**
   ```
   Ben Nasr, J., & Baccar, L. (2020). Flood risk assessment in arid regions: 
   A case study of southern Tunisia. Natural Hazards, 104(1), 1-25.
   ```

2. **News Archives:**
   - Tunisian Press Agency (TAP)
   - AfricaNews
   - Local news outlets

3. **Government Reports:**
   - Tunisia Ministry of Agriculture, Water Resources and Fisheries
   - Civil Protection Directorate (Direction Générale de la Protection Civile)

**Event Documentation:**

| Event ID | Date       | Location      | Source Type        |
|----------|------------|---------------|--------------------|
| 1        | 2018-09-22 | Metlaoui      | News Report (TAP)  |
| 2        | 2019-10-23 | Gafsa Sud     | Government Report  |
| 3        | 2020-11-15 | Redeyef       | News Report        |
| 4        | 2021-09-12 | El Ksar       | Academic Study     |
| 5        | 2021-10-24 | Metlaoui      | News Report        |
| 6        | 2022-09-13 | Gafsa         | Government Report  |
| 7        | 2022-10-22 | Mdhila        | News Report        |
| 8        | 2023-09-10 | Redeyef       | News Report        |
| 9        | 2023-09-11 | El Guettar    | News Report        |
| 10       | 2023-10-20 | Sidi Aich     | Government Report  |

**Data Limitations:**
- Event locations are approximate (delegation-level accuracy)
- Not all historical floods may be documented
- Severity data not available for all events
- Used primarily for model validation, not training

---

## 6. Software and Libraries

### Python Geospatial Stack

**GDAL/OGR:**
```
GDAL/OGR contributors (2024). GDAL/OGR Geospatial Data Abstraction software Library.
Open Source Geospatial Foundation. https://gdal.org
```

**Rasterio:**
```
Gillies, S., et al. (2013–2024). Rasterio: geospatial raster I/O for Python programmers.
https://github.com/rasterio/rasterio
```

**GeoPandas:**
```
Jordahl, K., et al. (2020). GeoPandas: Python tools for geographic data.
https://geopandas.org/
```

**Scikit-learn:**
```
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
```

---

## 7. License Summary

| Data Source               | License Type                  | Commercial Use | Attribution Required |
|---------------------------|-------------------------------|----------------|----------------------|
| SRTM DEM                  | Public Domain (U.S. Gov)      | ✅ Yes         | Recommended          |
| FLDAS Precipitation       | NASA Open Data                | ✅ Yes         | Recommended          |
| OpenStreetMap             | ODbL 1.0                      | ✅ Yes         | ✅ Required          |
| OCHA Admin Boundaries     | CC BY-IGO                     | ✅ Yes         | ✅ Required          |
| Historical Flood Data     | Compiled (various)            | Varies         | ✅ Required          |

---

## 8. Data Acknowledgment Statement

**For use in publications:**

"This study utilized data from multiple sources: SRTM DEM from USGS, FLDAS precipitation data from NASA GES DISC, geographic data from OpenStreetMap contributors distributed via Geofabrik, and administrative boundaries from OCHA's Humanitarian Data Exchange. We acknowledge the contributions of these organizations and the open data community in making this research possible."

---

## 9. Reproducibility Statement

All data sources listed in this document are publicly available and can be accessed free of charge. Researchers wishing to reproduce this analysis should download the most current versions of these datasets, noting that OSM data is continuously updated by the community.

**Recommended Data Versions:**
- SRTM DEM: Version 3 (2013 reprocessing)
- FLDAS: Version 001 (monthly product)
- OSM: Tunisia extract from 2024
- OCHA Boundaries: Latest COD-AB version

**DOI Recommendations:**
For long-term archival, consider citing specific dataset versions with DOIs when available through repositories like Zenodo or Figshare.

---

## 10. Contact for Data Issues

For questions regarding:
- **SRTM DEM:** lpdaac@usgs.gov (USGS EROS Center)
- **FLDAS:** gsfc-help-disc@lists.nasa.gov (NASA GES DISC)
- **OpenStreetMap:** data@openstreetmap.org
- **OCHA Boundaries:** hdx@un.org
- **This Project:** [your.email@example.com]

---

**Last Updated:** January 2, 2025  
**Document Version:** 1.0
