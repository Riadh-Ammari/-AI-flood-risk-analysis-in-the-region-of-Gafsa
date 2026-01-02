# 🌊 Flood Risk Analysis for Gafsa Governorate, Tunisia

A machine learning-based flood susceptibility mapping system using geospatial data analysis and predictive modeling.

## 📋 Project Overview

This project develops an AI-driven flood risk assessment framework for Gafsa Governorate, Tunisia, combining Digital Elevation Models (DEM), precipitation data, land use information, and historical flood records to predict flood-prone areas with 95.35% accuracy.

**Study Area:** Gafsa Governorate, Tunisia (11 delegations)  
**Timeframe:** 1981-2025 (44 years of precipitation data)  
**Resolution:** 30m spatial resolution  
**Model Performance:** 95.35% accuracy (Gradient Boosting)

## 🎯 Key Results

- **Analyzed:** 8.6 million pixels across Gafsa Governorate
- **Assessed:** 23,614 buildings for flood exposure
- **Identified:** 16,868 buildings at high/very high flood risk
- **Validated:** Against 10 historical flood events
- **High-Risk Delegations:** El Ksar (83.8%), Gafsa Sud (63.4%), Redeyef (51.3%)

## 📊 Data Sources

### Elevation Data
- **SRTM DEM 30m** - Shuttle Radar Topography Mission
  - Source: USGS Earth Explorer (https://earthexplorer.usgs.gov/)
  - Resolution: 30m
  - Coverage: Gafsa region (8.04°-9.60°E, 34.08°-34.77°N)

### Precipitation Data
- **FLDAS (Famine Early Warning Systems Network Land Data Assimilation System)**
  - Source: NASA GES DISC (https://disc.gsfc.nasa.gov/)
  - Product: FLDAS_NOAH01_C_GL_M (Monthly, 0.1° resolution)
  - Period: January 1981 - January 2025
  - Dataset: 417,960 precipitation records

### Geographic Data
- **OpenStreetMap (OSM)**
  - Source: Geofabrik (https://download.geofabrik.de/)
  - Layers: Buildings, landuse, waterways, natural features, roads
  - Dataset: 588,641 building polygons

### Administrative Boundaries
- **Tunisia Administrative Divisions**
  - Source: OCHA HDX (Humanitarian Data Exchange)
  - Dataset: delegations-full.geojson
  - Scope: All Tunisia delegations with Gafsa subset (11 delegations)

### Historical Flood Data
- **Flood Events Database**
  - Source: Compiled from multiple sources (academic literature, news reports)
  - Dataset: 10 documented flood events in Gafsa (2018-2023)

## 🏗️ Project Structure

```
GeoaAi/
├── code/                              # Python pipeline scripts
│   ├── 01_data_processing_and_stats.py     # Data cleaning & statistics
│   ├── 02_feature_engineering.py           # Feature extraction from DEM
│   ├── 03_model_training.py                # ML model training
│   ├── 04_flood_susceptibility_mapping.py  # Risk mapping
│   ├── 05_validation.py                    # Historical validation
│   └── 06_delegation_statistics.py         # Per-delegation analysis
│
├── data/                              # Raw input data
│   ├── elevation/                     # DEM files
│   ├── precipitation/                 # Rainfall data
│   ├── osm/                          # OpenStreetMap extracts
│   ├── administrative/               # Boundary shapefiles
│   └── historical_floods/            # Flood event records
│
├── processed/                         # Cleaned data
│   └── gafsa_rainfall_cleaned.csv
│
├── features/                          # Engineered features
│   ├── slope.tif                      # Terrain slope
│   ├── aspect.tif                     # Terrain aspect
│   ├── curvature.tif                  # Surface curvature
│   ├── roughness.tif                  # Terrain roughness
│   ├── twi.tif                        # Topographic Wetness Index
│   ├── elevation_classes.tif          # Elevation categories
│   ├── distance_to_water.tif          # Proximity to water bodies
│   ├── distance_to_waterways.tif      # Proximity to streams
│   ├── building_density.tif           # Building concentration
│   ├── landuse_risk.tif               # Land use flood risk
│   ├── annual_rainfall_stats.csv      # Yearly statistics
│   ├── seasonal_rainfall_stats.csv    # Seasonal patterns
│   ├── extreme_rainfall_events.csv    # Extreme events
│   └── ml_dataset.csv                 # Combined ML training data
│
├── models/                            # Trained ML models
│   ├── rf_model.pkl                   # Random Forest (91.74% accuracy)
│   ├── gb_model.pkl                   # Gradient Boosting (95.35%)
│   └── feature_scaler.pkl             # Feature normalization
│
├── outputs/                           # Results & visualizations
│   ├── data_statistics_report.md      # Data quality report
│   ├── WORKFLOW_EXPLANATION.md        # Methodology documentation
│   ├── flood_susceptibility_*.tif     # Risk maps (GeoTIFF)
│   ├── flood_susceptibility_maps.png  # Risk visualization
│   ├── building_exposure_analysis.png # Building risk charts
│   ├── delegation_risk_analysis.png   # Choropleth maps
│   ├── delegation_risk_statistics.csv # Per-delegation metrics
│   ├── delegation_risk_report.md      # Delegation analysis
│   ├── flood_validation_report.md     # Validation results
│   └── *.png                          # Various visualizations
│
├── docs/                              # Documentation
│   ├── PROJECT_STRUCTURE.md           # Directory guide
│   └── DATA_SOURCES.md                # Data citations
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions
└── README.md                          # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- GDAL libraries
- ~50GB disk space for data files

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gafsa-flood-risk.git
cd gafsa-flood-risk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Download
Due to size constraints, raw data files are not included in this repository. Download from:

1. **DEM Data:** USGS Earth Explorer (SRTM 30m for Gafsa region)
2. **Precipitation:** NASA GES DISC (FLDAS monthly data, 1981-2025)
3. **OSM Data:** Geofabrik Tunisia extract
4. **Administrative Boundaries:** OCHA HDX Tunisia dataset

Place downloaded files in the `data/` directory following the structure above.

## 📖 Usage

Run the pipeline scripts in order:

```bash
# 1. Data processing and cleaning
python code/01_data_processing_and_stats.py

# 2. Feature engineering
python code/02_feature_engineering.py

# 3. Model training
python code/03_model_training.py

# 4. Flood susceptibility mapping
python code/04_flood_susceptibility_mapping.py

# 5. Validation against historical floods
python code/05_validation.py

# 6. Delegation-level statistics
python code/06_delegation_statistics.py
```

## 🧪 Methodology

### Feature Engineering
10 flood risk features extracted:
- **Terrain:** Slope, aspect, curvature, roughness, TWI, elevation classes
- **Proximity:** Distance to water bodies, distance to waterways
- **Human:** Building density, land use risk

### Machine Learning
- **Models:** Random Forest, Gradient Boosting
- **Training:** 80/20 split, 5-fold cross-validation
- **Features:** 10 engineered + 3 rainfall statistics
- **Samples:** 3,328 labeled pixels
- **Best Model:** Gradient Boosting (95.35% accuracy)

### Risk Classification
- **Class 0:** Low Risk (43.97% of area)
- **Class 1:** Medium Risk (23.18%)
- **Class 2:** High Risk (20.28%)
- **Class 3:** Very High Risk (12.56%)

## 📈 Key Findings

### Feature Importance
1. **Topographic Wetness Index (TWI):** 61.9%
2. **Distance to Water:** 23.9%
3. **Distance to Waterways:** 7.9%
4. **Elevation Classes:** 2.5%
5. **Others:** <2% each

### High-Risk Delegations
1. **El Ksar:** 83.8% high/very high risk, 925 buildings at risk
2. **Gafsa Sud:** 63.4% high/very high risk, 5,791 buildings at risk
3. **Redeyef:** 51.3% high/very high risk, 2,774 buildings at risk
4. **Mdhila:** 44.2% high/very high risk, 1,759 buildings at risk
5. **Metlaoui:** 42.1% high/very high risk, 2,128 buildings at risk

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@software{gafsa_flood_risk_2025,
  title={Flood Risk Analysis for Gafsa Governorate, Tunisia},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gafsa-flood-risk}
}
```

## 📚 References

1. **SRTM DEM:** Farr, T. G., et al. (2007). The Shuttle Radar Topography Mission. Reviews of Geophysics, 45(2).
2. **FLDAS:** McNally, A., et al. (2017). A land data assimilation system for sub-Saharan Africa food and water security applications. Scientific Data, 4, 170012.
3. **OpenStreetMap:** OpenStreetMap contributors (2024). Planet OSM. https://planet.openstreetmap.org
4. **Administrative Boundaries:** OCHA Regional Office for West and Central Africa (2024). Tunisia Administrative Boundaries.

## 📧 Contact

For questions or collaboration opportunities, please contact: [your.email@example.com]

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA GES DISC for providing FLDAS precipitation data
- USGS for SRTM DEM data
- OpenStreetMap contributors for geographic data
- OCHA for administrative boundary datasets
