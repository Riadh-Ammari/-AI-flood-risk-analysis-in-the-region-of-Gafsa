# Project Structure Documentation

## Directory Organization

### `/code/` - Analysis Pipeline

Contains 6 sequential scripts that form the complete flood risk analysis pipeline:

#### `01_data_processing_and_stats.py`
**Purpose:** Data cleaning, quality assessment, and statistics generation

**Inputs:**
- `data/precipitation/tunisia_rainfall_full.csv` - Raw precipitation data
- `data/administrative/delegations-full.geojson` - Administrative boundaries
- `data/osm/*.shp` - OpenStreetMap shapefiles (buildings, landuse, water, etc.)
- `data/historical_floods/flood_events.csv` - Historical flood records

**Outputs:**
- `processed/gafsa_rainfall_cleaned.csv` - Filtered and cleaned rainfall data
- `outputs/data_statistics_report.md` - Comprehensive data quality report
- Console output with data summary statistics

**Key Functions:**
- Filters data to Gafsa region (PCODE = TN-42)
- Removes duplicates and handles missing values
- Generates data completeness metrics (99.9% quality score)
- Creates statistical summaries for all datasets

---

#### `02_feature_engineering.py`
**Purpose:** Extract flood risk features from geospatial data

**Inputs:**
- `data/elevation/gafsa_dem.tif` - Digital Elevation Model
- `processed/gafsa_rainfall_cleaned.csv` - Cleaned rainfall data
- `data/osm/gis_osm_water_*.shp` - Water bodies and waterways
- `data/osm/gis_osm_buildings_*.shp` - Building footprints
- `data/osm/gis_osm_landuse_*.shp` - Land use polygons
- `data/historical_floods/flood_events_ml.csv` - Labeled flood locations

**Outputs:**
- `features/slope.tif` - Terrain slope (degrees)
- `features/aspect.tif` - Terrain aspect (degrees)
- `features/curvature.tif` - Surface curvature
- `features/roughness.tif` - Terrain roughness
- `features/twi.tif` - Topographic Wetness Index
- `features/elevation_classes.tif` - Elevation categories (1-5)
- `features/distance_to_water.tif` - Euclidean distance to water bodies (m)
- `features/distance_to_waterways.tif` - Distance to streams/rivers (m)
- `features/building_density.tif` - Buildings per km²
- `features/landuse_risk.tif` - Land use flood risk score (0-3)
- `features/annual_rainfall_stats.csv` - Yearly precipitation statistics
- `features/seasonal_rainfall_stats.csv` - Seasonal patterns
- `features/extreme_rainfall_events.csv` - Extreme precipitation events
- `features/ml_dataset.csv` - Combined training dataset (3,328 samples)

**Key Functions:**
- Calculates terrain derivatives using GDAL
- Computes Topographic Wetness Index (TWI)
- Generates proximity rasters using distance transforms
- Assigns risk scores to land use types
- Merges all features with labeled flood/non-flood points

---

#### `03_model_training.py`
**Purpose:** Train and evaluate machine learning models

**Inputs:**
- `features/ml_dataset.csv` - Feature matrix with labels

**Outputs:**
- `models/rf_model.pkl` - Trained Random Forest classifier
- `models/gb_model.pkl` - Trained Gradient Boosting classifier
- `models/feature_scaler.pkl` - StandardScaler for feature normalization
- `outputs/model_training_results.png` - Performance visualizations
- Console output with accuracy metrics and confusion matrices

**Key Functions:**
- Splits data into 80% training, 20% testing
- Trains Random Forest (100 trees) and Gradient Boosting (200 estimators)
- Performs 5-fold cross-validation
- Calculates feature importance rankings
- Generates confusion matrices and ROC curves

**Results:**
- Random Forest: 91.74% test accuracy
- Gradient Boosting: **95.35% test accuracy** (selected model)
- Cross-validation: 93.5% ± 1.29%

---

#### `04_flood_susceptibility_mapping.py`
**Purpose:** Apply trained model to entire study area

**Inputs:**
- `models/gb_model.pkl` - Best performing model
- `models/feature_scaler.pkl` - Feature scaler
- `features/*.tif` - All 10 feature rasters
- `data/osm/gis_osm_buildings_*.shp` - Buildings for exposure analysis

**Outputs:**
- `outputs/flood_susceptibility_gb_class.tif` - Risk classification map (0-3)
- `outputs/flood_susceptibility_gb_prob.tif` - Risk probability map (0-1)
- `outputs/flood_susceptibility_rf_class.tif` - Random Forest predictions
- `outputs/flood_susceptibility_maps.png` - 4-panel visualization
- `outputs/building_exposure_analysis.png` - Building risk charts
- `outputs/flood_susceptibility_statistics.csv` - Area statistics by risk class

**Key Functions:**
- Loads and stacks all feature rasters
- Applies trained model to 8.6 million pixels
- Classifies each pixel into risk categories (0-3)
- Performs spatial join with building footprints
- Calculates building exposure statistics

**Results:**
- 43.97% Low Risk
- 23.18% Medium Risk
- 20.28% High Risk
- 12.56% Very High Risk
- 16,868 buildings at high/very high risk

---

#### `05_validation.py`
**Purpose:** Validate predictions against historical flood events

**Inputs:**
- `data/historical_floods/flood_events.csv` - 10 historical floods
- `data/administrative/delegations-full.geojson` - Delegation boundaries
- `outputs/flood_susceptibility_gb_class.tif` - Predicted risk map

**Outputs:**
- `outputs/flood_validation_results.png` - Map overlay of predictions and events
- `outputs/flood_validation_report.md` - Validation summary
- `outputs/flood_validation_details.csv` - Per-event risk scores

**Key Functions:**
- Maps flood events to delegation centroids (approximate locations)
- Extracts predicted risk at flood locations
- Compares predictions with observed events
- Generates validation visualizations

**Limitations:**
- Historical flood locations are approximate (delegation-level)
- Limited to 10 documented events
- Validation accuracy constrained by location precision

---

#### `06_delegation_statistics.py`
**Purpose:** Generate per-delegation risk statistics for decision-makers

**Inputs:**
- `data/administrative/delegations-full.geojson` - Administrative boundaries
- `outputs/flood_susceptibility_gb_class.tif` - Risk classification map
- `data/osm/gis_osm_buildings_*.shp` - Building footprints

**Outputs:**
- `outputs/delegation_risk_statistics.csv` - Metrics for each delegation
- `outputs/delegation_risk_analysis.png` - Choropleth maps and charts
- `outputs/priority_delegations.png` - Top 5 high-risk delegations
- `outputs/delegation_risk_report.md` - Summary report

**Key Functions:**
- Filters Gafsa delegations from full Tunisia dataset
- Clips risk raster to each delegation boundary
- Calculates area percentages by risk class
- Computes composite risk scores (0-1)
- Identifies buildings at risk per delegation
- Ranks delegations by flood vulnerability

**Results:**
- 10 delegations successfully analyzed (1 outside raster bounds)
- El Ksar identified as highest risk (83.8% high/very high)
- 23,614 buildings assessed across all delegations

---

## `/data/` - Raw Input Data

### `elevation/`
- **gafsa_dem.tif** - SRTM 30m Digital Elevation Model
- Source: USGS Earth Explorer
- Coverage: 8.04°-9.60°E, 34.08°-34.77°N
- Resolution: ~30m × 30m

### `precipitation/`
- **tunisia_rainfall_full.csv** - 417,960 precipitation records
- Source: NASA FLDAS (NOAH01_C_GL_M)
- Period: January 1981 - January 2025
- Variables: Date, Latitude, Longitude, Precipitation (mm)

### `administrative/`
- **delegations-full.geojson** - Tunisia administrative boundaries
- Source: OCHA Humanitarian Data Exchange
- Contains: All Tunisia delegations with attributes (names, codes)
- Used for: Filtering Gafsa region and spatial aggregation

### `osm/`
OpenStreetMap shapefiles from Geofabrik:
- **gis_osm_buildings_a_free_1.shp** - 588,641 building polygons
- **gis_osm_landuse_a_free_1.shp** - 28,379 land use features
- **gis_osm_water_*.shp** - Water bodies and waterways
- **gis_osm_natural_*.shp** - Natural features
- **gis_osm_roads_*.shp** - Road network
- **gis_osm_railways_*.shp** - Railway lines

### `historical_floods/`
- **flood_events.csv** - 10 documented flood events (2018-2023)
- **flood_events_ml.csv** - Labeled points for ML training
- Compiled from: Academic literature, news reports, government records

---

## `/processed/` - Cleaned Data

### `gafsa_rainfall_cleaned.csv`
- Filtered rainfall data for Gafsa region only
- 417,960 records → Gafsa subset
- Duplicates removed, missing values handled
- Ready for feature engineering

---

## `/features/` - Engineered Features

### Terrain Features (GeoTIFF rasters)
1. **slope.tif** - Terrain slope in degrees (0-90°)
2. **aspect.tif** - Slope direction (0-360°)
3. **curvature.tif** - Surface curvature (convex/concave)
4. **roughness.tif** - Terrain roughness index
5. **twi.tif** - Topographic Wetness Index (most important feature: 61.9%)
6. **elevation_classes.tif** - Categorized elevation (5 classes)

### Proximity Features (GeoTIFF rasters)
7. **distance_to_water.tif** - Distance to nearest water body (meters)
8. **distance_to_waterways.tif** - Distance to nearest stream (meters)

### Human Activity Features (GeoTIFF rasters)
9. **building_density.tif** - Building concentration (buildings/km²)
10. **landuse_risk.tif** - Land use flood susceptibility (0-3)

### Rainfall Features (CSV files)
- **annual_rainfall_stats.csv** - Yearly totals, mean, std, max
- **seasonal_rainfall_stats.csv** - Wet/dry season patterns
- **extreme_rainfall_events.csv** - Events exceeding thresholds

### ML Dataset
- **ml_dataset.csv** - Combined feature matrix (3,328 samples × 13 features)
  - Columns: All 10 terrain/proximity features + 3 rainfall statistics + label (0/1)
  - Used for: Model training and evaluation

---

## `/models/` - Trained ML Models

### `rf_model.pkl`
- Random Forest classifier (100 trees)
- Test accuracy: 91.74%
- Trained on 2,662 samples

### `gb_model.pkl` ⭐
- Gradient Boosting classifier (200 estimators)
- **Test accuracy: 95.35%** (best model)
- Selected for final predictions

### `feature_scaler.pkl`
- StandardScaler object
- Fitted on training data
- Required for: Normalizing features before prediction

---

## `/outputs/` - Results and Visualizations

### Reports
- **data_statistics_report.md** - Data quality metrics and summaries
- **WORKFLOW_EXPLANATION.md** - Methodology documentation
- **delegation_risk_report.md** - Per-delegation analysis
- **flood_validation_report.md** - Model validation results

### Statistics (CSV)
- **delegation_risk_statistics.csv** - Risk metrics by delegation
- **flood_susceptibility_statistics.csv** - Area distribution by risk class
- **flood_validation_details.csv** - Historical event validation

### Risk Maps (GeoTIFF)
- **flood_susceptibility_gb_class.tif** - Classification (0-3) - 2.54 MB
- **flood_susceptibility_gb_prob.tif** - Probability (0-1) - 28.51 MB
- **flood_susceptibility_rf_class.tif** - Random Forest predictions

### Visualizations (PNG)
- **01_workflow_diagram.png** - Complete pipeline flowchart
- **02_data_quality.png** - Data completeness charts
- **03_spatial_overview.png** - Study area map
- **04_features_summary.png** - 9-panel feature visualization
- **05_correlations.png** - Feature correlation heatmap
- **model_training_results.png** - Confusion matrices and performance
- **flood_susceptibility_maps.png** - 4-panel risk visualization
- **building_exposure_analysis.png** - Building risk distribution
- **delegation_risk_analysis.png** - Choropleth maps by delegation
- **priority_delegations.png** - Top 5 high-risk delegations
- **flood_validation_results.png** - Historical flood overlay

---

## File Size Summary

### Large Files (Not in Git)
- **flood_susceptibility_gb_prob.tif**: 28.51 MB
- **gafsa_dem.tif**: ~15 MB (estimated)
- **OSM shapefiles**: ~200 MB (all combined)
- **FLDAS NetCDF**: ~500 MB (if stored)

### Medium Files
- **Models**: ~5 MB combined
- **Feature rasters**: ~20 MB total
- **Visualizations**: ~10 MB total

### Small Files
- **CSV datasets**: <5 MB
- **Reports**: <1 MB
- **Code**: <500 KB

**Total Project Size:** ~50 GB (including all raw data)  
**Core Results Only:** ~100 MB (without raw data files)

---

## Data Flow Summary

```
Raw Data (data/)
    ↓
01_data_processing_and_stats.py
    ↓
Cleaned Data (processed/)
    ↓
02_feature_engineering.py
    ↓
Features (features/)
    ↓
03_model_training.py
    ↓
Trained Models (models/)
    ↓
04_flood_susceptibility_mapping.py
    ↓
Risk Maps (outputs/*.tif)
    ↓
05_validation.py → Validation Results (outputs/)
    ↓
06_delegation_statistics.py → Delegation Analysis (outputs/)
```

---

## Reproducibility Notes

To reproduce this analysis:

1. **Download raw data** (see README.md Data Sources section)
2. **Place files** in appropriate `data/` subdirectories
3. **Run scripts sequentially** (01 → 02 → 03 → 04 → 05 → 06)
4. **Intermediate outputs** will be generated automatically
5. **Final results** will appear in `outputs/` directory

All file paths are relative to project root, ensuring portability across systems.
