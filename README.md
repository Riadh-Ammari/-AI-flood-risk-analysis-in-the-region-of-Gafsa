# 🌊 GeoAI Flood Susceptibility Mapping System
### Machine Learning-Based Flood Risk Analysis for Gafsa, Tunisia

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)

A comprehensive geospatial AI system for flood susceptibility mapping and risk assessment in the Gafsa Governorate, Tunisia. This project combines machine learning, geographic information systems (GIS), and interactive visualization to predict flood risk zones and support disaster management planning.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Sources](#-data-sources)
- [Methodology](#-methodology)
- [Results](#-results)
- [Web Application](#-web-application)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This system analyzes multiple environmental and geographical factors to identify flood-prone areas in Gafsa, Tunisia. Using advanced machine learning algorithms and 45 years of historical rainfall data, the system achieves **95.35% accuracy** in predicting flood risk zones.

### Key Capabilities

- **Multi-factor Analysis**: Integrates 10+ geographical and hydrological features
- **Historical Validation**: Validated against real flood events
- **Interactive Mapping**: Web-based interface for risk visualization
- **Delegation-Level Statistics**: Risk metrics for all 11 delegations in Gafsa
- **Building Exposure Analysis**: Assessment of infrastructure at risk

### Study Area

- **Region**: Gafsa Governorate, Tunisia
- **Coverage**: 11 administrative delegations
- **Bounding Box**: 8.04°E to 9.60°E, 34.08°N to 34.77°N
- **Temporal Coverage**: 1981-2025 (45 years of rainfall data)

---

## ✨ Features

### 🤖 Machine Learning Models

- **Gradient Boosting Classifier**: 95.35% accuracy (primary model)
- **Random Forest Classifier**: 91.74% accuracy
- **Cross-validation**: 93.5% ± 1.29%
- Multi-class classification (Low, Medium, High, Very High risk)

### 📊 Comprehensive Analysis

- **Terrain Analysis**: Slope, aspect, curvature, roughness
- **Hydrological Features**: TWI (Topographic Wetness Index), distance to water bodies
- **Land Use Integration**: Risk scoring based on land cover types
- **Building Density**: Infrastructure exposure assessment
- **Rainfall Patterns**: Historical precipitation analysis

### 🗺️ Interactive Web Application

- Real-time flood risk prediction
- Interactive maps with risk zone visualization
- Delegation-specific risk statistics
- Building exposure analysis
- Historical trend analysis
- Export capabilities for reports

---

## 📁 Project Structure

```
GeoaAi/
│
├── app.py                          # Streamlit web application
├── generate_visualizations.py      # Visualization generation script
├── README.md                       # Project documentation
│
├── code/                           # Analysis pipeline scripts
│   ├── 01_data_processing_and_stats.py    # Data cleaning & statistics
│   ├── 02_feature_engineering.py          # Feature extraction
│   ├── 03_model_training.py               # ML model training
│   ├── 04_flood_susceptibility_mapping.py # Risk map generation
│   ├── 05_validation.py                   # Historical flood validation
│   └── 06_delegation_statistics.py        # Delegation-level analysis
│
├── data/                           # Input datasets (not in repository)
│   ├── administrative/             # Boundaries and administrative data
│   ├── buildings/                  # Building footprints (shapefile)
│   ├── elevation/                  # SRTM DEM data (.hgt)
│   ├── historical_floods/          # Historical flood event records
│   ├── landuse/                    # Land use/land cover data
│   ├── precipitation/              # Rainfall time series
│   ├── soil/                       # Soil properties
│   └── water/                      # Water areas and waterways
│
├── features/                       # Engineered features (generated)
│   ├── ml_dataset.csv              # Final ML-ready dataset
│   ├── annual_rainfall_stats.csv   # Yearly rainfall statistics
│   ├── seasonal_rainfall_stats.csv # Seasonal patterns
│   └── extreme_rainfall_events.csv # Extreme event records
│
├── models/                         # Trained ML models (generated)
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── outputs/                        # Analysis results
│   ├── maps/                       # GeoTIFF risk maps (28+ MB, excluded)
│   ├── visualizations/             # Charts and figures
│   ├── reports/                    # Markdown and text reports
│   └── statistics/                 # CSV and JSON statistics
│
└── processed/                      # Processed/cleaned data (generated)
    ├── gafsa_buildings.geojson
    ├── gafsa_landuse.geojson
    ├── gafsa_rainfall_cleaned.csv
    ├── gafsa_water_areas.geojson
    └── gafsa_waterways.geojson
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Required Libraries

```bash
# Core dependencies
pip install pandas numpy
pip install geopandas shapely rasterio
pip install scikit-learn
pip install matplotlib seaborn plotly
pip install streamlit streamlit-folium
pip install folium
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/GeoaAi.git
cd GeoaAi
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Usage

### Complete Analysis Pipeline

Run the analysis scripts in sequence:

```bash
# 1. Data processing and statistics
python code/01_data_processing_and_stats.py

# 2. Feature engineering
python code/02_feature_engineering.py

# 3. Train ML models
python code/03_model_training.py

# 4. Generate flood susceptibility maps
python code/04_flood_susceptibility_mapping.py

# 5. Validate with historical floods
python code/05_validation.py

# 6. Generate delegation statistics
python code/06_delegation_statistics.py
```

### Generate Visualizations

```bash
python generate_visualizations.py
```

### Launch Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 📦 Data Sources

### Required Input Data

1. **Administrative Boundaries** (`gafsa_boundaries.geojson`)
   - Delegation boundaries for Gafsa Governorate

2. **Elevation Data** (`n34e008.hgt`)
   - SRTM 1 Arc-Second Global (30m resolution)
   - Source: [USGS Earth Explorer](https://earthexplorer.usgs.gov/)

3. **Rainfall Data** (`tunisia_rainfall_full.csv`)
   - Historical precipitation records (1981-2025)
   - Daily measurements with spatial coverage

4. **Building Footprints** (`buildings.shp`)
   - Vector data of structures in Gafsa

5. **Land Use/Land Cover** (`landuse.shp`)
   - Land cover classification

6. **Water Features**
   - `water_areas.shp`: Lakes, reservoirs, wetlands
   - `waterways.shp`: Rivers, streams, drainage

7. **Historical Floods** (`flood_events_ml.csv`)
   - Documented flood occurrences for validation

8. **Soil Properties** (`soil_properties.csv`)
   - Soil characteristics affecting drainage

> **Note**: Due to size and licensing restrictions, raw data files are not included in this repository. Contact the project maintainer for data access.

---

## 🔬 Methodology

### 1. Data Processing

- Clipping datasets to Gafsa boundary
- Quality assessment and cleaning
- Coordinate system standardization (EPSG:4326)
- Missing value handling (99.9% data completeness achieved)

### 2. Feature Engineering

**Terrain Features** (from DEM):
- Slope (degrees)
- Aspect (orientation)
- Curvature (surface shape)
- Roughness (terrain variability)
- Topographic Wetness Index (TWI)
- Elevation classification

**Hydrological Features**:
- Distance to water bodies
- Distance to waterways
- Drainage density

**Anthropogenic Features**:
- Building density
- Land use risk scoring

**Meteorological Features**:
- Rainfall intensity
- Extreme precipitation events
- Seasonal patterns

### 3. Model Training

**Algorithm Selection**: Ensemble methods chosen for robustness
- Gradient Boosting Classifier (best performance)
- Random Forest Classifier (baseline comparison)

**Training Strategy**:
- Train/test split: 70/30
- 5-fold cross-validation
- Feature standardization (StandardScaler)
- Hyperparameter optimization

**Risk Classification**:
- **Low Risk**: Minimal flood probability
- **Medium Risk**: Moderate susceptibility
- **High Risk**: Elevated flood potential
- **Very High Risk**: Critical flood zones

### 4. Validation

- Historical flood event overlay analysis
- Confusion matrix evaluation
- Precision, recall, F1-score metrics
- Spatial validation with known flood locations

---

## 📈 Results

### Model Performance

| Metric | Gradient Boosting | Random Forest |
|--------|------------------|---------------|
| **Accuracy** | 95.35% | 91.74% |
| **Precision** | 94.2% | 90.1% |
| **Recall** | 95.1% | 91.5% |
| **F1-Score** | 94.6% | 90.8% |
| **Cross-Validation** | 93.5% ± 1.29% | 89.2% ± 2.15% |

### Risk Distribution

| Risk Category | Area Coverage |
|--------------|---------------|
| Low Risk | 43.97% |
| Medium Risk | 23.18% |
| High Risk | 20.28% |
| **Very High Risk** | **12.56%** |

### Top 5 High-Risk Delegations

1. **Gafsa Nord** - 18.3% very high risk
2. **Gafsa Sud** - 16.7% very high risk
3. **El Ksar** - 14.2% very high risk
4. **Sidi Aich** - 12.9% very high risk
5. **Redeyef** - 11.4% very high risk

### Historical Validation

- **84%** of documented floods occurred in High/Very High risk zones
- **92%** spatial correlation with flood-prone areas
- Strong validation against 2018-2023 flood events

### Building Exposure

- **2,847 buildings** in Very High risk zones
- **4,521 buildings** in High risk zones
- Total: 7,368 structures requiring flood mitigation measures

---

## 💻 Web Application

The interactive Streamlit application provides:

### 📍 Interactive Risk Map
- Folium-based map with risk zone overlays
- Delegation boundaries
- Color-coded risk classification
- Click-to-inspect functionality

### 📊 Real-Time Prediction
- Input custom geographical parameters
- Instant flood risk classification
- Probability scores
- Feature importance visualization

### 📈 Statistical Dashboard
- Delegation comparison charts
- Risk distribution pie charts
- Historical trend analysis
- Building exposure graphs

### 📥 Export Features
- Download risk maps
- Generate PDF reports
- Export statistics to CSV
- Share analysis results

---

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **GeoPandas**: Spatial data manipulation
- **Rasterio**: Raster data processing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Folium**: Interactive mapping
- **Plotly/Matplotlib**: Data visualization
- **Pandas/NumPy**: Data analysis

---

## 📝 Output Files

### Generated Visualizations

All visualizations are saved to `outputs/visualizations/`:
- `model_training_results.png` - Model performance metrics
- `flood_susceptibility_maps.png` - 4-panel risk maps
- `building_exposure_analysis.png` - Infrastructure at risk
- `delegation_risk_analysis.png` - Choropleth maps
- `priority_delegations.png` - Top 5 risk areas
- `flood_validation_results.png` - Historical validation
- `feature_engineering_results.png` - Feature distributions
- `05_correlations.png` - Feature correlation heatmap

### Reports

Available in `outputs/reports/`:
- **Data Statistics Report** (`data_statistics_report.md`) - Comprehensive data quality assessment
- **Delegation Risk Report** (`delegation_risk_report.md`) - Per-delegation risk metrics
- **Flood Validation Report** (`flood_validation_report.md`) - Model validation results
- **Feature Engineering Summary** (`feature_engineering_summary.txt`) - Feature extraction log

### Maps

GeoTIFF files in `outputs/maps/` (excluded from Git due to size):
- `flood_susceptibility_gb_class.tif` - Risk classification raster
- `flood_susceptibility_gb_prob.tif` - Probability raster (0-1)
- `flood_susceptibility_rf_class.tif` - Random Forest predictions

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional ML algorithms (SVM, Neural Networks)
- Real-time rainfall data integration
- Mobile application development
- Multi-language support
- Additional validation metrics
- Performance optimization

---

## 🔮 Future Enhancements

- [ ] Real-time monitoring integration
- [ ] Climate change scenario modeling
- [ ] Ensemble model with deep learning
- [ ] Mobile app for field data collection
- [ ] API for third-party integration
- [ ] Early warning system
- [ ] Social vulnerability assessment
- [ ] Economic impact analysis

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **USGS**: For providing SRTM elevation data
- **OpenStreetMap**: For building and infrastructure data
- **Tunisia National Meteorological Institute**: For historical rainfall data
- **Gafsa Governorate**: For administrative boundaries and flood records
- **scikit-learn community**: For excellent machine learning tools
- **Streamlit team**: For the amazing web framework

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/GeoaAi/issues)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 📚 Citations

If you use this project in your research, please cite:

```bibtex
@software{geoai_flood_mapping,
  author = {Your Name},
  title = {GeoAI Flood Susceptibility Mapping System for Gafsa, Tunisia},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/GeoaAi}
}
```

---

## 📊 Project Status

🟢 **Active Development** - Regular updates and improvements ongoing

**Last Updated**: January 2026

---

<div align="center">

### ⭐ If you find this project useful, please give it a star!

Made with ❤️ for disaster risk reduction and community resilience

</div>
