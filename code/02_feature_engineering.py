"""
Feature Engineering for Flood Risk Analysis
Calculates terrain features, proximity metrics, and prepares ML dataset
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEATURE ENGINEERING FOR FLOOD RISK ANALYSIS")
print("="*80)

# ============================================================================
# LOAD PROCESSED DATA
# ============================================================================
print("\n[1/6] Loading Processed Data...")

# Load DEM
dem_file = 'processed/gafsa_dem.tif'
with rasterio.open(dem_file) as src:
    dem = src.read(1)
    dem_meta = src.meta
    dem_transform = src.transform
    dem_crs = src.crs
    
print(f"✓ DEM loaded: {dem.shape} pixels")
print(f"  Resolution: ~{abs(dem_transform[0]) * 111:.1f} km/pixel")

# Load vector data
gafsa = gpd.read_file('data/administrative/gafsa_boundaries.geojson')
water_areas = gpd.read_file('processed/gafsa_water_areas.geojson')
waterways = gpd.read_file('processed/gafsa_waterways.geojson')
buildings = gpd.read_file('processed/gafsa_buildings.geojson')
landuse = gpd.read_file('processed/gafsa_landuse.geojson')

print(f"✓ Vector data loaded:")
print(f"  - {len(water_areas)} water bodies")
print(f"  - {len(waterways)} waterways")
print(f"  - {len(buildings)} buildings")
print(f"  - {len(landuse)} land use polygons")

# ============================================================================
# TERRAIN FEATURES FROM DEM
# ============================================================================
print("\n[2/6] Calculating Terrain Features...")

# Replace nodata values
dem_clean = dem.copy().astype(np.float32)
dem_clean[dem_clean == -9999] = np.nan

# 1. SLOPE (degrees)
print("  Computing slope...")
dy, dx = np.gradient(dem_clean, abs(dem_transform[0]) * 111000)  # Convert to meters
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
print(f"  ✓ Slope: mean={np.nanmean(slope):.2f}°, max={np.nanmax(slope):.2f}°")

# 2. ASPECT (direction of slope)
print("  Computing aspect...")
aspect = np.degrees(np.arctan2(-dx, dy))
aspect[aspect < 0] += 360  # Convert to 0-360 range
print(f"  ✓ Aspect: mean={np.nanmean(aspect):.2f}°")

# 3. CURVATURE (concave/convex terrain)
print("  Computing curvature...")
dxx = np.gradient(dx, axis=1)
dyy = np.gradient(dy, axis=0)
curvature = -(dxx + dyy)
print(f"  ✓ Curvature: mean={np.nanmean(curvature):.6f}")

# 4. ROUGHNESS (terrain variability)
print("  Computing roughness...")
from scipy.ndimage import generic_filter
roughness = generic_filter(dem_clean, np.std, size=3, mode='constant', cval=np.nan)
print(f"  ✓ Roughness: mean={np.nanmean(roughness):.2f}m")

# 5. ELEVATION CLASSES
print("  Creating elevation classes...")
elev_class = np.zeros_like(dem_clean)
elev_class[(dem_clean >= 0) & (dem_clean < 200)] = 1  # Low
elev_class[(dem_clean >= 200) & (dem_clean < 400)] = 2  # Medium
elev_class[dem_clean >= 400] = 3  # High
print(f"  ✓ Elevation classes: 1=Low, 2=Medium, 3=High")

# 6. TOPOGRAPHIC WETNESS INDEX (TWI)
# TWI = ln(contributing area / tan(slope))
# Simplified version using elevation
print("  Computing Topographic Wetness Index (TWI)...")
slope_rad = np.radians(slope)
slope_rad[slope_rad < 0.001] = 0.001  # Avoid division by zero
# Simple approximation: lower elevations and lower slopes = higher wetness
contributing_area = 1000 - dem_clean  # Inverse elevation as proxy
contributing_area[contributing_area < 1] = 1
twi = np.log((contributing_area + 1) / (np.tan(slope_rad) + 0.001))
print(f"  ✓ TWI: mean={np.nanmean(twi):.2f}, max={np.nanmax(twi):.2f}")

# Save terrain features as GeoTIFFs
print("\n  Saving terrain feature rasters...")
for name, data in [('slope', slope), ('aspect', aspect), ('curvature', curvature), 
                    ('roughness', roughness), ('twi', twi), ('elev_class', elev_class)]:
    output_file = f'features/{name}.tif'
    with rasterio.open(output_file, 'w', **dem_meta) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"    ✓ Saved {output_file}")

# ============================================================================
# PROXIMITY FEATURES
# ============================================================================
print("\n[3/6] Calculating Proximity Features...")

# Create binary rasters for water features
height, width = dem.shape

# Distance to water bodies
print("  Computing distance to water bodies...")
water_mask = rasterize(
    [(geom, 1) for geom in water_areas.geometry],
    out_shape=(height, width),
    transform=dem_transform,
    fill=0,
    dtype=np.uint8
)
dist_water = distance_transform_edt(water_mask == 0) * abs(dem_transform[0]) * 111  # km
print(f"  ✓ Distance to water: mean={np.mean(dist_water):.2f}km, max={np.max(dist_water):.2f}km")

# Distance to waterways
print("  Computing distance to waterways...")
waterway_mask = rasterize(
    [(geom, 1) for geom in waterways.geometry],
    out_shape=(height, width),
    transform=dem_transform,
    fill=0,
    dtype=np.uint8
)
dist_waterways = distance_transform_edt(waterway_mask == 0) * abs(dem_transform[0]) * 111  # km
print(f"  ✓ Distance to waterways: mean={np.mean(dist_waterways):.2f}km, max={np.max(dist_waterways):.2f}km")

# Building density (buildings per km²)
print("  Computing building density...")
building_mask = rasterize(
    [(geom, 1) for geom in buildings.geometry],
    out_shape=(height, width),
    transform=dem_transform,
    fill=0,
    dtype=np.uint8
)
from scipy.ndimage import uniform_filter
building_density = uniform_filter(building_mask.astype(float), size=50, mode='constant')
print(f"  ✓ Building density calculated")

# Land use categories
print("  Processing land use types...")
landuse_risk = {'residential': 3, 'commercial': 3, 'industrial': 2, 'farmland': 1, 
                'farmyard': 1, 'orchard': 1, 'forest': 0, 'park': 1}
landuse['risk_value'] = landuse['fclass'].map(landuse_risk).fillna(0) if 'fclass' in landuse.columns else 0
landuse_raster = rasterize(
    [(geom, val) for geom, val in zip(landuse.geometry, landuse['risk_value'])],
    out_shape=(height, width),
    transform=dem_transform,
    fill=0,
    dtype=np.float32
)
print(f"  ✓ Land use risk map created")

# Save proximity features
print("\n  Saving proximity feature rasters...")
for name, data in [('dist_water', dist_water), ('dist_waterways', dist_waterways), 
                    ('building_density', building_density), ('landuse_risk', landuse_raster)]:
    output_file = f'features/{name}.tif'
    with rasterio.open(output_file, 'w', **dem_meta) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"    ✓ Saved {output_file}")

# ============================================================================
# RAINFALL STATISTICS
# ============================================================================
print("\n[4/6] Extracting Rainfall Statistics...")

# Load rainfall data
rainfall = pd.read_csv('processed/gafsa_rainfall_cleaned.csv')
rainfall['date'] = pd.to_datetime(rainfall['date'])

# Calculate annual statistics
print("  Computing annual rainfall statistics...")
rainfall['year'] = rainfall['date'].dt.year
annual_stats = rainfall.groupby('year').agg({
    'rfh': ['mean', 'max', 'std'],
    'rfh_avg': ['mean', 'max'],
    'rfq': ['mean', 'max']
}).reset_index()
annual_stats.columns = ['_'.join(col).strip('_') for col in annual_stats.columns]
print(f"  ✓ Annual statistics: {len(annual_stats)} years")

# Extreme rainfall events
extreme_days = rainfall[rainfall['rfh'] > 50].copy()
print(f"  ✓ Extreme rainfall days (>50mm): {len(extreme_days)}")

# Seasonal patterns
rainfall['season'] = rainfall['date'].dt.month.apply(
    lambda m: 'Winter' if m in [12,1,2] else 
              'Spring' if m in [3,4,5] else 
              'Summer' if m in [6,7,8] else 'Fall'
)
seasonal_stats = rainfall.groupby('season')['rfh'].agg(['mean', 'max', 'count'])
print(f"\n  Seasonal rainfall patterns:")
for season in seasonal_stats.index:
    print(f"    {season}: mean={seasonal_stats.loc[season, 'mean']:.2f}mm, max={seasonal_stats.loc[season, 'max']:.2f}mm")

# Save rainfall statistics
annual_stats.to_csv('features/annual_rainfall_stats.csv', index=False)
extreme_days.to_csv('features/extreme_rainfall_events.csv', index=False)
seasonal_stats.to_csv('features/seasonal_rainfall_stats.csv')
print(f"\n  ✓ Rainfall statistics saved to features/")

# ============================================================================
# CREATE ML DATASET
# ============================================================================
print("\n[5/6] Creating Machine Learning Dataset...")

# Sample points across the study area (grid sampling)
print("  Generating sampling grid...")
sample_size = 5000  # Number of sample points
rows = np.random.randint(0, height, sample_size)
cols = np.random.randint(0, width, sample_size)

# Extract feature values at sample points
features_dict = {
    'elevation': dem_clean[rows, cols],
    'slope': slope[rows, cols],
    'aspect': aspect[rows, cols],
    'curvature': curvature[rows, cols],
    'roughness': roughness[rows, cols],
    'twi': twi[rows, cols],
    'elev_class': elev_class[rows, cols],
    'dist_water': dist_water[rows, cols],
    'dist_waterways': dist_waterways[rows, cols],
    'building_density': building_density[rows, cols],
    'landuse_risk': landuse_raster[rows, cols]
}

# Create DataFrame
ml_dataset = pd.DataFrame(features_dict)

# Add coordinates
ml_dataset['lon'] = [dem_transform[2] + col * dem_transform[0] for col in cols]
ml_dataset['lat'] = [dem_transform[5] + row * dem_transform[4] for row in rows]

# Remove NaN values
ml_dataset = ml_dataset.dropna()
print(f"  ✓ ML dataset created: {len(ml_dataset)} valid samples")

# Create flood risk score (composite index for initial classification)
print("  Computing preliminary flood risk scores...")
ml_dataset_scaled = ml_dataset.copy()

# Normalize features (0-1 scale)
scaler = StandardScaler()
feature_cols = ['slope', 'twi', 'dist_water', 'dist_waterways', 'building_density', 'landuse_risk']
ml_dataset_scaled[feature_cols] = scaler.fit_transform(ml_dataset[feature_cols])

# Risk score: higher TWI, lower slope, closer to water, higher building density = higher risk
ml_dataset['risk_score'] = (
    ml_dataset_scaled['twi'] * 0.3 +
    (1 - ml_dataset_scaled['slope'] / ml_dataset_scaled['slope'].max()) * 0.2 +
    (1 - ml_dataset_scaled['dist_water'] / ml_dataset_scaled['dist_water'].max()) * 0.2 +
    (1 - ml_dataset_scaled['dist_waterways'] / ml_dataset_scaled['dist_waterways'].max()) * 0.15 +
    ml_dataset_scaled['building_density'] * 0.1 +
    ml_dataset_scaled['landuse_risk'] * 0.05
)

# Classify into risk categories
ml_dataset['risk_category'] = pd.cut(ml_dataset['risk_score'], 
                                      bins=[0, 0.25, 0.5, 0.75, 1.0],
                                      labels=['Low', 'Medium', 'High', 'Very High'])

print(f"\n  Risk distribution:")
for cat in ['Low', 'Medium', 'High', 'Very High']:
    count = (ml_dataset['risk_category'] == cat).sum()
    pct = count / len(ml_dataset) * 100
    print(f"    {cat}: {count} samples ({pct:.1f}%)")

# Save ML dataset
ml_dataset.to_csv('features/ml_dataset.csv', index=False)
print(f"\n  ✓ ML dataset saved: features/ml_dataset.csv")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Generating Feature Visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Feature Engineering Results - Gafsa Flood Risk Analysis', fontsize=16, fontweight='bold')

# Plot features
features_to_plot = [
    (dem_clean, 'Elevation (m)', 'terrain'),
    (slope, 'Slope (degrees)', 'YlOrRd'),
    (twi, 'Topographic Wetness Index', 'Blues'),
    (dist_water, 'Distance to Water (km)', 'viridis_r'),
    (dist_waterways, 'Distance to Waterways (km)', 'plasma_r'),
    (building_density, 'Building Density', 'hot'),
    (landuse_raster, 'Land Use Risk', 'RdYlGn_r'),
    (curvature, 'Curvature', 'RdBu'),
    (roughness, 'Roughness (m)', 'copper')
]

for idx, (data, title, cmap) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('outputs/feature_engineering_results.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: outputs/feature_engineering_results.png")

# Plot risk score distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(ml_dataset['risk_score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Flood Risk Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Flood Risk Scores', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

risk_counts = ml_dataset['risk_category'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Risk Category', fontsize=12)
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.set_title('Flood Risk Classification', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/risk_score_distribution.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: outputs/risk_score_distribution.png")

# ============================================================================
# FEATURE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print("="*80)

summary = f"""
FEATURE ENGINEERING SUMMARY
==========================

1. TERRAIN FEATURES (from 30m DEM):
   - Elevation: {np.nanmean(dem_clean):.1f}m ± {np.nanstd(dem_clean):.1f}m
   - Slope: {np.nanmean(slope):.2f}° ± {np.nanstd(slope):.2f}°
   - TWI: {np.nanmean(twi):.2f} ± {np.nanstd(twi):.2f}
   - Curvature: {np.nanmean(curvature):.6f}
   - Roughness: {np.nanmean(roughness):.2f}m

2. PROXIMITY FEATURES:
   - Distance to water bodies: {np.mean(dist_water):.2f} ± {np.std(dist_water):.2f}km
   - Distance to waterways: {np.mean(dist_waterways):.2f} ± {np.std(dist_waterways):.2f}km
   - Building density: {len(buildings)} buildings mapped

3. RAINFALL STATISTICS:
   - Annual data: {len(annual_stats)} years (1981-2025)
   - Extreme events (>50mm): {len(extreme_days)} days
   - Highest seasonal rainfall: Fall

4. ML DATASET:
   - Total samples: {len(ml_dataset):,}
   - Features: {len(feature_cols) + 4} (terrain + proximity + location)
   - Risk categories: Low ({(ml_dataset['risk_category']=='Low').sum()}), 
                      Medium ({(ml_dataset['risk_category']=='Medium').sum()}), 
                      High ({(ml_dataset['risk_category']=='High').sum()}), 
                      Very High ({(ml_dataset['risk_category']=='Very High').sum()})

FILES CREATED:
   ✓ features/*.tif - 10 raster feature layers
   ✓ features/ml_dataset.csv - ML-ready dataset
   ✓ features/*_stats.csv - Rainfall statistics
   ✓ outputs/feature_engineering_results.png
   ✓ outputs/risk_score_distribution.png

NEXT STEPS:
   1. Train flood risk classification model
   2. Generate flood susceptibility maps
   3. Validate against historical floods
   4. Create final risk assessment report
"""

print(summary)

# Save summary
with open('outputs/feature_engineering_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("✓ Summary saved: outputs/feature_engineering_summary.txt")
