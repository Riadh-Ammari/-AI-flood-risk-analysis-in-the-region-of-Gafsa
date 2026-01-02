import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FLOOD SUSCEPTIBILITY MAPPING")
print("=" * 80)

# Load trained models and scaler
print("\n1. Loading trained models...")
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)
with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("   Models loaded successfully!")

# Load all feature rasters
print("\n2. Loading feature rasters...")
feature_files = [
    'features/slope.tif',
    'features/aspect.tif',
    'features/curvature.tif',
    'features/roughness.tif',
    'features/twi.tif',
    'features/elev_class.tif',
    'features/dist_water.tif',
    'features/dist_waterways.tif',
    'features/building_density.tif',
    'features/landuse_risk.tif'
]

feature_names = ['slope', 'aspect', 'curvature', 'roughness', 'twi', 'elev_class',
                 'dist_water', 'dist_waterways', 'building_density', 'landuse_risk']

# Read first raster to get dimensions and metadata
with rasterio.open(feature_files[0]) as src:
    height, width = src.height, src.width
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    print(f"   Raster dimensions: {height} x {width} = {height * width:,} pixels")

# Load all features into a 3D array
features_stack = np.zeros((len(feature_files), height, width), dtype=np.float32)
for i, file in enumerate(feature_files):
    with rasterio.open(file) as src:
        features_stack[i] = src.read(1)
    print(f"   Loaded: {feature_names[i]}")

print(f"\n3. Preparing data for prediction...")
# Reshape for model prediction: (height*width, n_features)
n_pixels = height * width
features_2d = features_stack.reshape(len(feature_files), -1).T
print(f"   Feature array shape: {features_2d.shape}")

# Create mask for valid pixels (no NaN in any feature)
valid_mask = ~np.any(np.isnan(features_2d), axis=1)
valid_pixels = features_2d[valid_mask]
print(f"   Valid pixels: {valid_pixels.shape[0]:,} ({100 * valid_pixels.shape[0] / n_pixels:.1f}%)")
print(f"   Invalid pixels (NaN): {n_pixels - valid_pixels.shape[0]:,}")

# Standardize features
print(f"\n4. Standardizing features...")
valid_pixels_scaled = scaler.transform(valid_pixels)
print(f"   Features scaled using saved scaler")

# Predict using both models
print(f"\n5. Generating predictions...")
print(f"   Random Forest prediction...")
rf_pred = rf_model.predict(valid_pixels_scaled)
rf_proba = rf_model.predict_proba(valid_pixels_scaled)

print(f"   Gradient Boosting prediction...")
gb_pred = gb_model.predict(valid_pixels_scaled)
gb_proba = gb_model.predict_proba(valid_pixels_scaled)

# Create output rasters
print(f"\n6. Creating flood susceptibility maps...")

# Initialize output arrays with NaN
rf_susceptibility = np.full((height, width), np.nan, dtype=np.float32)
gb_susceptibility = np.full((height, width), np.nan, dtype=np.float32)
rf_class = np.full((height, width), np.nan, dtype=np.float32)
gb_class = np.full((height, width), np.nan, dtype=np.float32)

# Fill valid pixels
flat_indices = np.arange(n_pixels)[valid_mask]
rf_susceptibility.flat[flat_indices] = rf_pred
gb_susceptibility.flat[flat_indices] = gb_pred
rf_class.flat[flat_indices] = rf_pred
gb_class.flat[flat_indices] = gb_pred

# Also create probability maps for "Very High" risk
rf_very_high_prob = np.full((height, width), np.nan, dtype=np.float32)
gb_very_high_prob = np.full((height, width), np.nan, dtype=np.float32)
rf_very_high_prob.flat[flat_indices] = rf_proba[:, 3]  # Class 3 = Very High
gb_very_high_prob.flat[flat_indices] = gb_proba[:, 3]

# Save susceptibility maps
print(f"\n7. Saving susceptibility rasters...")

# Save Gradient Boosting classification map
with rasterio.open(
    'outputs/flood_susceptibility_gb_class.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=np.float32,
    crs=crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(gb_class, 1)
print(f"   Saved: outputs/flood_susceptibility_gb_class.tif")

# Save Gradient Boosting probability map
with rasterio.open(
    'outputs/flood_susceptibility_gb_prob.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=np.float32,
    crs=crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(gb_very_high_prob, 1)
print(f"   Saved: outputs/flood_susceptibility_gb_prob.tif")

# Save Random Forest classification map
with rasterio.open(
    'outputs/flood_susceptibility_rf_class.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=np.float32,
    crs=crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(rf_class, 1)
print(f"   Saved: outputs/flood_susceptibility_rf_class.tif")

# Calculate statistics
print(f"\n8. Calculating risk statistics...")

# Count pixels per risk class
risk_labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

gb_counts = {}
rf_counts = {}
for i in range(4):
    gb_counts[risk_labels[i]] = np.sum(gb_pred == i)
    rf_counts[risk_labels[i]] = np.sum(rf_pred == i)

print(f"\n   Gradient Boosting Classification:")
total_valid = valid_pixels.shape[0]
for label, count in gb_counts.items():
    pct = 100 * count / total_valid
    print(f"   {label:12s}: {count:8,} pixels ({pct:5.2f}%)")

print(f"\n   Random Forest Classification:")
for label, count in rf_counts.items():
    pct = 100 * count / total_valid
    print(f"   {label:12s}: {count:8,} pixels ({pct:5.2f}%)")

# Load buildings for exposure analysis
print(f"\n9. Calculating exposure statistics...")
buildings = gpd.read_file('backup/01_raw_data/osm/gis_osm_buildings_a_free_1.shp')
print(f"   Loaded {len(buildings):,} buildings")

# Sample building locations on the susceptibility map
from rasterio.transform import rowcol
building_risks_gb = []
building_risks_rf = []

for idx, building in buildings.iterrows():
    x, y = building.geometry.centroid.x, building.geometry.centroid.y
    row, col = rowcol(transform, x, y)
    if 0 <= row < height and 0 <= col < width:
        gb_val = gb_class[row, col]
        rf_val = rf_class[row, col]
        if not np.isnan(gb_val):
            building_risks_gb.append(int(gb_val))
            building_risks_rf.append(int(rf_val))

building_exposure_gb = {}
building_exposure_rf = {}
for i in range(4):
    building_exposure_gb[risk_labels[i]] = building_risks_gb.count(i)
    building_exposure_rf[risk_labels[i]] = building_risks_rf.count(i)

print(f"\n   Building Exposure (Gradient Boosting):")
for label, count in building_exposure_gb.items():
    pct = 100 * count / len(building_risks_gb) if len(building_risks_gb) > 0 else 0
    print(f"   {label:12s}: {count:6,} buildings ({pct:5.2f}%)")

# Generate visualizations
print(f"\n10. Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Flood Susceptibility Maps - Gafsa Governorate', fontsize=18, fontweight='bold', y=0.995)

# Custom colormap for risk classes
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # Green, Yellow, Orange, Red
cmap = LinearSegmentedColormap.from_list('risk', colors, N=4)

# Gradient Boosting Classification
im1 = axes[0, 0].imshow(gb_class, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
axes[0, 0].set_title('Gradient Boosting - Risk Classification (95% Accuracy)', fontweight='bold', fontsize=12)
axes[0, 0].axis('off')
cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, ticks=[0.375, 1.125, 1.875, 2.625])
cbar1.ax.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])

# Gradient Boosting Probability
im2 = axes[0, 1].imshow(gb_very_high_prob, cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
axes[0, 1].set_title('Gradient Boosting - Very High Risk Probability', fontweight='bold', fontsize=12)
axes[0, 1].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
cbar2.set_label('Probability', rotation=270, labelpad=15)

# Random Forest Classification
im3 = axes[1, 0].imshow(rf_class, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
axes[1, 0].set_title('Random Forest - Risk Classification (92% Accuracy)', fontweight='bold', fontsize=12)
axes[1, 0].axis('off')
cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[0.375, 1.125, 1.875, 2.625])
cbar3.ax.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])

# Statistics comparison
risk_categories = list(risk_labels.values())
gb_percentages = [100 * gb_counts[cat] / total_valid for cat in risk_categories]
rf_percentages = [100 * rf_counts[cat] / total_valid for cat in risk_categories]

x = np.arange(len(risk_categories))
width = 0.35
axes[1, 1].bar(x - width/2, gb_percentages, width, label='Gradient Boosting', color='darkgreen', alpha=0.8)
axes[1, 1].bar(x + width/2, rf_percentages, width, label='Random Forest', color='steelblue', alpha=0.8)
axes[1, 1].set_xlabel('Risk Category', fontweight='bold')
axes[1, 1].set_ylabel('Area Coverage (%)', fontweight='bold')
axes[1, 1].set_title('Risk Distribution Comparison', fontweight='bold', fontsize=12)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(risk_categories)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/flood_susceptibility_maps.png', dpi=300, bbox_inches='tight')
print(f"   Saved: outputs/flood_susceptibility_maps.png")
plt.close()

# Create detailed building exposure plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Building Exposure to Flood Risk', fontsize=14, fontweight='bold')

# Gradient Boosting
gb_exposure_pct = [100 * building_exposure_gb[cat] / len(building_risks_gb) if len(building_risks_gb) > 0 else 0 
                   for cat in risk_categories]
axes[0].bar(risk_categories, gb_exposure_pct, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_title('Gradient Boosting Model', fontweight='bold')
axes[0].set_ylabel('Buildings (%)', fontweight='bold')
axes[0].set_ylim([0, max(gb_exposure_pct) * 1.1])
for i, (cat, val) in enumerate(zip(risk_categories, gb_exposure_pct)):
    count = building_exposure_gb[cat]
    axes[0].text(i, val + 1, f'{count:,}\n({val:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Random Forest
rf_exposure_pct = [100 * building_exposure_rf[cat] / len(building_risks_rf) if len(building_risks_rf) > 0 else 0 
                   for cat in risk_categories]
axes[1].bar(risk_categories, rf_exposure_pct, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_title('Random Forest Model', fontweight='bold')
axes[1].set_ylabel('Buildings (%)', fontweight='bold')
axes[1].set_ylim([0, max(rf_exposure_pct) * 1.1])
for i, (cat, val) in enumerate(zip(risk_categories, rf_exposure_pct)):
    count = building_exposure_rf[cat]
    axes[1].text(i, val + 1, f'{count:,}\n({val:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/building_exposure_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: outputs/building_exposure_analysis.png")
plt.close()

# Save statistics to CSV
print(f"\n11. Saving statistics...")
stats_df = pd.DataFrame({
    'Risk_Category': risk_categories,
    'GB_Pixels': [gb_counts[cat] for cat in risk_categories],
    'GB_Percentage': gb_percentages,
    'RF_Pixels': [rf_counts[cat] for cat in risk_categories],
    'RF_Percentage': rf_percentages,
    'GB_Buildings': [building_exposure_gb[cat] for cat in risk_categories],
    'RF_Buildings': [building_exposure_rf[cat] for cat in risk_categories]
})
stats_df.to_csv('outputs/flood_susceptibility_statistics.csv', index=False)
print(f"   Saved: outputs/flood_susceptibility_statistics.csv")

print(f"\n{'='*80}")
print(f"FLOOD SUSCEPTIBILITY MAPPING COMPLETE!")
print(f"{'='*80}")
print(f"\nKey Results (Gradient Boosting - Best Model):")
print(f"  • High + Very High Risk Area: {gb_percentages[2] + gb_percentages[3]:.1f}% of study area")
print(f"  • Buildings in High/Very High Risk: {building_exposure_gb['High'] + building_exposure_gb['Very High']:,}")
print(f"  • Total valid predictions: {total_valid:,} pixels")
print(f"\nNext step: Validate against historical flood events")
