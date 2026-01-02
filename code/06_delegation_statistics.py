import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FLOOD RISK ANALYSIS BY DELEGATION")
print("=" * 80)

# Load Gafsa delegations
print("\n1. Loading Gafsa delegation boundaries...")
delegations_all = gpd.read_file('backup/01_raw_data/administrative/delegations-full.geojson')

# Filter for Gafsa governorate - try both French and Arabic columns
gafsa_delegations = delegations_all[
    (delegations_all['gov_name_f'].str.contains('Gafsa', case=False, na=False)) |
    (delegations_all['gov_name_a'].str.contains('قفصة', case=False, na=False))
].copy()

# Create name translation mapping (Arabic to French/English)
name_translation = {
    'المتلوي': 'Metlaoui',
    'أم العرائس': 'Oum El Araies',
    'بلخير': 'Belkhir',
    'قفصة الشمالية': 'Gafsa Nord',
    'قفصة الجنوبية': 'Gafsa Sud',
    'القطار': 'El Guettar',
    'القصر': 'El Ksar',
    'المضيلة': 'Mdhila',
    'الرديف': 'Redeyef',
    'السند': 'Sned',
    'سيدي عيش': 'Sidi Aich'
}

# Add French/English name column
gafsa_delegations['deleg_name_fr'] = gafsa_delegations['deleg_name'].map(name_translation).fillna(gafsa_delegations['deleg_name'])

print(f"   Found {len(gafsa_delegations)} delegations in Gafsa governorate")
print(f"   Delegations: {', '.join(gafsa_delegations['deleg_name_fr'].unique())}")

# Load susceptibility map
print("\n2. Loading flood susceptibility map...")
with rasterio.open('outputs/flood_susceptibility_gb_class.tif') as src:
    susceptibility = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    profile = src.profile

print(f"   Map CRS: {crs}")
print(f"   Map bounds: {bounds}")

# Ensure CRS match
if gafsa_delegations.crs != crs:
    print(f"   Reprojecting delegations from {gafsa_delegations.crs} to {crs}")
    gafsa_delegations = gafsa_delegations.to_crs(crs)

# Load buildings
print("\n3. Loading buildings data...")
buildings = gpd.read_file('backup/01_raw_data/osm/gis_osm_buildings_a_free_1.shp')
if buildings.crs != crs:
    buildings = buildings.to_crs(crs)
print(f"   Loaded {len(buildings):,} buildings")

# Calculate statistics for each delegation
print("\n4. Calculating risk statistics for each delegation...")

delegation_stats = []
risk_labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

for idx, delegation in gafsa_delegations.iterrows():
    deleg_name = delegation['deleg_name']
    deleg_name_fr = delegation['deleg_name_fr']
    deleg_geom = delegation.geometry
    
    print(f"\n   Processing: {deleg_name_fr}")
    
    # Clip susceptibility raster to delegation
    try:
        clipped, clipped_transform = mask(
            rasterio.open('outputs/flood_susceptibility_gb_class.tif'),
            [deleg_geom],
            crop=True,
            nodata=np.nan
        )
        clipped_data = clipped[0]
        
        # Calculate pixel counts for each risk level
        valid_pixels = clipped_data[~np.isnan(clipped_data)]
        total_pixels = len(valid_pixels)
        
        if total_pixels > 0:
            risk_counts = {}
            risk_percentages = {}
            
            for risk_val, risk_name in risk_labels.items():
                count = np.sum(valid_pixels == risk_val)
                percentage = 100 * count / total_pixels
                risk_counts[risk_name] = int(count)
                risk_percentages[risk_name] = percentage
            
            # Calculate buildings in this delegation
            buildings_in_deleg = buildings[buildings.within(deleg_geom)]
            total_buildings = len(buildings_in_deleg)
            
            # Sample building risk levels
            building_risk_counts = {label: 0 for label in risk_labels.values()}
            
            if total_buildings > 0:
                for _, building in buildings_in_deleg.iterrows():
                    centroid = building.geometry.centroid
                    row = int((bounds.top - centroid.y) / abs(profile['transform'][4]))
                    col = int((centroid.x - bounds.left) / abs(profile['transform'][0]))
                    
                    if 0 <= row < susceptibility.shape[0] and 0 <= col < susceptibility.shape[1]:
                        risk_val = susceptibility[row, col]
                        if not np.isnan(risk_val):
                            building_risk_counts[risk_labels[int(risk_val)]] += 1
            
            # Calculate composite risk score (weighted average)
            risk_score = (
                risk_percentages['Low'] * 0.25 +
                risk_percentages['Medium'] * 0.5 +
                risk_percentages['High'] * 0.75 +
                risk_percentages['Very High'] * 1.0
            ) / 100
            
            delegation_stats.append({
                'Delegation': deleg_name_fr,
                'Delegation_AR': deleg_name,
                'Total_Pixels': total_pixels,
                'Total_Buildings': total_buildings,
                'Low_Pixels': risk_counts['Low'],
                'Medium_Pixels': risk_counts['Medium'],
                'High_Pixels': risk_counts['High'],
                'VeryHigh_Pixels': risk_counts['Very High'],
                'Low_Percent': risk_percentages['Low'],
                'Medium_Percent': risk_percentages['Medium'],
                'High_Percent': risk_percentages['High'],
                'VeryHigh_Percent': risk_percentages['Very High'],
                'Buildings_Low': building_risk_counts['Low'],
                'Buildings_Medium': building_risk_counts['Medium'],
                'Buildings_High': building_risk_counts['High'],
                'Buildings_VeryHigh': building_risk_counts['Very High'],
                'Risk_Score': risk_score,
                'High_VeryHigh_Percent': risk_percentages['High'] + risk_percentages['Very High']
            })
            
            print(f"      Pixels: {total_pixels:,} | High/Very High: {risk_percentages['High'] + risk_percentages['Very High']:.1f}%")
            print(f"      Buildings: {total_buildings:,} | At Risk: {building_risk_counts['High'] + building_risk_counts['Very High']:,}")
        
    except Exception as e:
        print(f"      Error processing {deleg_name_fr}: {e}")

# Create DataFrame
stats_df = pd.DataFrame(delegation_stats)
stats_df = stats_df.sort_values('Risk_Score', ascending=False)

print("\n5. Generating summary statistics...")
print(f"\n{'='*80}")
print("DELEGATION RISK RANKING (Highest to Lowest Risk)")
print(f"{'='*80}\n")

for idx, row in stats_df.iterrows():
    print(f"{row['Delegation']:25s} | Risk Score: {row['Risk_Score']:.3f} | "
          f"High/Very High: {row['High_VeryHigh_Percent']:5.1f}% | "
          f"Buildings at Risk: {row['Buildings_High'] + row['Buildings_VeryHigh']:,}")

# Save detailed statistics
stats_df.to_csv('outputs/delegation_risk_statistics.csv', index=False)
print(f"\n   Saved: outputs/delegation_risk_statistics.csv")

# Create visualizations
print("\n6. Generating visualizations...")

# Merge statistics with geometries
gafsa_delegations = gafsa_delegations.merge(
    stats_df, 
    left_on='deleg_name_fr', 
    right_on='Delegation', 
    how='left'
)

# Figure 1: Choropleth maps
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle('Flood Risk Analysis by Delegation - Gafsa Governorate', 
             fontsize=16, fontweight='bold', y=0.995)

# Map 1: Risk Score
ax1 = axes[0, 0]
gafsa_delegations.plot(column='Risk_Score', 
                       cmap='YlOrRd', 
                       legend=True,
                       edgecolor='black',
                       linewidth=1.5,
                       ax=ax1,
                       legend_kwds={'label': 'Risk Score (0-1)', 'orientation': 'vertical'})
ax1.set_title('Composite Risk Score by Delegation', fontweight='bold', fontsize=12)
ax1.axis('off')

# Add delegation labels
for idx, row in gafsa_delegations.iterrows():
    if pd.notna(row['Delegation']):
        centroid = row.geometry.centroid
        ax1.annotate(row['Delegation'], 
                    xy=(centroid.x, centroid.y), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Map 2: High/Very High Risk Percentage
ax2 = axes[0, 1]
gafsa_delegations.plot(column='High_VeryHigh_Percent', 
                       cmap='Reds', 
                       legend=True,
                       edgecolor='black',
                       linewidth=1.5,
                       ax=ax2,
                       legend_kwds={'label': 'High/Very High Risk (%)', 'orientation': 'vertical'})
ax2.set_title('High/Very High Risk Area Coverage', fontweight='bold', fontsize=12)
ax2.axis('off')

for idx, row in gafsa_delegations.iterrows():
    if pd.notna(row['High_VeryHigh_Percent']):
        centroid = row.geometry.centroid
        ax2.annotate(f"{row['High_VeryHigh_Percent']:.1f}%", 
                    xy=(centroid.x, centroid.y), 
                    ha='center', 
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# Chart 3: Buildings at risk by delegation
ax3 = axes[1, 0]
deleg_sorted = stats_df.sort_values('Risk_Score', ascending=True)
y_pos = np.arange(len(deleg_sorted))

building_data = deleg_sorted[['Buildings_Medium', 'Buildings_High', 'Buildings_VeryHigh']].values
colors = ['#f1c40f', '#e67e22', '#e74c3c']
labels = ['Medium Risk', 'High Risk', 'Very High Risk']

left = np.zeros(len(deleg_sorted))
for i, (data, color, label) in enumerate(zip(building_data.T, colors, labels)):
    ax3.barh(y_pos, data, left=left, color=color, label=label, edgecolor='black')
    left += data

ax3.set_yticks(y_pos)
ax3.set_yticklabels(deleg_sorted['Delegation'], fontsize=9)
ax3.set_xlabel('Number of Buildings', fontweight='bold')
ax3.set_title('Buildings Exposure by Risk Level', fontweight='bold', fontsize=12)
ax3.legend(loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# Chart 4: Area distribution by risk level
ax4 = axes[1, 1]
deleg_sorted2 = stats_df.sort_values('High_VeryHigh_Percent', ascending=True)
y_pos2 = np.arange(len(deleg_sorted2))

area_data = deleg_sorted2[['Low_Percent', 'Medium_Percent', 'High_Percent', 'VeryHigh_Percent']].values
colors2 = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
labels2 = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']

left2 = np.zeros(len(deleg_sorted2))
for i, (data, color, label) in enumerate(zip(area_data.T, colors2, labels2)):
    ax4.barh(y_pos2, data, left=left2, color=color, label=label, edgecolor='black')
    left2 += data

ax4.set_yticks(y_pos2)
ax4.set_yticklabels(deleg_sorted2['Delegation'], fontsize=9)
ax4.set_xlabel('Area Coverage (%)', fontweight='bold')
ax4.set_title('Area Distribution by Risk Level', fontweight='bold', fontsize=12)
ax4.legend(loc='lower right')
ax4.grid(axis='x', alpha=0.3)
ax4.set_xlim([0, 100])

plt.tight_layout()
plt.savefig('outputs/delegation_risk_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: outputs/delegation_risk_analysis.png")
plt.close()

# Figure 2: Priority delegations
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle('Priority Delegations for Flood Mitigation - Gafsa Governorate', 
             fontsize=14, fontweight='bold')

top_delegations = stats_df.head(5)
x = np.arange(len(top_delegations))
width = 0.35

bars1 = ax.bar(x - width/2, top_delegations['High_VeryHigh_Percent'], width, 
               label='High/Very High Risk Area (%)', color='#e74c3c', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, top_delegations['Buildings_High'] + top_delegations['Buildings_VeryHigh'], 
                width, label='Buildings at Risk', color='#3498db', alpha=0.8)

ax.set_xlabel('Delegation', fontweight='bold', fontsize=11)
ax.set_ylabel('High/Very High Risk Area (%)', fontweight='bold', fontsize=11, color='#e74c3c')
ax2.set_ylabel('Buildings at High Risk', fontweight='bold', fontsize=11, color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(top_delegations['Delegation'], rotation=45, ha='right')
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#3498db')

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, top_delegations['High_VeryHigh_Percent'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

for i, (bar, val) in enumerate(zip(bars2, top_delegations['Buildings_High'] + top_delegations['Buildings_VeryHigh'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f'{int(val):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/priority_delegations.png', dpi=300, bbox_inches='tight')
print(f"   Saved: outputs/priority_delegations.png")
plt.close()

# Generate summary report
print("\n7. Creating summary report...")

report = f"""# Flood Risk Analysis by Delegation - Gafsa Governorate

## Executive Summary

This analysis breaks down flood risk across all {len(stats_df)} delegations in Gafsa Governorate.

### Top 5 Priority Delegations (Highest Risk)

| Rank | Delegation | Risk Score | High/Very High Area (%) | Buildings at Risk |
|------|------------|------------|-------------------------|-------------------|
"""

for rank, (idx, row) in enumerate(stats_df.head(5).iterrows(), 1):
    buildings_at_risk = row['Buildings_High'] + row['Buildings_VeryHigh']
    report += f"| {rank} | {row['Delegation']:20s} | {row['Risk_Score']:.3f} | {row['High_VeryHigh_Percent']:6.1f}% | {buildings_at_risk:,} |\n"

report += f"""
### Risk Distribution Summary

**Total Area Analyzed:** {stats_df['Total_Pixels'].sum():,} pixels
**Total Buildings:** {stats_df['Total_Buildings'].sum():,} buildings

#### Delegation-Level Statistics

| Delegation | Total Area (pixels) | Low Risk (%) | Medium (%) | High (%) | Very High (%) | Buildings |
|------------|---------------------|--------------|------------|----------|---------------|-----------|
"""

for idx, row in stats_df.iterrows():
    report += f"| {row['Delegation']:20s} | {row['Total_Pixels']:,} | {row['Low_Percent']:5.1f} | {row['Medium_Percent']:5.1f} | {row['High_Percent']:5.1f} | {row['VeryHigh_Percent']:5.1f} | {row['Total_Buildings']:,} |\n"

report += f"""
### Key Findings

1. **Highest Risk Delegation:** {stats_df.iloc[0]['Delegation']} 
   - Risk Score: {stats_df.iloc[0]['Risk_Score']:.3f}
   - High/Very High Risk Coverage: {stats_df.iloc[0]['High_VeryHigh_Percent']:.1f}%
   - Buildings at Risk: {stats_df.iloc[0]['Buildings_High'] + stats_df.iloc[0]['Buildings_VeryHigh']:,}

2. **Safest Delegation:** {stats_df.iloc[-1]['Delegation']}
   - Risk Score: {stats_df.iloc[-1]['Risk_Score']:.3f}
   - High/Very High Risk Coverage: {stats_df.iloc[-1]['High_VeryHigh_Percent']:.1f}%

3. **Regional Pattern:**
   - Average High/Very High Risk Coverage: {stats_df['High_VeryHigh_Percent'].mean():.1f}%
   - Standard Deviation: {stats_df['High_VeryHigh_Percent'].std():.1f}%

### Recommendations by Priority

#### Priority 1: Immediate Action Required
**Delegations:** {', '.join(stats_df.head(2)['Delegation'].tolist())}
- Install early warning systems
- Develop emergency evacuation plans
- Conduct infrastructure vulnerability assessment

#### Priority 2: Medium-Term Planning
**Delegations:** {', '.join(stats_df.iloc[2:5]['Delegation'].tolist())}
- Improve drainage infrastructure
- Implement flood mitigation measures
- Establish emergency response protocols

#### Priority 3: Monitoring and Prevention
**Remaining delegations**
- Regular risk assessment updates
- Public awareness campaigns
- Land use planning restrictions in vulnerable zones

---
*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Model: Gradient Boosting (95% accuracy)*
"""

with open('outputs/delegation_risk_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"   Saved: outputs/delegation_risk_report.md")

print(f"\n{'='*80}")
print(f"DELEGATION ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\n✓ Analyzed {len(stats_df)} delegations")
print(f"✓ Processed {stats_df['Total_Pixels'].sum():,} pixels")
print(f"✓ Assessed {stats_df['Total_Buildings'].sum():,} buildings")
print(f"\nOutputs:")
print(f"  • delegation_risk_statistics.csv - Detailed statistics")
print(f"  • delegation_risk_analysis.png - Choropleth maps and charts")
print(f"  • priority_delegations.png - Top 5 priority visualization")
print(f"  • delegation_risk_report.md - Summary report")
