"""
Script to generate synthetic flood dataset
"""
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 5000

# Features
rainfall_mm = np.random.normal(50, 40, n_samples)  # Mean 50mm, std 40mm
rainfall_mm = np.clip(rainfall_mm, 0, 300)  # Clip to realistic range

river_level_m = np.random.normal(2.5, 1.2, n_samples)  # Mean 2.5m, std 1.2m
river_level_m = np.clip(river_level_m, 0.5, 8)

soil_moisture_percent = np.random.normal(55, 20, n_samples)
soil_moisture_percent = np.clip(soil_moisture_percent, 10, 100)

temperature_c = np.random.normal(20, 10, n_samples)
temperature_c = np.clip(temperature_c, -5, 45)

humidity_percent = np.random.normal(60, 20, n_samples)
humidity_percent = np.clip(humidity_percent, 20, 100)

elevation_m = np.random.normal(500, 300, n_samples)
elevation_m = np.clip(elevation_m, 50, 2000)

drainage_capacity_index = np.random.uniform(1, 10, n_samples)

# Create flood risk based on probabilistic logic
# Higher rainfall, river level, soil moisture → higher flood risk
# Higher elevation, drainage capacity → lower flood risk
flood_probability = (
    (rainfall_mm / 300) * 0.3 +  # 30% weight on rainfall
    (river_level_m / 8) * 0.3 +   # 30% weight on river level
    (soil_moisture_percent / 100) * 0.15 +  # 15% weight on soil moisture
    (humidity_percent / 100) * 0.1 -  # 10% weight on humidity (but inverse)
    (elevation_m / 2000) * 0.1 -   # 10% weight on elevation (negative)
    (drainage_capacity_index / 10) * 0.05  # 5% weight on drainage (negative)
)

flood_probability = np.clip(flood_probability, 0, 1)

# Generate binary flood risk with probabilistic threshold
flood_risk = np.random.binomial(1, flood_probability, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'rainfall_mm': rainfall_mm,
    'river_level_m': river_level_m,
    'soil_moisture_percent': soil_moisture_percent,
    'temperature_c': temperature_c,
    'humidity_percent': humidity_percent,
    'elevation_m': elevation_m,
    'drainage_capacity_index': drainage_capacity_index,
    'flood_risk': flood_risk
})

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'data', 'flood_data.csv')
data.to_csv(output_path, index=False)

print(f"✅ Dataset generated successfully!")
print(f"📊 Dataset shape: {data.shape}")
print(f"📁 Saved to: {output_path}")
print(f"\nDataset Preview:")
print(data.head())
print(f"\nFlood Risk Distribution:")
print(data['flood_risk'].value_counts())
