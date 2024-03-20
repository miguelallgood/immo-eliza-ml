"""
This snippet of code loads latitude and longitude coordinates from a CSV file,
generates additional data points for apartment properties based on these coordinates,
and saves the updated dataset to a new CSV file.

Attributes:
    df_geo_post_codes (DataFrame): DataFrame containing latitude and longitude coordinates.
    additional_data (dict): Dictionary containing additional data points for apartment properties.
    df_additional (DataFrame): DataFrame containing the generated additional data points.
"""
import pandas as pd
import numpy as np

# Load the dataset containing latitude and longitude coordinates
df_geo_post_codes = pd.read_csv(r'C:\Users\migue\OneDrive\BeCode\immo-eliza-ml\data\GPC-STRT-GEO-SAMPLE-BE.csv', delimiter=';')

# Generate additional data points
additional_data = {
    'number_rooms': np.random.randint(1, 4, size=len(df_geo_post_codes)),
    'living_area': np.random.randint(80, 200, size=len(df_geo_post_codes)),
    'garden_area': np.random.randint(0, 50, size=len(df_geo_post_codes)),
    'number_facades': np.random.randint(1, 4, size=len(df_geo_post_codes)),
    'Longitude': df_geo_post_codes['longitude'],
    'Latitude': df_geo_post_codes['latitude']
}

# Create DataFrame for additional data points
df_additional = pd.DataFrame(additional_data)

# Save the updated DataFrame to a new CSV file
df_additional.to_csv('data/new_apartment_data.csv', index=False)