#!/usr/bin/env python
# coding: utf-8

# In[1]:


######### Import Libraries ####################

## EE Libraries ##
import ee
import geemap
ee.Authenticate()
ee.Initialize()

## Other Libraries ##
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd



# In[ ]:


###########TAI EXPORT over all ROIs ###################

# Load a Sentinel-2 image collection
dataset = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2017-12-31', '2018-01-01')

# Define a function to calculate TAI for each image
def applyTAI(image):
    TAI = image.select('B12').subtract(image.select('B11')).divide(image.select('B8A')) \
        .rename('TAI')
    return image.addBands(TAI)

# Apply the applyTAI function to each image in the ImageCollection
TAIcollection = dataset.map(applyTAI)

# List of GeoJSON file paths
geojson_paths = [
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Dunkerque.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\IJmuiden.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Mundra.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Rostock.geojson'
]



# Loop through each GeoJSON file
for geojson_path in geojson_paths:
    # Load the GeoJSON file using geemap
    ee_object = geemap.geojson_to_ee(geojson_path)

    # Create a FeatureCollection from the GeoJSON
    roi_fc = ee.FeatureCollection(ee_object)

    # Filter the Sentinel-2 image collection based on the GeoJSON geometry
    dataset_roi = dataset.filterBounds(roi_fc)

    # Apply the TAI function to each image in the filtered collection
    TAIcollection_roi = dataset_roi.map(applyTAI)

    # Clip and export each image in the TAIcollection_roi
    for i in range(TAIcollection_roi.size().getInfo()):
        image = ee.Image(TAIcollection_roi.toList(TAIcollection_roi.size()).get(i))

        # Clip the image to the ROI
        clipped_image = image.clip(roi_fc).select('TAI')

        # Get the name of the ROI from the GeoJSON file path
        roi_name = geojson_path.split('\\')[-1].split('.')[0]

        # Export the clipped TAI image to Google Drive within the specified subfolder
        task = ee.batch.Export.image.toDrive(
            image=clipped_image,
            folder= roi_name,  # Adjust the folder path as needed
            scale=10,
            region=roi_fc.geometry().bounds(),
            fileFormat='GeoTIFF',
            fileNamePrefix=image.get('system:index').getInfo(),
        )

        # Start the export task
        task.start()


# In[ ]:


##################################### S2 EXPORT over all ROIs ################################################
# Load a Sentinel-2 image collection
dataset = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2017-12-01', '2018-01-01')

# List of GeoJSON file paths
geojson_paths = [
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Dunkerque.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\IJmuiden.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Mundra.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Rostock.geojson'
]

# Specify the main folder in Google Drive
main_folder = 'TAI'

# Specify the output subfolder prefix
output_subfolder_prefix = '_S2'

# Loop through each GeoJSON file
for geojson_path in geojson_paths:
    # Load the GeoJSON file using geemap
    ee_object = geemap.geojson_to_ee(geojson_path)

    # Create a FeatureCollection from the GeoJSON
    roi_fc = ee.FeatureCollection(ee_object)

    # Filter the Sentinel-2 image collection based on the GeoJSON geometry
    dataset_roi = dataset.filterBounds(roi_fc)

    # Get the name of the ROI from the GeoJSON file path
    roi_name = geojson_path.split('\\')[-1].split('.')[0]

    # Loop through each image in the filtered collection
    for i in range(dataset_roi.size().getInfo()):
        # Get the image
        image = ee.Image(dataset_roi.toList(dataset_roi.size()).get(i))
        
        image = image.toUint16()
        
        clipped_image = image.clip(roi_fc)

        # Define the export parameters
        export_params = {
            'image': clipped_image,
            'folder': roi_name + output_subfolder_prefix,
            'scale': 10,  # Adjust the scale as needed
            'region': roi_fc.geometry().bounds(),
            'fileNamePrefix':image.get('system:index').getInfo()
        }

        # Export the image to Google Drive
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()

        # Print a message indicating the export task has been submitted
        print(f'Exporting {roi_name}_S2_{i} to {roi_name + output_subfolder_prefix}... Task ID: {task.id}')

print("Export tasks submitted. Please check your Google Drive for the exported images.")


# In[ ]:


################ DISPLAY ROIs ON MAP #######################################
# Load a Sentinel-2 image collection
dataset = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2017-12-01', '2018-01-01')
# Define a function to calculate TAI for each image
def applyTAI(image):
    TAI = image.select('B12').subtract(image.select('B11')).divide(image.select('B8A')) \
        .rename('TAI')
    return image.addBands(TAI)

# Apply the applyTAI function to each image in the ImageCollection
TAIcollection = dataset.map(applyTAI)

# List of GeoJSON file paths
geojson_paths = [
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Dunkerque.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\IJmuiden.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Mundra.geojson',
    r'C:\Users\DELL\OneDrive\Desktop\TAI\Rostock.geojson'
]

# Create a map
Map = geemap.Map()

for geojson_path in geojson_paths:
    # Load the GeoJSON file using geemap
    ee_object = geemap.geojson_to_ee(geojson_path)

    # Create a FeatureCollection from the GeoJSON
    roi_fc = ee.FeatureCollection(ee_object)

    # Filter the Sentinel-2 image collection based on the GeoJSON geometry
    dataset_roi = dataset.filterBounds(roi_fc)

    # Apply the TAI function to each image in the filtered collection
    TAIcollection_roi = dataset_roi.map(applyTAI)

    # Clip the TAI dataset to the ROI and select only the TAI band
    TAIcollection_roi_clipped = TAIcollection_roi.map(lambda image: image.clip(roi_fc).select('TAI'))

    # Visualization parameters for True Color layer
    vis_true_color = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
    }

    # Visualization parameters for False Color layer
    vis_false_color = {
        'bands': ['B12', 'B11', 'B8A'],
        'min': 0,
        'max': 3000,
    }

    # Visualization parameters for TAI layer
    vis_tai = {
        'min': -2,
        'max': 5,
        'palette': ['blue', 'green', 'red'],
    }

    # Clip and display True Color layer
    true_color_roi_clipped = dataset_roi.map(lambda image: image.clip(roi_fc)).select(['B4', 'B3', 'B2'])
    Map.addLayer(true_color_roi_clipped, vis_true_color, 'True Color - ' + geojson_path.split('\\')[-1].split('.')[0])

    # Clip and display False Color layer
    false_color_roi_clipped = dataset_roi.map(lambda image: image.clip(roi_fc)).select(['B12', 'B11', 'B8A'])
    Map.addLayer(false_color_roi_clipped, vis_false_color, 'False Color - ' + geojson_path.split('\\')[-1].split('.')[0])

    # Display TAI layer (already clipped)
    Map.addLayer(TAIcollection_roi_clipped, vis_tai, 'TAI - ' + geojson_path.split('\\')[-1].split('.')[0])

# Center the map on the first ROI
Map.centerObject(roi_fc, 10)

# Add layer control
Map.addLayerControl()

# Display the map
Map


# In[ ]:


############################ EXPORT OVER SINGLE ROI ##################################
# Load a Sentinel-2 image collection
dataset = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2023-01-01', '2024-01-01')

# Load the Gijon region from the GeoJSON file
gijon_geojson_path = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon.geojson'
gijon_ee_object = geemap.geojson_to_ee(gijon_geojson_path)
gijon_roi_fc = ee.FeatureCollection(gijon_ee_object)

# Filter the Sentinel-2 image collection based on the Gijon region
dataset_gijon = dataset.filterBounds(gijon_roi_fc)

# Loop through each image in the filtered collection
for i in range(dataset_gijon.size().getInfo()):
    # Get the image
    image = ee.Image(dataset_gijon.toList(dataset_gijon.size()).get(i))
    
    image = image.toUint16()
    
    clipped_image = image.clip(gijon_roi_fc)

    # Define the export parameters
    export_params = {
        'image': clipped_image,
        'folder': 'GIJON',
        'scale': 10,  # Adjust the scale as needed
        'region': gijon_roi_fc.geometry().bounds(),
        'fileNamePrefix': image.get('system:index').getInfo()
    }

    # Export the image to Google Drive
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()

    # Print a message indicating the export task has been submitted
    print(f'Exporting Gijon_S2_{i} to TIME_SERIES_GIJON... Task ID: {task.id}')

print("Export tasks submitted for Gijon. Please check your Google Drive for the exported images.")


# In[ ]:


################################## OCCURRENCE FREQ ###################################################

# Directory containing the TAI raster stack
tai_dir = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon_TAI'

# List all TAI raster files
tai_files = [os.path.join(tai_dir, f) for f in os.listdir(tai_dir) if f.endswith('.tif')]

# Initialize an empty array to hold TAI values
tai_stack = []

# Read TAI rasters and stack them
for tai_file in tai_files:
    with rasterio.open(tai_file) as src:
        tai_stack.append(src.read(1, masked=True))  # Read with nodata values as masked array

# Convert the list of arrays to a 3D numpy array
tai_stack = np.ma.array(tai_stack, mask=np.isnan(tai_stack))  # Mask invalid values

# Calculate occurrence frequency
occurrence_frequency = np.sum(tai_stack > 1, axis=0)

# Save the occurrence frequency raster
output_file = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon_Occurrence_Frequency.tif'

with rasterio.open(tai_files[0]) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint16, count=1, nodata=0)  # Set nodata value to 0

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(occurrence_frequency.filled(0).astype(rasterio.uint16), 1)  # Fill masked values with 0


# In[ ]:


############################ TIME SERIES USING KERNEL / BOUNDING BOX ###########################################

# Directory containing the TAI raster stack
tai_dir = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon_TAI'

# Bounding box defining the extent of the feature
xmin, ymin, xmax, ymax = 764399.9862972005503252, 4824219.9187874495983124, 764460.1364439337048680, 4824279.9449132625013590

# Initialize lists to store timestamps and average pixel values
timestamps = []
average_values = []

# Loop through TAI rasters
for tai_file in os.listdir(tai_dir):
    if tai_file.endswith('.tif'):
        # Extract timestamp from file name and format it as YYYY-MM-DD
        timestamp = tai_file.split('_')[0][:4] + '-' + tai_file.split('_')[0][4:6] + '-' + tai_file.split('_')[0][6:8]
        timestamps.append(timestamp)

        # Read TAI raster
        with rasterio.open(os.path.join(tai_dir, tai_file)) as src:
            # Define window based on bounding box
            window = src.window(xmin, ymin, xmax, ymax)

            # Read raster values corresponding to the feature
            data = src.read(1, window=window)
            
            # Calculate average value for the feature
            if np.any(data):  # Check if the array is not empty
                average_value = np.mean(data)
            else:
                average_value = np.nan  # Assign NaN if no data is found
            average_values.append(average_value)

# Set the figure size to make the graph wider
plt.figure(figsize=(12, 6))

# Plot the time series
plt.plot(timestamps, average_values, marker='o')
plt.xlabel('Date')
plt.ylabel('Average Pixel Value')
plt.title('Time Series of Average Pixel Value for the Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[2]:


############## TIME SERIES USING CLIPPED RASTERS ##################


#### CLIPPING ####

# Define paths to raster stack and shapefile
raster_folder = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon_TAI'
shapefile_path = r'C:\Users\DELL\OneDrive\Desktop\TAI\TAI_Flare.shp'

# Load the shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Create a folder to store clipped raster files
clipped_folder = r'C:\Users\DELL\OneDrive\Desktop\TAI\TAI_Clipped'
os.makedirs(clipped_folder, exist_ok=True)

# Function to clip raster with the shapefile
def clip_raster(raster_file, shapefile, output_folder):
    with rasterio.open(raster_file) as src:
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta

    # Update metadata for the clipped raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    # Output file path for the clipped raster
    output_file = os.path.join(output_folder, os.path.basename(raster_file))

    # Write the clipped raster to the output file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_image)

# Clip each raster in the raster folder
for raster_file in os.listdir(raster_folder):
    if raster_file.endswith('.tif'):
        clip_raster(os.path.join(raster_folder, raster_file), gdf, clipped_folder)

print("Clipping completed.")

### PLOTTING ###

# Define the folder containing the clipped raster files
clipped_folder = r'C:\Users\DELL\OneDrive\Desktop\TAI\TAI_Clipped'

# List to store mean values
mean_values = []

# Iterate over the clipped raster files
for clipped_file in os.listdir(clipped_folder):
    if clipped_file.endswith('.tif'):
        # Open clipped raster file
        with rasterio.open(os.path.join(clipped_folder, clipped_file)) as src:
            # Read raster data
            raster = src.read(1)
            # Calculate mean value of clipped area
            total = 0
            count = 0
            for row in raster:
                for pixel in row:
                    if not src.profile['nodata'] or pixel != src.profile['nodata']:
                        total += pixel
                        count += 1
            if count > 0:
                mean_value = total / count
                mean_values.append(mean_value)

# Plot mean pixel values
plt.figure(figsize=(10, 6))
plt.plot(mean_values, marker='o', color='b', linestyle='-')
plt.xlabel('Image')
plt.ylabel('Average TAI')
plt.title('TAI TIME SERIES')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[ ]:


############## DETRENDED TIME SERIES #############

import rasterio
import os
import matplotlib.pyplot as plt
from scipy import signal

# Define the folder containing the clipped raster files
clipped_folder = r'C:\Users\DELL\OneDrive\Desktop\TAI\TAI_Clipped'

# List to store mean values
mean_values = []

# Iterate over the clipped raster files
for clipped_file in os.listdir(clipped_folder):
    if clipped_file.endswith('.tif'):
        # Open clipped raster file
        with rasterio.open(os.path.join(clipped_folder, clipped_file)) as src:
            # Read raster data
            raster = src.read(1)
            # Calculate mean value of clipped area
            total = 0
            count = 0
            for row in raster:
                for pixel in row:
                    if not src.profile['nodata'] or pixel != src.profile['nodata']:
                        total += pixel
                        count += 1
            if count > 0:
                mean_value = total / count
                mean_values.append(mean_value)

# De-trend the time series
detrended = signal.detrend(mean_values)

# Plot de-trended time series
plt.figure(figsize=(10, 6))
plt.plot(detrended, marker='o', color='r', linestyle='-')
plt.xlabel('Image')
plt.ylabel('De-trended Time Series')
plt.title('Average TAI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[ ]:


############################## TIME SERIES FOR MULTIPLE FEATURES ###########################
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window

# Directory containing the TAI raster stack
tai_dir = r'C:\Users\DELL\OneDrive\Desktop\TAI\Gijon_TAI'

# Define bounding box coordinates for each feature
bounding_boxes = {
    'Steel Plant': (764479.4993254075525329, 4824939.6912259720265865, 764500.3536948875989765, 4824960.3363542864099145),
    'Coke Oven 1': (764378.4490076476940885, 4824459.4039161987602711, 764421.2331255407771096, 4824520.7898244801908731),
    'Coke Oven 2': (764378.6600540479412302, 4824539.0734332753345370, 764420.9804455315461382, 4824600.6303663421422243),
    'Slag Pit': (763799.6033849872183055, 4824199.8138268338516355, 763880.1574432225897908, 4824240.1263423208147287),
    'Slag Pit 2': (763919.6396984631428495, 4824318.9909333279356360, 763980.5928524849005044, 4824360.5042954292148352)
}

# Initialize a dictionary to store timestamps and average pixel values for each feature
feature_data = {feature: {'timestamps': [], 'average_values': []} for feature in bounding_boxes}

# Loop through TAI rasters
for tai_file in os.listdir(tai_dir):
    if tai_file.endswith('.tif'):
        timestamp = tai_file.split('_')[1][:8]  # Extract YYYYMMDD from file name

        for feature, bbox in bounding_boxes.items():
            xmin, ymin, xmax, ymax = bbox
            with rasterio.open(os.path.join(tai_dir, tai_file)) as src:
                window = src.window(xmin, ymin, xmax, ymax)
                data = src.read(1, window=window)
                
                if np.any(data):
                    average_value = np.mean(data)
                else:
                    average_value = np.nan
                # Format the timestamp as 'YYYY-MM-DD' before appending
                formatted_timestamp = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                feature_data[feature]['timestamps'].append(formatted_timestamp)
                feature_data[feature]['average_values'].append(average_value)

# Plot the time series for each feature
plt.figure(figsize=(10, 6))
for feature, data in feature_data.items():
    plt.plot(data['timestamps'], data['average_values'], marker='o', label=feature)

plt.xlabel('Time')
plt.ylabel('Average Pixel Value')
plt.title('Time Series of Average Pixel Value for Features')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

