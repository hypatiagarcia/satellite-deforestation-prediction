# src/data/download_data.py
import ee
import os
import time
import yaml # Import YAML library
import argparse # Import argparse for config file path
from pathlib import Path # Use Pathlib for paths

# Import project utilities
from utils import ensure_dir, init_earth_engine, generate_tiles, maskS2clouds

# --- GEE Data Fetching Functions ---
# (These functions remain the same as before)
def get_sentinel2_composite(region: ee.Geometry, start_date: str, end_date: str, s2_bands: list) -> ee.Image:
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate(start_date, end_date)
                 .map(maskS2clouds)) # Assuming maskS2clouds is defined in utils
    s2_median = s2_sr_col.median()
    # Ensure correct bands are selected and named
    return s2_median.select(s2_bands, s2_bands)

def get_sentinel1_composite(region: ee.Geometry, start_date: str, end_date: str, s1_bands: list) -> ee.Image:
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              # Ensure required polarizations exist before selecting
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', s1_bands[0]))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', s1_bands[1]))
              .select(s1_bands))
    s1_median = s1_col.median()
    # Ensure correct bands are selected and named
    return s1_median.select(s1_bands, s1_bands)

def get_dem(region: ee.Geometry, dem_bands: list) -> ee.Image:
    dem = ee.Image('NASA/NASADEM_HGT/001').select(dem_bands[0]) # Select the single DEM band
    # Ensure correct band is selected and named
    return dem.select(dem_bands, dem_bands)

def get_hansen_labels(region: ee.Geometry, loss_year: int, hansen_version: str, label_bands: list) -> ee.Image:
    hansen = ee.Image(hansen_version)
    target_loss_code = loss_year - 2000 # Assumes standard Hansen lossyear encoding
    loss_mask = hansen.select('lossyear').eq(target_loss_code).rename(label_bands[0]) # Rename to target name
    # Ensure correct band is selected and named
    return loss_mask.unmask(0).select(label_bands, label_bands)
# --- End GEE Functions ---


# --- Main Execution Logic ---
if __name__ == '__main__':
    # --- Argument Parsing for Config File ---
    parser = argparse.ArgumentParser(description="Download Stacked Satellite Data from GEE")
    parser.add_argument("--config", type=str, required=True, help="Path to the data download configuration YAML file")
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        exit()

    # --- Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize Earth Engine ---
    # Make sure PROJECT_ID is available, either from config or environment
    # project_id = config.get('project_id', os.environ.get('GOOGLE_CLOUD_PROJECT'))
    # if not project_id:
    #     print("Error: Google Cloud Project ID not found in config or environment.")
    #     exit()
    init_earth_engine() # Assuming init_earth_engine handles authentication/project

    # --- Generate Tiles for the LARGER Region ---
    region_coords = config['region_coords']
    tile_size_deg = config['tile_size_deg']
    tiles = generate_tiles(region_coords, tile_size_deg=tile_size_deg) # generate_tiles from utils
    print(f"Generated {len(tiles)} tiles to export for region {region_coords}.")
    if len(tiles) == 0:
        print("Error: No tiles generated. Check region coordinates and tile size.")
        exit()
    if len(tiles) > 3000: # GEE Task Limit Warning
         print(f"Warning: Generated {len(tiles)} tiles, which may exceed GEE's concurrent task limit (around 3000).")
         print("         Consider using larger tile_size_deg or splitting the region if exports fail.")


    # --- Prepare for Export ---
    gdrive_folder_name = config['gdrive_folder_name']
    task_delay = config['export_task_delay_seconds']
    ee_crs = config['ee_crs'] # You likely need to fix these too
    ee_scale = config['ee_scale']
    feature_start_date = config['feature_start_date']
    feature_end_date = config['feature_end_date']
    label_year = config['label_year']
    hansen_version = config['hansen_version']
    s2_bands = config['s2_bands']
    s1_bands = config['s1_bands']
    dem_bands = config['dem_bands']
    label_bands = config['label_bands']

    print(f"\n--- Starting GEE Export ---")
    print(f"Target Google Drive Folder: '{gdrive_folder_name}'")
    print(f"Delay between tasks: {task_delay} seconds")

    submitted_tasks = 0
    failed_submissions = 0

    # --- Loop Through Tiles and Submit Export Tasks ---
    for idx, tile_coords in enumerate(tiles):
        print("-" * 50)
        print(f"Processing tile {idx + 1}/{len(tiles)}: {tile_coords}")

        # Define geometry for the current tile
        tile_geom = ee.Geometry.Rectangle(tile_coords, proj=ee_crs, geodesic=False)

        # Define unique filename prefix and task description
        # Using index ensures uniqueness even if dates are the same across runs
        file_name = f"stack_tile_{idx}_{feature_start_date}_{label_year}"
        task_description = f"Export_Stack_Tile_{idx}" # Keep description concise

        try:
            # --- Fetch and Stack Image Components ---
            # print("  Getting Sentinel-2 composite...") # Less verbose logging
            s2_img = get_sentinel2_composite(tile_geom, feature_start_date, feature_end_date, s2_bands)
            # print("  Getting Sentinel-1 composite...")
            s1_img = get_sentinel1_composite(tile_geom, feature_start_date, feature_end_date, s1_bands)
            # print("  Getting DEM...")
            dem_img = get_dem(tile_geom, dem_bands)
            # print("  Getting Hansen labels...")
            label_img = get_hansen_labels(tile_geom, label_year, hansen_version, label_bands)

            # print("  Stacking bands...")
            # Ensure band order matches intended mapping if critical downstream
            # Order: S1, S2, DEM, Label (matches typical processing)
            stacked_image = s1_img.addBands(s2_img).addBands(dem_img).addBands(label_img)
            stacked_image = stacked_image.toFloat() # Consistent data type

            # --- Create and Start Export Task ---
            task = ee.batch.Export.image.toDrive(
                image=stacked_image,
                description=task_description,
                folder=gdrive_folder_name,
                fileNamePrefix=file_name,
                region=tile_geom,
                scale=ee_scale,
                crs=ee_crs,
                maxPixels=1e11, # Increase further for potentially larger tiles if needed
                fileFormat='GeoTIFF',
                # shardSize=256 # Consider adding if individual TIFs become >2GB
            )

            task.start()
            print(f"  Submitted Task: {task_description} (ID: {task.id})")
            submitted_tasks += 1

            # --- Apply Increased Delay ---
            # print(f"  Waiting {task_delay} seconds...")
            time.sleep(task_delay)

        except ee.EEException as e:
            print(f"[!] GEE Error submitting task for tile {idx + 1}: {e}")
            failed_submissions += 1
            # Optional: Add a longer sleep after an error to let GEE recover
            # time.sleep(task_delay * 2)
        except Exception as e:
            print(f"[!] Unexpected Error processing tile {idx + 1}: {e}")
            failed_submissions += 1
            # time.sleep(task_delay * 2)

    # --- Final Summary ---
    print("-" * 50)
    print("Task Submission Summary:")
    print(f"  Tasks Submitted: {submitted_tasks}")
    print(f"  Failed Submissions: {failed_submissions}")
    print(f"  Total Tiles Attempted: {len(tiles)}")
    print(f"\nCheck the 'Tasks' tab in the GEE Code Editor or your Google Drive folder '{gdrive_folder_name}' for export progress.")
    if failed_submissions > 0:
         print("Warning: Some tasks failed during submission. Check logs above.")