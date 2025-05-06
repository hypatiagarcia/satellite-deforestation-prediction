# utils.py
import os
import requests
import ee
import numpy as np
import time
from config import PROJECT_ID

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def init_earth_engine():
    """Authenticate and initialize the Earth Engine API once."""
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("GEE Authenticated and Initialized Successfully.")

def download_image_via_ee(image: ee.Image, region: ee.Geometry,
                         scale: int, crs: str, out_path: str,
                         retries: int = 3, wait_seconds: int = 5):
    if os.path.exists(out_path):
        print(f"[✓] Exists: {out_path}")
        return True

    params = {
        'scale': scale,
        'crs': crs,
        'region': region,
        'format': 'GEO_TIFF'
    }

    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt}/{retries}: Preparing/Downloading {os.path.basename(out_path)}...")
            image_to_download = image.clip(region)
            url = image_to_download.getDownloadURL(params)

            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(1024 * 1024):
                    f.write(chunk)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                 print(f"[✓] Saved: {out_path}")
                 return True
            else:
                 if os.path.exists(out_path): os.remove(out_path)
                 raise ValueError("Downloaded file is empty.")

        except Exception as e:
            print(f"[!] Attempt {attempt} failed: {e}")
            last_exception = e
            if os.path.exists(out_path):
                 try: os.remove(out_path)
                 except OSError: pass
            if attempt < retries:
                print(f"    Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"[✗] Failed after {retries} attempts: {out_path}")

    return False

def generate_tiles(bbox, tile_size_deg=0.1):
    """
    Splits a bounding box into smaller tiles.
    """
    xmin, ymin, xmax, ymax = bbox
    x_steps = np.arange(xmin, xmax, tile_size_deg)
    y_steps = np.arange(ymin, ymax, tile_size_deg)

    tiles = []
    for x in x_steps:
        for y in y_steps:
            tile = [
                x,
                y,
                min(x + tile_size_deg, xmax),
                min(y + tile_size_deg, ymax)
            ]
            if tile[2] > tile[0] and tile[3] > tile[1]:
                tiles.append(tile)
    return tiles

def maskS2clouds(image: ee.Image) -> ee.Image:
    """Masks clouds in a Sentinel-2 SR image using the QA60 band."""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.select("B.*").updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])