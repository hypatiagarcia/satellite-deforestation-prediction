import os

# ── Region of Interest ─────────────────────────────────────────────────────────
REGION_COORDS = [-63.5, -10.0, -62.5, -9.0] # Rondônia, Brazil

# ── Date Ranges (Example: Features from 2019, predict 2020 loss) ──────────────
FEATURE_START_DATE = '2019-01-01'
FEATURE_END_DATE   = '2019-12-31'
LABEL_YEAR = 2020 

# ── Output Directories ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
)

STACKED_DATA_DIR = os.path.join(BASE_DIR, 'stacked_tiles')

# ── Earth Engine Defaults ──────────────────────────────────────────────────────
EE_SCALE = 10        
EE_CRS   = 'EPSG:4326' 

# --- Hansen GFC Version (Check GEE Catalog for latest) ---
HANSEN_VERSION = 'UMD/hansen/global_forest_change_2023_v1_11'

# --- Project ID (Replace with your actual GEE Project ID) ---
PROJECT_ID = "***"

# --- Bands to Select ---
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'] # Blue, Green, Red, NIR, SWIR1, SWIR2
S1_BANDS = ['VV', 'VH']
DEM_BANDS = ['elevation']
LABEL_BANDS = ['lossyear'] 