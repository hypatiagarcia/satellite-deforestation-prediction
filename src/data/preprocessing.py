# src/data/preprocessing.py
import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random
import warnings 
import albumentations as A
from albumentations.pytorch import ToTensorV2
# In preprocessing.py
import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random
import warnings

# --- Normalization Statistics Calculation (Removed block_shape) ---

def calculate_normalization_stats(
    tile_paths: List[Path],
    band_mapping: Dict[str, int],
    bands_to_standardize: List[str] = ['S1_VV', 'S1_VH', 'DEM'],
    bands_to_scale: List[str] = ['S2_B', 'S2_G', 'S2_R', 'S2_N', 'S2_S1', 'S2_S2'],
    percentiles: Tuple[float, float] = (2.0, 98.0),
    max_tiles_for_scaling_stats: Optional[int] = 600
    ) -> Dict:
    """
    Calculates mean/std and percentile min/max using default windowed reading
    and sampling to manage memory. Standardization uses all tiles, scaling uses a sample.

    Args:
        tile_paths: List of paths to ALL GeoTIFF files (training set).
        band_mapping: Dictionary mapping band names to their 0-based index.
        bands_to_standardize: List of band names to calculate mean/std for.
        bands_to_scale: List of band names to calculate percentiles for.
        percentiles: Tuple of lower and upper percentiles for scaling.
        max_tiles_for_scaling_stats: Max # tiles sampled for scaling percentiles.
    Returns:
        Dictionary containing the calculated statistics.
    """
    stats_std = {band: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for band in bands_to_standardize}
    stats_scale_values = {band: [] for band in bands_to_scale}

    total_tiles = len(tile_paths)
    if total_tiles == 0: raise ValueError("tile_paths list is empty.")

    print(f"Calculating stats from {total_tiles} tiles (using default windowed reading)...")

    # Determine tiles for scaling percentile calculation
    scaling_tile_indices_set = set()
    if bands_to_scale:
        if max_tiles_for_scaling_stats is None or max_tiles_for_scaling_stats >= total_tiles:
            scaling_tile_indices_set = set(range(total_tiles))
            print(f"  Will accumulate scaling values from all {total_tiles} tiles.")
        else:
            num_to_sample = min(max_tiles_for_scaling_stats, total_tiles)
            scaling_tile_indices = random.sample(range(total_tiles), num_to_sample)
            scaling_tile_indices_set = set(scaling_tile_indices)
            print(f"  Will accumulate scaling values from a random sample of {num_to_sample} tiles.")

    # Process Tiles using default Windowed Reading
    for i, tile_path in enumerate(tqdm(tile_paths, desc="Processing Tiles")):
        try:
            with rasterio.open(tile_path) as src:
                # --- FIX: Iterate using default block windows ---
                for _, window in src.block_windows(): # Removed block_shape argument
                    # --- Standardization ---
                    for band_name in bands_to_standardize:
                        # ... (rest of standardization logic inside window loop is unchanged) ...
                        if band_name not in band_mapping: continue
                        band_idx = band_mapping[band_name]
                        if band_idx + 1 > src.count: continue
                        data = src.read(band_idx + 1, window=window).astype(np.float64)
                        nodata_val = src.nodata
                        valid_mask = data != nodata_val if nodata_val is not None else data > -999
                        valid_data = data[valid_mask]
                        if valid_data.size > 0:
                            stats_std[band_name]['sum'] += valid_data.sum()
                            stats_std[band_name]['sum_sq'] += np.sum(np.square(valid_data))
                            stats_std[band_name]['count'] += valid_data.size
                        del data, valid_data, valid_mask

                    # --- Scaling (only if tile is sampled) ---
                    if i in scaling_tile_indices_set:
                        for band_name in bands_to_scale:
                            # ... (rest of scaling logic inside window loop is unchanged) ...
                             if band_name not in band_mapping: continue
                             band_idx = band_mapping[band_name]
                             if band_idx + 1 > src.count: continue
                             data = src.read(band_idx + 1, window=window).astype(np.float32)
                             nodata_val = src.nodata
                             valid_mask = data != nodata_val if nodata_val is not None else data > 0
                             valid_data = data[valid_mask]
                             if valid_data.size > 0:
                                 stats_scale_values[band_name].append(valid_data.flatten()) # Flatten here before appending
                             del data, valid_data, valid_mask

        except Exception as e:
            # Changed print to warning for better traceability
            warnings.warn(f"Error processing {tile_path.name} for stats: {e}. Skipping file.", UserWarning)

    # --- Finalize statistics (keep unchanged) ---
        final_stats = {'standardization': {}, 'scaling': {}}
    print("\nFinalizing standardization statistics...")
    for band_name in bands_to_standardize:
        # ... (calculation logic for mean, std_dev) ...
        s = stats_std[band_name]
        count = s['count']
        if count > 0:
            mean = s['sum'] / count
            variance = (s['sum_sq'] / count) - (mean ** 2)
            std_dev = np.sqrt(max(0, variance))
            # --- FIX: Convert numpy types to Python float ---
            final_stats['standardization'][band_name] = {'mean': float(mean), 'std': float(std_dev)}
            print(f"  {band_name}: Mean={mean:.4f}, Std={std_dev:.4f} (from {count} pixels)")
        else:
            print(f"  {band_name}: No valid data found. Using default mean=0, std=1.")
            final_stats['standardization'][band_name] = {'mean': 0.0, 'std': 1.0}

    print(f"\nCalculating percentiles {percentiles} for scaling bands (from sampled tiles)...")
    for band_name in bands_to_scale:
        # ... (concatenation logic) ...
        if stats_scale_values[band_name]:
             # ... (concatenation and percentile calculation) ...
             try:
                 print(f"  Concatenating data for {band_name}...")
                 all_values = np.concatenate(stats_scale_values[band_name])
                 print(f"  Calculating percentiles for {band_name} ({all_values.size} pixels)...")
             except MemoryError:
                 warnings.warn(f"MemoryError during concatenation for {band_name}. Reduce 'max_tiles_for_scaling_stats'. Using default min/max.", UserWarning)
                 all_values = np.array([])

             if all_values.size > 0:
                try:
                    p_low = np.percentile(all_values, percentiles[0])
                    p_high = np.percentile(all_values, percentiles[1])
                except IndexError:
                    warnings.warn(f"Percentile calculation failed for {band_name}. Using min/max.", UserWarning)
                    p_low = np.min(all_values)
                    p_high = np.max(all_values)
                # --- FIX: Convert numpy types to Python float ---
                final_stats['scaling'][band_name] = {'min': float(p_low), 'max': float(p_high)}
                print(f"  {band_name}: Min ({percentiles[0]}%)={p_low:.4f}, Max ({percentiles[1]}%)={p_high:.4f}")
             else:
                 print(f"  {band_name}: No valid data for percentiles. Using default min=0, max=1.")
                 final_stats['scaling'][band_name] = {'min': 0.0, 'max': 1.0}
        else:
             print(f"  {band_name}: No pixel data accumulated. Using default min=0, max=1.")
             final_stats['scaling'][band_name] = {'min': 0.0, 'max': 1.0}

        stats_scale_values[band_name] = []
        all_values = None

    return final_stats

# --- Custom Dataset (No changes needed here) ---
class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for loading stacked GeoTIFF tiles, extracting random patches,
    applying on-the-fly normalization, and augmentations.
    Includes a strategy for sampling patches with positive examples.
    """
    def __init__(self,
                 tile_paths: List[Path],
                 band_mapping: Dict[str, int],
                 normalization_stats: Dict,
                 patch_size: int = 256,
                 patches_per_epoch: Optional[int] = None,
                 augment: bool = False,
                 positive_sampling_rate: float = 0.5, # Proportion of patches to attempt to get positives
                 max_positive_sample_attempts: int = 10): # Retries for finding a positive patch
        super().__init__()
        self.tile_paths = [Path(p) for p in tile_paths] # Ensure Path objects
        self.band_mapping = band_mapping
        self.index_to_name = {v: k for k, v in band_mapping.items()}
        self.normalization_stats = normalization_stats
        self.patch_size = patch_size
        self.augment = augment
        self.positive_sampling_rate = positive_sampling_rate
        self.max_positive_sample_attempts = max_positive_sample_attempts

        if patches_per_epoch is None:
             # Default to 10 patches per tile if not specified
             self.num_patches_per_epoch = len(self.tile_paths) * 10
        else:
             self.num_patches_per_epoch = patches_per_epoch

        print(f"Dataset initialized. Target patches per epoch: {self.num_patches_per_epoch}.")

        self.feature_indices = [v for k, v in band_mapping.items() if k != 'Label']
        self.label_index = band_mapping['Label']
        self.num_feature_bands = len(self.feature_indices)

        self._tile_dims = {}
        self._valid_tile_indices = [] # Indices relative to original self.tile_paths
        self._positive_tile_indices = [] # Indices of tiles confirmed to have >0 positive labels

        print("Pre-checking tiles for dimensions and positive labels...")
        for i, tile_path in enumerate(tqdm(self.tile_paths, desc="Scanning Tiles")):
            try:
                if not tile_path.exists():
                    warnings.warn(f"Path does not exist: {tile_path}. Skipping.", UserWarning)
                    continue
                with rasterio.open(tile_path) as src:
                    if src.count != len(self.band_mapping):
                        warnings.warn(f"Tile {tile_path.name} has {src.count} bands, expected {len(self.band_mapping)}. Skipping.", UserWarning)
                        continue
                    if src.width >= self.patch_size and src.height >= self.patch_size:
                        self._tile_dims[i] = (src.height, src.width)
                        self._valid_tile_indices.append(i)

                        # Check for positive labels in this tile
                        # Read only the label band, potentially a small window or downsampled for speed
                        try:
                            label_band_data = src.read(self.label_index + 1) # Rasterio is 1-indexed
                            if np.any(label_band_data > 0):
                                self._positive_tile_indices.append(i)
                        except Exception as e_label:
                             warnings.warn(f"Could not read label band for {tile_path.name} during positive check: {e_label}. Assuming no positives.", UserWarning)

            except rasterio.errors.RasterioIOError as e_rio:
                warnings.warn(f"RasterioIOError reading {tile_path.name}: {e_rio}. Skipping.", UserWarning)
            except Exception as e:
                warnings.warn(f"Generic error processing {tile_path.name} during pre-check: {e}. Skipping.", UserWarning)

        if not self._valid_tile_indices:
            raise ValueError(f"CRITICAL: No valid tiles found suitable for patch size {self.patch_size} among {len(self.tile_paths)} paths.")
        print(f"Found {len(self._valid_tile_indices)} tiles suitable for patch size {self.patch_size}.")
        print(f"Found {len(self._positive_tile_indices)} tiles containing positive labels.")

        if self.positive_sampling_rate > 0 and not self._positive_tile_indices:
            warnings.warn("Positive sampling requested, but no tiles with positive labels found during scan. Will sample randomly.", UserWarning)
            self.positive_sampling_rate = 0 # Disable if no positive tiles

        # Define augmentations using Albumentations
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Add more augmentations here as needed:
                # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                # A.GaussNoise(var_limit=(1.0, 10.0), p=0.2), # Careful with units
                # A.ElasticTransform(p=0.1, alpha=10, sigma=50 * 0.05, alpha_affine=50 * 0.03),
            ])
        else:
            self.transform = None # No augmentations

    def __len__(self) -> int:
        return self.num_patches_per_epoch

    def _get_random_patch_from_tile(self, tile_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper to extract a random patch from a specific tile index."""
        tile_path = self.tile_paths[tile_idx]
        tile_height, tile_width = self._tile_dims[tile_idx]

        row_off = random.randint(0, tile_height - self.patch_size)
        col_off = random.randint(0, tile_width - self.patch_size)
        window = Window(col_off, row_off, self.patch_size, self.patch_size)

        try:
            with rasterio.open(tile_path) as src:
                # Read all bands for the window
                patch_stack = src.read(window=window).astype(np.float32)

            # Ensure correct band order and separate features/label
            # Create an empty array for the ordered stack to handle potential band_mapping gaps
            ordered_patch_stack = np.zeros((len(self.band_mapping), self.patch_size, self.patch_size), dtype=np.float32)
            for i in range(src.count): # Assuming src.count is reliable here after pre-check
                 if i in self.index_to_name : # Check if this band index is in our mapping
                    target_idx_in_map = i # This is the original index from band_mapping
                    ordered_patch_stack[target_idx_in_map] = patch_stack[i]


            features = ordered_patch_stack[self.feature_indices, :, :]
            label = ordered_patch_stack[self.label_index, :, :] # Should be (H, W)
            return features, label
        except Exception as e:
            warnings.warn(f"Error loading patch from {tile_path.name} at window {window}: {e}", UserWarning)
            return None, None


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features_patch, label_patch = None, None
        attempt_positive_sample = random.random() < self.positive_sampling_rate and self._positive_tile_indices

        if attempt_positive_sample:
            # Try to get a patch with positive pixels
            for _ in range(self.max_positive_sample_attempts):
                # Select a random tile known to have positive pixels
                tile_idx = random.choice(self._positive_tile_indices)
                current_features, current_label = self._get_random_patch_from_tile(tile_idx)
                if current_label is not None and np.any(current_label > 0):
                    features_patch, label_patch = current_features, current_label
                    break
            if features_patch is None: # Failed to find a positive patch after attempts
                 pass # Fall through to random sampling
        
        if features_patch is None: # Fallback or normal random sampling
            if not self._valid_tile_indices: # Should not happen if __init__ check passed
                 return self._get_dummy_item()
            tile_idx = random.choice(self._valid_tile_indices)
            features_patch, label_patch = self._get_random_patch_from_tile(tile_idx)

        # If still None after all attempts (e.g., file read error in _get_random_patch_from_tile)
        if features_patch is None or label_patch is None:
            warnings.warn(f"Failed to load any valid patch for item {idx}. Returning zeros.", UserWarning)
            return self._get_dummy_item()

        # Apply normalization
        for i, band_abs_idx in enumerate(self.feature_indices):
            band_name = self.index_to_name.get(band_abs_idx) # Use .get for safety
            if band_name is None: continue # Should not happen if band_mapping is correct

            if band_name in self.normalization_stats.get('standardization', {}):
                stats = self.normalization_stats['standardization'][band_name]
                mean, std = stats['mean'], stats['std']
                features_patch[i] = (features_patch[i] - mean) / (std + 1e-7)
            elif band_name in self.normalization_stats.get('scaling', {}):
                stats = self.normalization_stats['scaling'][band_name]
                p_min, p_max = stats['min'], stats['max']
                delta = p_max - p_min
                features_patch[i] = np.clip((features_patch[i] - p_min) / (delta + 1e-7), 0.0, 1.0)

        # Apply augmentations if enabled
        if self.transform:
            augmented = self.transform(image=features_patch.transpose(1, 2, 0), mask=label_patch) # Albumentations expects HWC
            features_patch_aug = augmented['image'].transpose(2, 0, 1) # Back to CHW
            label_patch_aug = augmented['mask']
        else:
            features_patch_aug = features_patch
            label_patch_aug = label_patch
        
        # Convert to PyTorch Tensors
        features_tensor = torch.from_numpy(features_patch_aug.copy()).float() # .copy() for safety
        label_tensor = torch.from_numpy(label_patch_aug.copy()).long().unsqueeze(0) # Add channel dim for some losses: (1, H, W)

        return features_tensor, label_tensor

    def _get_dummy_item(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a dummy zero tensor pair for error cases."""
        return torch.zeros((self.num_feature_bands, self.patch_size, self.patch_size), dtype=torch.float32), \
               torch.zeros((1, self.patch_size, self.patch_size), dtype=torch.long) # Match label_tensor shape