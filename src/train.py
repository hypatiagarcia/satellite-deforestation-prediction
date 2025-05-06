# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import yaml
import json
from pathlib import Path
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
import sys
import warnings # Para manejar advertencias

# --- Import Project Modules ---
from models.unet_swin import get_unet_swin_model # Asumiendo Swin-UNet
from models.loss_functions import CombinedLoss    # Tu función de pérdida combinada
from data.preprocessing import DeforestationDataset   # LA CLASE DATASET MODIFICADA
from utils import calculate_iou, calculate_f1   # Tus funciones de métricas

# --- Funciones Auxiliares ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_split_paths(list_file_path: Path) -> list[str]:
    if not list_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {list_file_path}")
    with open(list_file_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def calculate_pos_weight(dataloader: DataLoader, device: torch.device, iter_limit: int = 500) -> torch.Tensor:
    print("Calculating positive weight for BCE loss...")
    count_0 = 0
    count_1 = 0
    # Iterar sobre una fracción del dataloader para estimar el peso
    # (Iterar sobre todo puede ser muy lento si patches_per_epoch es grande)
    num_batches_to_check = min(len(dataloader), iter_limit) # Limitar el número de batches a revisar

    for i, (_, labels) in enumerate(tqdm(dataloader, desc="Calculating class counts", total=num_batches_to_check)):
        if i >= num_batches_to_check:
            break
        # Labels vienen como (B, 1, H, W) o (B, H, W), asegurar que sean binarias
        # y cuenten correctamente los píxeles.
        labels_binary = (labels.to(device) > 0.5).long() # Asumiendo que las etiquetas ya son 0 o 1
        count_1 += (labels_binary == 1).sum().item()
        count_0 += (labels_binary == 0).sum().item()

    if count_1 == 0:
        warnings.warn("Warning: No positive samples found in sampled batches for pos_weight calculation. Using weight 1.0.", UserWarning)
        pos_weight_value = 1.0
    else:
        pos_weight_value = count_0 / count_1
    print(f"Class Counts (from {num_batches_to_check} batches) - 0: {count_0}, 1: {count_1}")
    print(f"Calculated BCE pos_weight: {pos_weight_value:.4f}")
    return torch.tensor([pos_weight_value], device=device)

def find_optimal_threshold_and_evaluate(model, dataloader, device, criterion, base_threshold=0.5):
    """
    (ESQUELETO) Encuentra el umbral óptimo en el set de validación y re-evalúa.
    Deberías llamar a esto después del entrenamiento o al final de cada epoch si quieres optimizarlo dinámicamente.
    """
    print("\nFinding optimal threshold on validation set...")
    model.eval()
    all_sigmoid_outputs_list = []
    all_labels_list = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Getting Val Predictions for Thresholding"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1) # (B, 1, H, W)
            outputs = model(features)
            sigmoid_outputs = torch.sigmoid(outputs)
            all_sigmoid_outputs_list.append(sigmoid_outputs.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy().astype(np.uint8))

    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs_list)
    all_labels = np.concatenate(all_labels_list)

    best_f1 = -1.0
    optimal_threshold = base_threshold
    best_iou = -1.0

    # Escanear umbrales
    thresholds_to_scan = np.arange(0.1, 0.95, 0.05)
    for thr in tqdm(thresholds_to_scan, desc="Scanning Thresholds"):
        preds_binary = (all_sigmoid_outputs > thr).astype(np.uint8)
        f1 = calculate_f1(preds_binary.reshape(-1), all_labels.reshape(-1), positive_label=1)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = thr
            best_iou = calculate_iou(preds_binary.reshape(-1), all_labels.reshape(-1), positive_label=1)

    print(f"Optimal threshold found: {optimal_threshold:.2f} (F1: {best_f1:.4f}, IoU: {best_iou:.4f})")
    # Re-evaluar con el umbral óptimo si es diferente al base
    # Esto es solo para mostrar, la evaluación principal del epoch ya se hizo
    return optimal_threshold, best_f1, best_iou
# --- Fin Funciones Auxiliares ---

# --- Función Principal de Entrenamiento ---
def train_model(config_path: str):
    # -- Cargar Configuración --
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # -- Setup --
    set_seed(config['training']['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    run_output_dir = project_root / Path(config['output']['base_dir']) / config['output']['run_name']
    model_save_dir = run_output_dir / "models"
    log_dir = run_output_dir / "logs" # Para TensorBoard u otros logs
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(run_output_dir / 'config_used.yml', 'w') as f:
        yaml.dump(config, f)

    # -- Cargar Datos --
    print("Loading data paths...")
    data_config = config['data']
    train_list_file = project_root / data_config['train_tiles_list_file']
    val_list_file = project_root / data_config['val_tiles_list_file']

    train_paths_rel = load_split_paths(train_list_file)
    val_paths_rel = load_split_paths(val_list_file)

    # Convertir rutas relativas (desde la raíz del proyecto) a absolutas
    train_paths = [project_root / p for p in train_paths_rel]
    val_paths = [project_root / p for p in val_paths_rel]

    print(f"Total training paths: {len(train_paths)}, Validation paths: {len(val_paths)}")
    if not train_paths: raise ValueError("No training paths loaded.")
    if not val_paths: warnings.warn("No validation paths loaded.", UserWarning)
    print(f"Example training path: {train_paths[0]}")
    if not train_paths[0].exists(): raise FileNotFoundError(f"Example training path {train_paths[0]} does not exist.")


    norm_stats_file = project_root / data_config['normalization_stats_file']
    with open(norm_stats_file, 'r') as f:
        normalization_stats = json.load(f)
    band_mapping = data_config['band_mapping']

    # -- Crear Datasets --
    print("Creating datasets...")
    train_cfg = config['training']
    # Obtener parámetros para el Dataset (con valores por defecto si no están en config)
    positive_sampling_rate = train_cfg.get('positive_sampling_rate', 0.5) # Default 0.5
    max_positive_sample_attempts = train_cfg.get('max_positive_sample_attempts', 10) # Default 10

    train_dataset = DeforestationDataset(
        tile_paths=train_paths,
        band_mapping=band_mapping,
        normalization_stats=normalization_stats,
        patch_size=train_cfg['patch_size'],
        patches_per_epoch=train_cfg.get('patches_per_epoch'), # Puede ser None
        augment=True,
        positive_sampling_rate=positive_sampling_rate,
        max_positive_sample_attempts=max_positive_sample_attempts
    )
    val_dataset = DeforestationDataset(
        tile_paths=val_paths,
        band_mapping=band_mapping,
        normalization_stats=normalization_stats,
        patch_size=train_cfg['patch_size'],
        patches_per_epoch=None, # Validar sobre todos los patches posibles una vez
        augment=False,
        positive_sampling_rate=0.0 # No forzar positivos en validación para una evaluación más real
    )

    # -- Crear DataLoaders --
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, # Shuffle es importante para train
        num_workers=train_cfg['num_workers'], pin_memory=train_cfg['pin_memory'], drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.get('batch_size_val', train_cfg['batch_size']), shuffle=False,
        num_workers=train_cfg['num_workers'], pin_memory=train_cfg['pin_memory'], drop_last=False
    )
    print(f"Train batches per epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # -- Calcular Peso Positivo para BCE --
    if isinstance(config['loss']['bce_pos_weight'], str) and config['loss']['bce_pos_weight'].lower() == "calculate":
        pos_weight_tensor = calculate_pos_weight(train_loader, device, iter_limit=train_cfg.get('pos_weight_calc_batches', 500))
    elif isinstance(config['loss']['bce_pos_weight'], (int, float)):
        pos_weight_tensor = torch.tensor([config['loss']['bce_pos_weight']], device=device)
        print(f"Using fixed BCE pos_weight: {pos_weight_tensor.item():.4f}")
    else: # Default o si no se especifica correctamente
        pos_weight_tensor = torch.tensor([1.0], device=device)
        warnings.warn("bce_pos_weight not specified or 'calculate', using default 1.0.", UserWarning)


    # -- Inicializar Modelo, Pérdida, Optimizador --
    print("Initializing model, loss, and optimizer...")
    model_cfg = config['model']
    num_input_channels = len([k for k in band_mapping if k != 'Label'])

    print("-" * 50)
    print(f"Initializing U-Net with backbone: {model_cfg['backbone']}")
    print(f"Input Channels: {num_input_channels}, Output Classes: 1")
    print(f"Encoder Weights: {model_cfg['encoder_weights']}")
    print("-" * 50)

    model = get_unet_swin_model( # O tu función para el modelo que estés usando
        num_input_channels=num_input_channels,
        num_output_classes=1, # Salida binaria (deforestación sí/no)
        backbone=model_cfg['backbone'],
        encoder_weights=model_cfg['encoder_weights'] # Puede ser None o 'imagenet'
    ).to(device)

    criterion = CombinedLoss(
        pos_weight=pos_weight_tensor,
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight']
    ).to(device)
    print(f"Using CombinedLoss with BCE Weight: {config['loss']['bce_weight']}, Dice Weight: {config['loss']['dice_weight']}")
    print(f"Using BCE pos_weight: {pos_weight_tensor.item():.4f}")


    optimizer_cfg = config['optimizer']
    if optimizer_cfg['name'].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])
    elif optimizer_cfg['name'].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg.get('weight_decay', 0))
    else:
        raise ValueError(f"Optimizer {optimizer_cfg['name']} not supported.")
    print(f"Using Optimizer: {optimizer_cfg['name']} with LR: {optimizer_cfg['lr']}, Weight Decay: {optimizer_cfg.get('weight_decay', 'N/A')}")

    scheduler = None
    scheduler_cfg = config['scheduler']
    if scheduler_cfg['use']:
        if scheduler_cfg['type'].lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_cfg['params'])
            print(f"Using ReduceLROnPlateau scheduler with params: {scheduler_cfg['params']}")
        elif scheduler_cfg['type'].lower() == "cosineannealinglr":
            # T_max es crucial para CosineAnnealingLR
            if 'T_max' not in scheduler_cfg['params']:
                scheduler_cfg['params']['T_max'] = config['training']['epochs'] # Usar total epochs si no se especifica
            scheduler = CosineAnnealingLR(optimizer, **scheduler_cfg['params'])
            print(f"Using CosineAnnealingLR scheduler with params: {scheduler_cfg['params']}")
        else:
            warnings.warn(f"Scheduler type '{scheduler_cfg['type']}' not implemented or recognized. No scheduler used.", UserWarning)

    # -- Bucle de Entrenamiento --
    print("Starting training loop...")
    best_val_metric_score = -1.0 if scheduler_cfg.get('params',{}).get('mode','max') == 'max' else float('inf')
    start_time_all = time.time()

    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loss_epoch = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")

        for features, labels in train_iterator:
            features = features.to(device, non_blocking=True) # non_blocking si pin_memory=True
            labels = labels.to(device, non_blocking=True) # Ya es (B, 1, H, W) y long desde Dataset

            optimizer.zero_grad()
            outputs = model(features) # Salida (B, 1, H, W)
            loss = criterion(outputs, labels.float()) # Criterion espera labels float para BCEWithLogits
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_epoch / len(train_loader)

        # -- Bucle de Validación --
        model.eval()
        val_loss_epoch = 0.0
        all_preds_val_list = []
        all_labels_val_list = []
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
        eval_threshold = config['evaluation']['threshold'] # Usar umbral de config

        with torch.no_grad():
            for features, labels in val_iterator:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) # (B, 1, H, W), long
                outputs = model(features) # (B, 1, H, W)
                loss = criterion(outputs, labels.float())
                val_loss_epoch += loss.item()

                # Convertir a binario para métricas
                preds_sigmoid = torch.sigmoid(outputs)
                preds_binary = (preds_sigmoid > eval_threshold).cpu().numpy().astype(np.uint8)
                all_preds_val_list.append(preds_binary)
                all_labels_val_list.append(labels.cpu().numpy().astype(np.uint8)) # labels ya son 0/1

        avg_val_loss = val_loss_epoch / len(val_loader)

        all_preds_val = np.concatenate(all_preds_val_list) # (N, 1, H, W)
        all_labels_val = np.concatenate(all_labels_val_list) # (N, 1, H, W)

        # Aplanar para calcular métricas globales (o calcular por imagen y promediar)
        preds_flat = all_preds_val.reshape(-1)
        labels_flat = all_labels_val.reshape(-1)

        val_iou = calculate_iou(preds_flat, labels_flat, positive_label=1)
        val_f1 = calculate_f1(preds_flat, labels_flat, positive_label=1)

        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} - "
              f"Duration: {epoch_duration:.2f}s - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"Val IoU (thr={eval_threshold}): {val_iou:.4f} - "
              f"Val F1 (thr={eval_threshold}): {val_f1:.4f}")

        # Loguear LR actual
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current LR: {current_lr:.2e}")

        # -- Programación de LR y Guardado del Modelo --
        metric_to_monitor = config['evaluation']['metric_to_monitor'].lower()
        scheduler_mode = scheduler_cfg.get('params',{}).get('mode','max')

        if metric_to_monitor == 'f1_score': current_metric_val = val_f1
        elif metric_to_monitor == 'iou': current_metric_val = val_iou
        elif metric_to_monitor == 'loss': current_metric_val = avg_val_loss # Para 'min' mode
        else: raise ValueError(f"Unsupported metric_to_monitor: {metric_to_monitor}")

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric_val)
            else: # Para CosineAnnealingLR y otros que se actualizan por epoch
                scheduler.step()

        improved = False
        if scheduler_mode == 'max':
            if current_metric_val > best_val_metric_score:
                improved = True
        else: # mode == 'min'
            if current_metric_val < best_val_metric_score:
                improved = True

        if improved:
            print(f"  Validation {metric_to_monitor} improved "
                  f"({best_val_metric_score:.4f} -> {current_metric_val:.4f}). Saving model...")
            best_val_metric_score = current_metric_val
            # Borrar modelos anteriores 'best' para ahorrar espacio
            for prev_best in model_save_dir.glob("best_model_*.pth"):
                try: prev_best.unlink()
                except OSError as e: print(f"Error deleting old best model {prev_best}: {e}")

            save_path = model_save_dir / f"best_model_epoch_{epoch+1}_{metric_to_monitor}_{best_val_metric_score:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** Best model saved to {save_path} ***")

    # -- Fin del Entrenamiento --
    total_training_time = time.time() - start_time_all
    print(f"\nTraining finished in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation {metric_to_monitor} achieved: {best_val_metric_score:.4f}")

    # (Opcional) Ejecutar búsqueda de umbral óptimo al final
    optimal_thr, best_f1_opt, best_iou_opt = find_optimal_threshold_and_evaluate(model, val_loader, device, criterion)
    print(f"After threshold optimization: Optimal Thr={optimal_thr:.2f}, F1={best_f1_opt:.4f}, IoU={best_iou_opt:.4f}")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deforestation Detection Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file")
    args = parser.parse_args()

    # Verificar si el archivo de config existe
    config_file = Path(args.config)
    if not config_file.is_file():
        # Intentar resolverlo relativo al script si no es absoluto
        script_dir = Path(__file__).resolve().parent
        config_file_rel = script_dir / args.config
        if config_file_rel.is_file():
            args.config = str(config_file_rel)
        else:
            print(f"ERROR: Configuration file not found at {args.config} or {config_file_rel}")
            sys.exit(1)

    train_model(args.config)