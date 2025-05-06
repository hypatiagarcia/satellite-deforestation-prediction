# src/models/unet_swin.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import warnings

def get_unet_swin_model(
    num_input_channels: int = 9,
    num_output_classes: int = 1,
    backbone: str = 'tu-swin_small_patch4_window7_224', # Example Swin backbone
    encoder_weights: str = 'imagenet' # Use pretrained weights
    ) -> nn.Module:
    """
    Initializes a U-Net model with a Swin Transformer backbone using segmentation-models-pytorch.

    Args:
        num_input_channels: Number of input bands/channels.
        num_output_classes: Number of output classes (1 for binary segmentation logits).
        backbone: Name of the Swin Transformer encoder backbone available in timm
                  (prefixed with 'tu-' for smp usually, e.g., 'tu-swin_tiny_patch4_window7_224',
                  'tu-swin_small_patch4_window7_224', 'tu-swin_base_patch4_window7_224').
        encoder_weights: Pre-training weights ('imagenet' or None). Using 'imagenet' is highly recommended.

    Returns:
        A PyTorch U-Net model with the specified Swin backbone.
    """
    print("-" * 50)
    print(f"Initializing U-Net with backbone: {backbone}")
    print(f"Input Channels: {num_input_channels}, Output Classes: {num_output_classes}")
    print(f"Encoder Weights: {encoder_weights}")

    if encoder_weights == "imagenet" and num_input_channels != 3:
        warnings.warn(
            f"Using 'imagenet' weights with {num_input_channels} input channels. "
            f"Relying on segmentation-models-pytorch (version >= 0.3.0 recommended) "
            f"and timm to handle the adaptation of the first layer automatically. "
            f"Verify this works as expected or consider training from scratch (weights=None).",
            UserWarning
        )
    elif encoder_weights is None:
         print("Initializing encoder weights from scratch.")
    elif num_input_channels == 3:
         print("Using standard 3-channel ImageNet weights.")

    try:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=num_input_channels,
            classes=num_output_classes,
        )
        print("Model initialized successfully.")
        print("-" * 50)

    except Exception as e:
        print(f"\n--- ERROR initializing model ---")
        print(f"Failed to create U-Net with backbone '{backbone}'.")
        print(f"Error message: {e}")
        print(f"Please ensure:")
        print(f"  1. The backbone name '{backbone}' is correct and supported by segmentation-models-pytorch/timm.")
        print(f"     (Common Swin names need 'tu-' prefix, e.g., 'tu-swin_small_patch4_window7_224')")
        print(f"  2. You have 'timm' installed (`pip install timm`).")
        print(f"  3. Your segmentation-models-pytorch version is compatible (>= 0.3.0 recommended).")
        print("-" * 50)
        raise e # Re-raise the exception to halt execution

    return model

# if __name__ == '__main__':
    # # Example instantiation
    # print("Attempting to instantiate Swin-Unet (example)...")
    # # Note: This requires timm to be installed
    # try:
    #     # Example using a smaller Swin variant
    #     model = get_unet_swin_model(
    #         num_input_channels=9,
    #         num_output_classes=1,
    #         backbone='tu-swin_tiny_patch4_window7_224', # Use tiny for basic test
    #         encoder_weights='imagenet'
    #     )
    #     # print("\nModel Architecture (partial):") # Printing full Swin can be very long
    #     # print(model.encoder.layer1) # Example part
    #     # print(model.decoder.blocks[0])

    #     # Test with dummy input
    #     print("\nTesting inference with dummy input...")
    #     dummy_input = torch.randn(2, 9, 256, 256) # Batch size 2, 9 channels, 256x256
    #     output = model(dummy_input)
    #     print(f"  Dummy Input Shape: {dummy_input.shape}")
    #     print(f"  Dummy Output Shape: {output.shape}") # Should be [2, 1, 256, 256]
    #     print("Dummy inference successful.")

    # except Exception as e:
    #     print(f"\n--- Example Instantiation Failed ---")
    #     print(f"  Could not run example. Error: {e}")
    #     print(f"  Check installation ('timm', 'segmentation-models-pytorch') and backbone name.")