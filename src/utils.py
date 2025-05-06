# src/utils.py
import numpy as np

# --- Add other utility functions as needed ---

def calculate_iou(preds_flat: np.ndarray, labels_flat: np.ndarray, positive_label: int = 1, smooth: float = 1e-6) -> float:
    """
    Calculates the Intersection over Union (IoU) or Jaccard Index for the positive class
    in a binary segmentation task.

    Args:
        preds_flat: Flattened numpy array of binary predictions (0 or 1).
        labels_flat: Flattened numpy array of ground truth labels (0 or 1).
        positive_label: The value representing the positive class (default: 1).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        The IoU score for the positive class. Returns 0.0 if no true positive
        examples exist in the labels or predictions.
    """
    # Ensure input arrays are boolean or integer {0, 1}
    preds_flat = (preds_flat == positive_label)
    labels_flat = (labels_flat == positive_label)

    # Calculate intersection (True Positives)
    intersection = np.sum(preds_flat & labels_flat)

    # Calculate union (True Positives + False Positives + False Negatives)
    total_preds = np.sum(preds_flat)
    total_labels = np.sum(labels_flat)
    union = total_preds + total_labels - intersection

    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)

    # Handle case where both union and intersection are 0 (no positives in ground truth and prediction)
    # In this specific case, arguably IoU is 1 if perfect match, but commonly treated as 0 or requires context.
    # For evaluating positive class detection, if union is 0, IoU is effectively 0 for that class.
    # Check if union is effectively zero considering smoothing
    if union < smooth: # Effectively means no positive labels or predictions
         # If intersection is also zero (correct prediction for 'no positive'), IoU could be argued as 1.
         # However, if focused on detecting the positive class, IoU is typically 0 here.
         # Let's return 0 if the goal is positive class detection performance.
         # If you need perfect background IoU, this might need adjustment.
         return 0.0

    return iou


def calculate_f1(preds_flat: np.ndarray, labels_flat: np.ndarray, positive_label: int = 1, smooth: float = 1e-6) -> float:
    """
    Calculates the F1 Score (Dice coefficient is mathematically equivalent for sets)
    for the positive class in a binary segmentation task.

    Args:
        preds_flat: Flattened numpy array of binary predictions (0 or 1).
        labels_flat: Flattened numpy array of ground truth labels (0 or 1).
        positive_label: The value representing the positive class (default: 1).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        The F1 score for the positive class. Returns 0.0 if no true positive
        examples exist in the labels or predictions.
    """
    preds_flat = (preds_flat == positive_label)
    labels_flat = (labels_flat == positive_label)

    # Calculate TP, FP, FN
    tp = np.sum(preds_flat & labels_flat)  # True Positives
    fp = np.sum(preds_flat & ~labels_flat) # False Positives
    fn = np.sum(~preds_flat & labels_flat) # False Negatives

    # Calculate Precision and Recall
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    # Handle edge case: no positive labels AND no positive predictions
    if (tp + fp + tp + fn) < smooth * 2: # Effectively if TP+FP+FN is 0
        # If TP is also 0, it means perfect prediction of 'no positive', F1 is arguably 1.
        # But like IoU, if focused on positive class detection, F1 is 0 here.
        return 0.0 # Return 0 if no positives were predicted or present

    return f1

if __name__ == '__main__':
    # --- Example Usage ---
    print("Example Metric Calculation:")
    # Perfect match
    labels = np.array([0, 1, 0, 1, 1, 0])
    preds  = np.array([0, 1, 0, 1, 1, 0])
    print(f" Labels: {labels}")
    print(f" Preds:  {preds}")
    print(f" IoU (Class 1): {calculate_iou(preds, labels):.4f}") # Should be 1.0
    print(f" F1 (Class 1):  {calculate_f1(preds, labels):.4f}")  # Should be 1.0
    print("-" * 20)

    # One mismatch (FN)
    labels = np.array([0, 1, 0, 1, 1, 0])
    preds  = np.array([0, 1, 0, 1, 0, 0]) # Missed one '1'
    # TP=2, FP=0, FN=1
    # IoU = 2 / (2 + 0 + 1) = 2/3 = 0.6667
    # Precision = 2 / (2+0) = 1.0
    # Recall = 2 / (2+1) = 2/3
    # F1 = 2 * (1 * 2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
    print(f" Labels: {labels}")
    print(f" Preds:  {preds}")
    print(f" IoU (Class 1): {calculate_iou(preds, labels):.4f}") # Should be ~0.6667
    print(f" F1 (Class 1):  {calculate_f1(preds, labels):.4f}")  # Should be 0.8
    print("-" * 20)

     # One mismatch (FP)
    labels = np.array([0, 1, 0, 1, 0, 0])
    preds  = np.array([0, 1, 0, 1, 1, 0]) # Predicted extra '1'
    # TP=2, FP=1, FN=0
    # IoU = 2 / (2 + 1 + 0) = 2/3 = 0.6667
    # Precision = 2 / (2+1) = 2/3
    # Recall = 2 / (2+0) = 1.0
    # F1 = 2 * (2/3 * 1) / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
    print(f" Labels: {labels}")
    print(f" Preds:  {preds}")
    print(f" IoU (Class 1): {calculate_iou(preds, labels):.4f}") # Should be ~0.6667
    print(f" F1 (Class 1):  {calculate_f1(preds, labels):.4f}")  # Should be 0.8
    print("-" * 20)

    # No positive labels or predictions
    labels = np.array([0, 0, 0, 0])
    preds  = np.array([0, 0, 0, 0])
    print(f" Labels: {labels}")
    print(f" Preds:  {preds}")
    print(f" IoU (Class 1): {calculate_iou(preds, labels):.4f}") # Should be 0.0 (based on implementation logic)
    print(f" F1 (Class 1):  {calculate_f1(preds, labels):.4f}")  # Should be 0.0 (based on implementation logic)
    print("-" * 20)

    # No positive labels, one false positive prediction
    labels = np.array([0, 0, 0, 0])
    preds  = np.array([0, 1, 0, 0])
    # TP=0, FP=1, FN=0
    print(f" Labels: {labels}")
    print(f" Preds:  {preds}")
    print(f" IoU (Class 1): {calculate_iou(preds, labels):.4f}") # Should be 0.0
    print(f" F1 (Class 1):  {calculate_f1(preds, labels):.4f}")  # Should be 0.0
    print("-" * 20)