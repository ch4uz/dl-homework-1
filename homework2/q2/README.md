# RNAcompete 

This project provides a pre-built data pipeline (`utils.py`) to handle data ingestion, cleaning, and normalization for the RNAcompete dataset. This allows you to focus strictly on designing your deep learning models.

## 1\. Setup

Ensure the following files are in your project directory:

1.  `utils.py` (The provided data loader)
2.  `norm_data.txt` (The raw data matrix)
3.  `metadata.xlsx`

**Dependencies:**
You will need the standard deep learning stack, plus `openpyxl` for reading the metadata Excel file:

```bash
pip install openpyxl
```
or
```bash
conda install conda-forge::openpyxl
```

Ignore the warning about `openpyxl`: `python3.8/site-packages/openpyxl/worksheet/_read_only.py:85: UserWarning: Unknown extension is not supported and will be removed`


## 2\. Configuration (`config.py`)

The `config.py` file contains the `RNAConfig` dataclass, which manages global settings for the data pipeline. You can modify this file directly or pass a custom config object to the loader.

## 3\. Usage

You do not need to parse files manually. Use the `load_rnacompete_data` function from `utils`. The first time you run the function, it will preprocess the data and save it for future use. Then, when you call the function again, it will load the preprocessed data.

### Example Code

```python
import torch
from torch.utils.data import DataLoader
from utils import load_rnacompete_data

# 1. Load Data for a specific protein (e.g., 'RBFOX1', 'PTB', 'A1CF')
# This returns a PyTorch TensorDataset ready for training
train_dataset = load_rnacompete_data(protein_name='A1CF', split='train')
val_dataset   = load_rnacompete_data(protein_name='A1CF', split='val')
test_dataset  = load_rnacompete_data(protein_name='A1CF', split='test')

# 2. Wrap in a standard PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Training Loop Example
for batch in train_loader:
    # Unpack the batch: (Sequences, Intensities, ValidityMasks)
    x, y, mask = batch
    
    # x shape:    (Batch, 41, 4)  <- One-Hot Encoded Sequence
    # y shape:    (Batch, 1)      <- Normalized Binding Intensity
    # mask shape: (Batch, 1)      <- 1.0 if valid, 0.0 if NaN
    
    # Forward pass
    predictions = model(x)
    
    # ... Calculate Loss (See Section 3) ...
```

## 4\. Data Format & Shapes

### Input: `x` (Sequence)

  * **Shape:** `(Batch_Size, 41, 4)`
  * **Format:** One-Hot Encoding.
      * A = `[1, 0, 0, 0]`
      * C = `[0, 1, 0, 0]`
      * G = `[0, 0, 1, 0]`
      * U = `[0, 0, 0, 1]`
      * N = `[0.25, 0.25, 0.25, 0.25]` (Unknown/Padding)

### Target: `y` (Intensity)

  * **Shape:** `(Batch_Size, 1)`
  * **Preprocessing:** The raw intensities have been:
    1.  Clipped (to remove extreme outliers).
    2.  Log-transformed (to handle high dynamic range).
    3.  Z-Scored (Standardized to Mean=0, Std=1).

### Mask: `mask` (Validity)

  * **Shape:** `(Batch_Size, 1)`
  * **Why?** The dataset contains `NaN` values (failed experiments). These have been replaced by `0.0` in the `y` tensor to prevent crashes, but **you must not train on them.**

**Critical: Implementing Masked Loss**
You must use the mask to zero out the loss for invalid data points. Standard `MSELoss` will try to force your model to predict `0.0` for `NaNs`, which is incorrect.

## 5. Helpers you should use

**Seed configuration:** `utils.configure_seed`

**Loss function:** `utils.masked_mse_loss`

**Correlation metric:** `utils.masked_spearman_correlation`

**Metrics over epochs plot function:** `utils.plot`