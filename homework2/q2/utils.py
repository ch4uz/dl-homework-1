import os
import random
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

try:
    from config import RNAConfig
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RNACompeteLoader:
    def __init__(self, config: RNAConfig):
        """
        Initializes the loader.
        """
        self.cfg = config
        self.meta_df = None
        self.data_df = None
        self.protein_to_id = None

        # Setup Encoding
        self.char_map = {
            'A': np.array([1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 1, 0, 0], dtype=np.float32),
            'G': np.array([0, 0, 1, 0], dtype=np.float32),
            'U': np.array([0, 0, 0, 1], dtype=np.float32),
            'N': np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        }
        self.padding_vec = np.zeros(4, dtype=np.float32)

    def _ensure_data_loaded(self):
        """Helper to load the heavy files only when necessary."""
        if self.data_df is not None:
            return
        
        # Load Metadata
        print(f"Loading Metadata from {self.cfg.METADATA_PATH}...")
        start_time = time.time()
        try:
            if self.cfg.METADATA_PATH.endswith('.xlsx'):
                # Requires 'openpyxl' installed!
                self.meta_df = pd.read_excel(
                    self.cfg.METADATA_PATH, 
                    sheet_name=self.cfg.METADATA_SHEET
                )
            else:
                self.meta_df = pd.read_csv(self.cfg.METADATA_PATH)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise e
        print(f"  > Metadata loaded in {time.time() - start_time:.2f} seconds.")

        # Clean column names (strip whitespace)
        self.meta_df.columns = [c.strip() for c in self.meta_df.columns]
        
        # Create Protein Name -> RNCMPT ID mapping
        self.protein_to_id = pd.Series(
            self.meta_df['Motif_ID'].values, 
            index=self.meta_df['Protein_name']
        ).to_dict()
        
        # Load Data 
        print(f"Loading Data from {self.cfg.DATA_PATH}...")
        start_time = time.time()

        # standard RNAcompete is tab-separated
        self.data_df = pd.read_csv(self.cfg.DATA_PATH, sep='\t', low_memory=False)    
        print(f"  > Data Matrix loaded in {time.time() - start_time:.2f} seconds.")

        # Clean data columns
        self.data_df.columns = [c.strip() for c in self.data_df.columns]

    def list_proteins(self) -> List[str]:
        """Returns a sorted list of available protein names."""
        self._ensure_data_loaded()
        valid_proteins = []
        matrix_cols = set(self.data_df.columns)
        
        for name, pid in self.protein_to_id.items():
            if pid in matrix_cols:
                valid_proteins.append(name)
        
        return sorted(valid_proteins)
    
    def _encode_sequence(self, seq: str) -> np.ndarray:
        """One-hot encodes a single RNA sequence."""
        # Handle NaN or non-string sequence entries gracefully
        if not isinstance(seq, str):
            seq = "N" * self.cfg.SEQ_MAX_LEN

        seq = seq.upper()[:self.cfg.SEQ_MAX_LEN]
        encoded = [self.char_map.get(base, self.char_map['N']) for base in seq]
        
        pad_len = self.cfg.SEQ_MAX_LEN - len(encoded)
        if pad_len > 0:
            encoded.extend([self.padding_vec] * pad_len)
            
        return np.array(encoded, dtype=np.float32)
    
    def _preprocess_intensities(self, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies: Mask NaNs -> Clip -> Log -> Z-score."""
        mask = (~np.isnan(intensities)).astype(np.float32)
        clean_vals = np.nan_to_num(intensities, nan=0.0)
        
        # Clip
        if np.sum(mask) > 0:
            valid_data = intensities[mask == 1]
            clip_val = np.percentile(valid_data, self.cfg.CLIP_PERCENTILE)
            clean_vals = np.clip(clean_vals, None, clip_val)

        # Log Transform (Shift to positive)
        min_val = np.min(clean_vals)
        shift = 0
        if min_val <= 0:
            shift = abs(min_val) + 1.0
        clean_vals = np.log(clean_vals + shift + self.cfg.EPSILON)
        
        # Z-Score
        masked_vals = clean_vals[mask == 1]
        if len(masked_vals) > 0:
            mean = np.mean(masked_vals)
            std = np.std(masked_vals) + self.cfg.EPSILON
            clean_vals = (clean_vals - mean) / std
        
        clean_vals = clean_vals * mask
        return clean_vals, mask
    
    def get_data(self, protein_name: str, split: str = 'train') -> TensorDataset:
        """
        Main method to get PyTorch Dataset for a specific protein.
        """
        # Check cache first
        os.makedirs(self.cfg.SAVE_DIR, exist_ok=True)
        data_path = os.path.join(self.cfg.SAVE_DIR, f"{protein_name}_{split}_data.pt")

        if os.path.exists(data_path):
            print(f"Found cached data for {protein_name} ({split}). Loading from {data_path}...")
            try:
                tensors = torch.load(data_path, weights_only=True)
                return TensorDataset(*tensors)
            except Exception as e:
                print(f"Cache seems corrupted: {e}. Will reload from scratch.")

        self._ensure_data_loaded()

        if protein_name not in self.protein_to_id:
            raise ValueError(f"Protein '{protein_name}' not found in metadata.")
        
        rncmpt_id = self.protein_to_id[protein_name]
        
        if rncmpt_id not in self.data_df.columns:
            raise ValueError(f"ID {rncmpt_id} for {protein_name} missing from data matrix.")

        s_lower = split.lower()

        if s_lower == 'test':
            # Test set is just SetB, nice and simple
            subset = self.data_df[self.data_df['Probe_Set'] == self.cfg.TEST_SPLIT_ID].copy()

        elif s_lower in ['train', 'val']:
            # For train/val, we need to split SetA. 
            # We use a fixed seed to ensure grading consistency (everyone gets the same split).
            full_set = self.data_df[self.data_df['Probe_Set'] == self.cfg.TRAIN_SPLIT_ID]
            
            # Explicitly sort by index to ensure deterministic order before shuffling
            full_set = full_set.sort_index()
            
            n_samples = len(full_set)
            indices = np.arange(n_samples)
            
            # Local RandomState prevents messing with global seeds
            rng = np.random.RandomState(self.cfg.SEED)
            rng.shuffle(indices)
            
            val_size = int(n_samples * self.cfg.VAL_SPLIT_PCT)
            
            if s_lower == 'val':
                # Validation gets the first chunk
                subset_indices = indices[:val_size]
            else:
                # Train gets the leftovers
                subset_indices = indices[val_size:]
                
            subset = full_set.iloc[subset_indices].copy()
        else:
            raise ValueError(f"Unknown split '{split}'. Please use 'train', 'val', or 'test'.")
        
        # Extract Sequences
        raw_seqs = subset['RNA_Seq'].values
        X = np.stack([self._encode_sequence(s) for s in raw_seqs])
        
        # Process Intensities
        # Force conversion to numeric (floats), turning any strings/errors into NaN
        raw_intensities = pd.to_numeric(subset[rncmpt_id], errors='coerce').values
        Y, mask = self._preprocess_intensities(raw_intensities)
        
        # Convert to Tensor
        dataset = TensorDataset(
            torch.FloatTensor(X),                     # (B, 41, 4)
            torch.FloatTensor(Y).unsqueeze(1),        # (B, 1)
            torch.FloatTensor(mask).unsqueeze(1)      # (B, 1)
        )
        
        # Save for next time
        print(f"Saving processed data to {data_path}...")
        torch.save(dataset.tensors, data_path)
        
        return dataset
    

def load_rnacompete_data(protein_name: str, split: str = 'train', config: RNAConfig = None):
    """
    Convenience function to load data for a single protein without manually managing the loader class.
    Note: Instantiates the loader from scratch (loads files). 
    For bulk processing, use RNACompeteLoader class directly.
    """
    if config is None:
        config = RNAConfig()

    loader = RNACompeteLoader(config)
    return loader.get_data(protein_name, split)


def masked_spearman_correlation(preds, targets, masks):
    """
    Calculates Spearman Rank Correlation on masked data.
    Expects:
        preds: (B, 1)
        targets: (B, 1)
        masks: (B, 1)
    Outputs:
        correlation: scalar
    """
    # Flatten and detach (metrics don't need gradients)
    preds = preds.squeeze().detach()
    targets = targets.squeeze().detach()
    masks = masks.squeeze().bool()
    
    valid_preds = preds[masks]
    valid_targets = targets[masks]
    
    if valid_preds.numel() < 2:
        return torch.tensor(0.0)

    # argsort twice gets us the ranks
    pred_ranks = valid_preds.argsort().argsort().float()
    target_ranks = valid_targets.argsort().argsort().float()

    # Pearson on ranks == Spearman
    pred_mean = pred_ranks.mean()
    target_mean = target_ranks.mean()

    pred_var = pred_ranks - pred_mean
    target_var = target_ranks - target_mean

    correlation = (pred_var * target_var).sum() / torch.sqrt((pred_var ** 2).sum() * (target_var ** 2).sum())

    return correlation


def masked_mse_loss(preds, targets, masks):
    """
    Calculates Mean Squared Error, ignoring padded elements.
    Expects:
        preds: (B, 1)
        targets: (B, 1)
        masks: (B, 1)
    Outputs:
        loss: scalar
    """
    # Flatten to 1D
    preds = preds.squeeze()
    targets = targets.squeeze()
    masks = masks.squeeze().bool()

    # Filter out padded values
    masked_preds = preds[masks]
    masked_targets = targets[masks]
    
    # Handle empty batch case
    if masked_preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    # MSE on valid data
    squared_error = (masked_preds - masked_targets) ** 2
    loss = torch.mean(squared_error)
    
    return loss


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')