#!/usr/bin/env python

# Deep Learning Homework 1 - Grid Search for Hyperparameter Tuning

import argparse
import time
import pickle
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import utils

class LogisticRegressor:
    def __init__(self, n_classes, n_features, learning_rate=0.0001, l2_penalty=0.00001):
        self.W = np.zeros((n_classes, n_features))
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def save(self, path):
        """
        Save to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        for i in range(X.shape[0]):
            logits = self.W @ X[i]
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            P = exp_logits / np.sum(exp_logits)

            ey = np.zeros(self.W.shape[0])
            ey[y[i]] = 1.0
            gradient = np.outer(P - ey, X[i]) + self.l2_penalty * self.W

            self.W = self.W - self.learning_rate * gradient

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        scores = self.W @ X.T
        return np.argmax(scores, axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        y_hat = self.predict(X)
        num_correct = 0
        for i in range(X.shape[0]):
            pred_class = y_hat[i]
            true_class = y[i]
            if pred_class == true_class:
                num_correct += 1

        accuracy = num_correct/X.shape[0]
        return accuracy


def feature_extractor_original(X):
    """
    Original feature representation - just return the raw pixels + bias
    X: (n_examples, 785) - flattened 28x28 images + bias term
    Returns: (n_examples, 785) - original representation
    """
    return X


def feature_extractor_hog(X):
    """
    X: (n_examples, 785) - flattened 28x28 images + bias term
    Returns: (n_examples, 1226) - original pixels + HOG features + bias

    Applies Histogram of Oriented Gradients (HOG) feature extraction:
    - Computes gradient magnitude and orientation at each pixel
    - Divides 28x28 image into 7x7 grid of 4x4 pixel cells
    - Creates 9-bin histogram of gradient orientations per cell (0-180 degrees)
    - Results in 7x7x9 = 441 HOG features per image
    """
    n_examples = X.shape[0]

    # Separate the bias term (last column)
    X_pixels = X[:, :-1]  # (n_examples, 784)
    bias = X[:, -1:]      # (n_examples, 1)

    # Reshape to (n_examples, 28, 28)
    X_reshaped = X_pixels.reshape(n_examples, 28, 28)

    # Compute gradients using central differences
    gx = np.zeros_like(X_reshaped)
    gy = np.zeros_like(X_reshaped)
    gx[:, :, 1:-1] = (X_reshaped[:, :, 2:] - X_reshaped[:, :, :-2]) / 2.0
    gy[:, 1:-1, :] = (X_reshaped[:, 2:, :] - X_reshaped[:, :-2, :]) / 2.0

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi  # Convert to degrees
    orientation = (orientation + 180) % 180  # Map to [0, 180)

    # HOG parameters
    cell_size = 4  # 4x4 pixels per cell (28/4 = 7 cells per dimension)
    n_bins = 9     # 9 orientation bins (0-180 degrees, 20 degrees per bin)
    n_cells = 28 // cell_size  # 7 cells per dimension

    # Extract HOG features for all examples
    hog_features = []

    for img_idx in range(n_examples):
        cell_histograms = []

        for i in range(n_cells):
            for j in range(n_cells):
                # Extract cell region
                cell_mag = magnitude[img_idx,
                                   i*cell_size:(i+1)*cell_size,
                                   j*cell_size:(j+1)*cell_size]
                cell_orient = orientation[img_idx,
                                        i*cell_size:(i+1)*cell_size,
                                        j*cell_size:(j+1)*cell_size]

                # Compute histogram of oriented gradients for this cell
                hist = np.zeros(n_bins)
                bin_width = 180.0 / n_bins  # 20 degrees per bin

                for y in range(cell_size):
                    for x in range(cell_size):
                        angle = cell_orient[y, x]
                        mag = cell_mag[y, x]
                        bin_idx = int(angle / bin_width) % n_bins
                        hist[bin_idx] += mag

                cell_histograms.append(hist)

        # Flatten all cell histograms for this image
        # 7x7 cells x 9 bins = 441 features
        hog_features.append(np.concatenate(cell_histograms))

    hog_features = np.array(hog_features)

    # Concatenate: original pixels + HOG features + bias
    # Total: 784 + 441 + 1 = 1226 features
    return np.concatenate([X_pixels, hog_features, bias], axis=1)


def train_single_config(X_train, y_train, X_valid, y_valid, X_test, y_test,
                       learning_rate, l2_penalty, epochs, config_name, save_dir,
                       n_classes, n_feats):
    """
    Train a single configuration and return results
    """
    print(f"\n{'='*80}")
    print(f"Training configuration: {config_name}")
    print(f"Learning rate: {learning_rate}, L2 penalty: {l2_penalty}")
    print(f"{'='*80}\n")

    # Initialize the model with specific hyperparameters
    model = LogisticRegressor(n_classes, n_feats, learning_rate, l2_penalty)

    epoch_range = np.arange(1, epochs + 1)
    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1

    # Create checkpoint path
    checkpoint_path = save_dir / f"checkpoint-{config_name}.pickle"

    for i in epoch_range:
        print(f'Training epoch {i}')
        train_order = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[train_order]
        y_train_shuffled = y_train[train_order]

        model.train_epoch(X_train_shuffled, y_train_shuffled)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(f'train acc: {train_acc:.4f} | val acc: {valid_acc:.4f}')

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(checkpoint_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f'Training took {minutes} minutes and {seconds} seconds')

    # Load best model and evaluate on test set
    print("Reloading best checkpoint")
    best_model = LogisticRegressor.load(checkpoint_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print(f'Best model (epoch {best_epoch}) - val acc: {best_valid:.4f}, test acc: {test_acc:.4f}')

    return {
        "config_name": config_name,
        "learning_rate": float(learning_rate),
        "l2_penalty": float(l2_penalty),
        "best_valid_acc": float(best_valid),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "training_time": float(elapsed_time)
    }


def main(args):
    utils.configure_seed(seed=args.seed)

    # Load data
    print("Loading dataset...")
    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]

    # Define grid search parameters
    learning_rates = [0.001, 0.0001, 0.00001]
    l2_penalties = [0.0001, 0.00001]
    feature_types = ["original", "hog"]

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run grid search
    total_configs = len(learning_rates) * len(l2_penalties) * len(feature_types)
    config_num = 0

    n_classes = np.unique(y_train).size

    for feature_type in feature_types:
        print(f"\n{'#'*80}")
        print(f"# Feature Type: {feature_type.upper()}")
        print(f"{'#'*80}\n")

        # Apply feature extraction
        if feature_type == "original":
            X_train = feature_extractor_original(X_train_raw)
            X_valid = feature_extractor_original(X_valid_raw)
            X_test = feature_extractor_original(X_test_raw)
        else:  # hog
            print("Extracting HOG features (this may take a moment)...")
            X_train = feature_extractor_hog(X_train_raw)
            X_valid = feature_extractor_hog(X_valid_raw)
            X_test = feature_extractor_hog(X_test_raw)
            print("Feature extraction complete!")

        n_feats = X_train.shape[1]

        for lr in learning_rates:
            for l2 in l2_penalties:
                config_num += 1
                config_name = f"{feature_type}_lr{lr}_l2{l2}"

                print(f"\n[Configuration {config_num}/{total_configs}]")

                result = train_single_config(
                    X_train, y_train, X_valid, y_valid, X_test, y_test,
                    lr, l2, args.epochs, config_name, checkpoint_dir,
                    n_classes, n_feats
                )
                result["feature_type"] = feature_type
                results.append(result)

    # Find best configuration
    best_config = max(results, key=lambda x: x["best_valid_acc"])

    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Config':<30} {'Feature':<10} {'LR':<10} {'L2':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-"*80)

    for result in results:
        print(f"{result['config_name']:<30} "
              f"{result['feature_type']:<10} "
              f"{result['learning_rate']:<10.6f} "
              f"{result['l2_penalty']:<10.6f} "
              f"{result['best_valid_acc']:<10.4f} "
              f"{result['test_acc']:<10.4f}")

    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Config name: {best_config['config_name']}")
    print(f"Feature type: {best_config['feature_type']}")
    print(f"Learning rate: {best_config['learning_rate']}")
    print(f"L2 penalty: {best_config['l2_penalty']}")
    print(f"Best validation accuracy: {best_config['best_valid_acc']:.4f}")
    print(f"Best epoch: {best_config['best_epoch']}")
    print(f"Test accuracy: {best_config['test_acc']:.4f}")
    print("="*80)

    # Save results to JSON
    scores_file = Path(args.scores)
    scores_file.parent.mkdir(parents=True, exist_ok=True)

    with open(scores_file, "w") as f:
        json.dump({
            "all_results": results,
            "best_config": best_config
        }, f, indent=4)

    print(f"\nResults saved to: {scores_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid search for logistic regression hyperparameters")
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number of epochs to train for each configuration")
    parser.add_argument('--data-path', type=str, default="data/emnist-letters.npz",
                        help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=str, default="q1/checkpoints/checkpoint-lr-c",
                        help="Directory to save model checkpoints")
    parser.add_argument("--scores", type=str, default="q1/scores/Q1-lr-scores-c.json",
                        help="Path to save results JSON file")
    args = parser.parse_args()
    main(args)
