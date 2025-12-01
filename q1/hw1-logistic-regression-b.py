#!/usr/bin/env python

# Deep Learning Homework 1

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
    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features))
        self.learning_rate = 0.0001
        self.l2_penalty = 0.00001

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


def feature_extractor(X):
    """
    X: (n_examples, 785) - flattened 28x28 images + bias term
    Returns: (n_examples, 813) - images with row indices added before each row + bias
    """
    n_examples = X.shape[0]

    # Separate the bias term (last column)
    X_pixels = X[:, :-1]  # (n_examples, 784)
    bias = X[:, -1:]      # (n_examples, 1)

    # Reshape to (n_examples, 28, 28)
    X_reshaped = X_pixels.reshape(n_examples, 28, 28)

    # Create row indices (0-27) and reshape to broadcast
    row_indices = np.arange(28).reshape(1, 28, 1)

    # Repeat row indices for all examples
    row_indices = np.broadcast_to(row_indices, (n_examples, 28, 1))

    # Concatenate row indices before each row of 28 pixels
    X_with_indices = np.concatenate([row_indices, X_reshaped], axis=2)

    # Flatten back to (n_examples, 812) - each row now has 29 values (1 index + 28 pixels)
    X_flattened = X_with_indices.reshape(n_examples, -1)

    # Append the bias term back at the end
    return np.concatenate([X_flattened, bias], axis=1)

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]

    X_train = feature_extractor(X_train_raw)
    X_valid = feature_extractor(X_valid_raw)
    X_test = feature_extractor(X_test_raw)

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = LogisticRegressor(n_classes, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = LogisticRegressor.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="data/emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="q1/checkpoints/checkpoint-lr-b.pickle")
    parser.add_argument("--accuracy-plot", default="q1/plots/Q1-lr-accs-b.pdf")
    parser.add_argument("--scores", default="q1/scores/Q1-lr-scores-b.json")
    args = parser.parse_args()
    main(args)
