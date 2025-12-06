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


class MultiLayerPerceptron:
    def __init__(self, n_classes, n_features):
        self.n_hidden_units = 100

        self.W_L1 = np.random.normal(loc=0.1, scale =0.1, size=(self.n_hidden_units, n_features))
        self.W_L2 = np.random.normal(loc=0.1, scale =0.1, size=(n_classes, self.n_hidden_units))
        self.b_2 = np.zeros((n_classes, ))

        self.learning_rate = 0.001

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
            z_l1 = self.W_L1 @ X[i]
            h_l1 = self.relu(z_l1)

            z_l2 = self.W_L2 @ h_l1 + self.b_2
            z_max = np.max(z_l2)
            exp_logits = np.exp(z_l2 - z_max)
            P = exp_logits / np.sum(exp_logits)

            ey = np.zeros(self.W_L2.shape[0])
            ey[y[i]] = 1.0

            gradient_w_l2 = np.outer(P - ey, h_l1)
            gradient_b_l2 = (P - ey)

            gradient_h_l1 = self.W_L2.T @ (P - ey)
            relu_derivative = (h_l1 > 0).astype(float)
            gradient_z_l1 = gradient_h_l1 * relu_derivative

            gradient_w_l1 = np.outer(gradient_z_l1, X[i])

            self.W_L2 = self.W_L2 - self.learning_rate * gradient_w_l2
            self.b_2 = self.b_2 - self.learning_rate * gradient_b_l2

            self.W_L1 = self.W_L1 - self.learning_rate * gradient_w_l1


    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        h_l1 = self.relu(self.W_L1 @ X.T)  # (n_hidden_units, n_examples)
        z_l2 = self.W_L2 @ h_l1 + self.b_2[:, np.newaxis]  # (n_classes, n_examples)
        scores = self.softmax(z_l2)
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


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = MultiLayerPerceptron(n_classes, n_feats)

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
    best_model = MultiLayerPerceptron.load(args.save_path)
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
    parser.add_argument("--save-path", default="q1/checkpoints/checkpoint-mlp.pickle")
    parser.add_argument("--accuracy-plot", default="q1/plots/Q1-mlp-accs.pdf")
    parser.add_argument("--scores", default="q1/scores/Q1-mlp-scores.json")
    args = parser.parse_args()
    main(args)
