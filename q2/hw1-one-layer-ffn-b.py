#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import utils.utils as utils


class FeedforwardNetwork(nn.Module):

    ACTIVATION_MAP = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }

    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()

    

        if activation_type not in self.ACTIVATION_MAP:
            raise ValueError(f"unsupported activation type: {activation_type}")
        self.activation = self.ACTIVATION_MAP[activation_type]()

        model = []

        current_input_size = n_features
        for _ in range(layers):
            model.append(nn.Linear(current_input_size, hidden_size))
            model.append(self.activation)
            model.append(nn.Dropout(dropout))
            current_input_size = hidden_size

        model.append(nn.Linear(hidden_size, n_classes))

        self.model = nn.Sequential(*model)


    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """

        return self.model(x)
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """

    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    model.eval()
    scores = model(X)
    return scores.argmax(dim=1)


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    model.eval()
    scores = model(X)
    preds = scores.argmax(dim=1)
    loss = criterion(scores, y)

    acc = (preds == y).float().mean()

    return loss.item(), acc.item()


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


def train_single_config(train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y, n_feats, n_classes,
                        hidden_size, layers, dropout, learning_rate, l2_penalty, activation, optimizer_name,
                        epochs, config_name):
    """
    Train a single FFN configuration and return reults
    """
    print(f"\n{'='*80}")
    print(f"Training configuration: {config_name}")
    print(f"Hidden: {hidden_size}, LR: {learning_rate}, L2: {l2_penalty}, Drop: {dropout}")
    print(f"{'='*80}\n")

    # Initialize the model with specific hyperparameters
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        layers,
        activation,
        dropout
    )

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    best_valid_acc = 0.0
    best_epoch = -1

    for i in range(1, epochs + 1):
        print(f'Training epoch {i}')
        
        model.train()
        for X_batch, y_batch in train_dataloader:
            train_batch(X_batch, y_batch, model, optimizer, criterion)

        _, train_acc = evaluate(model, train_X, train_y, criterion)
        _, valid_acc = evaluate(model, dev_X, dev_y, criterion)

        print(f'train acc: {train_acc:.4f} | val acc: {valid_acc:.4f}')

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = i


    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f'Training took {minutes} minutes and {seconds} seconds')

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    
    print(f'Best valid acc (epoch {best_epoch}): {best_valid_acc:.4f}')
    print(f'Final test acc: {test_acc:.4f}')

    return {
        "config_name": config_name,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "l2_penalty": l2_penalty,
        "dropout": dropout,
        "best_valid_acc": float(best_valid_acc),
        "test_acc": float(test_acc),
        "best_epoch": int(best_epoch),
        "training_time": float(elapsed_time)
    }

def run_best_model_and_plot(train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
                            n_feats, n_classes, hidden_size, lr, l2, dropout, activation, optimizer, epochs, model_name):
    """
    Retrains the BEST configuration, records history, and generates plots.
    """
    print(f"\n{'='*40}")
    print(f"RETRAINING BEST MODEL: {model_name}")
    print(f"Hidden: {hidden_size}, LR: {lr}, L2: {l2}, Drop: {dropout}")
    print(f"{'='*40}")

    model = FeedforwardNetwork(n_classes, n_feats, hidden_size, 1, activation, dropout)
    
    if optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    valid_accs = []
    plot_epochs = range(0, epochs + 1) # 0 to 30

    model.eval()
    init_loss, _ = evaluate(model, train_X, train_y, criterion)
    _, init_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(init_loss)
    valid_accs.append(init_val_acc)

    for i in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optim, criterion)
            epoch_losses.append(loss)
        
        avg_train_loss = torch.tensor(epoch_losses).mean().item()
        train_losses.append(avg_train_loss)

        _, val_acc = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        
        print(f"Epoch {i}: Train Loss {avg_train_loss:.4f}, Val Acc {val_acc:.4f}")

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    plot(plot_epochs, {"Train Loss": train_losses}, filename=f'q2/plots/{model_name}-loss.pdf')
    
    plot(plot_epochs, {"Valid Accuracy": valid_accs}, filename=f'q2/plots/{model_name}-accuracy.pdf')
    
    print(f"Plots saved to q2/plots/{model_name}-loss.pdf and q2/plots/{model_name}-accuracy.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='data/emnist-letters.npz',)
    parser.add_argument('-model', type=str, default='ffn', 
                        help="Name of the model for file saving.")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # Fill in with the best hyperparameters found from grid search from hw1-one-layer-ffn-a.py
    best_hidden = 256
    best_lr = 0.1
    best_l2 = 0.0
    best_drop = 0.0

    run_best_model_and_plot(
        train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
        n_feats, n_classes,
        hidden_size=best_hidden,
        lr=best_lr,
        l2=best_l2,
        dropout=best_drop,
        activation='relu',
        optimizer='sgd',
        epochs=30,
        model_name="best_model_ffn"
    )

if __name__ == '__main__':
    main()
