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
        "final_train_acc": float(train_acc),
        "best_epoch": int(best_epoch),
        "training_time": float(elapsed_time)
    }


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

    # ------ (a) ------
    # Define grid search parameters
    hidden_sizes = [16, 32, 64, 128, 256]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    l2_penalties = [0.0, 0.0001]
    dropouts = [0.0, 0.5]
    
    results = []

    total_configs = len(hidden_sizes) * len(learning_rates) * len(l2_penalties) * len(dropouts)
    config_num = 0
    print(f"\nStarting Grid Search with {total_configs} configurations")
    print(f"Optimizer: {opt.optimizer}, Activation: {opt.activation}, Epochs: {opt.epochs}")

    for hs in hidden_sizes:
        print(f"\nChecking Hidden Size: {hs}")
        
        best_acc_for_width = 0.0
        best_config_for_width = ""
        
        for lr in learning_rates:
            for l2 in l2_penalties:
                for drop in dropouts:
                    config_num += 1
                    config_name = f"h{hs}_lr{lr}_l2{l2}_drop{drop}"
                    
                    print(f"\n[Configuration {config_num}/{total_configs}]")
                    
                    result = train_single_config(
                        train_dataloader=train_dataloader,
                        train_X=train_X, train_y=train_y,
                        dev_X=dev_X, dev_y=dev_y,
                        test_X=test_X, test_y=test_y,
                        n_feats=n_feats,           
                        n_classes=n_classes,        
                        hidden_size=hs,
                        layers=1,                   
                        dropout=drop,
                        learning_rate=lr,
                        l2_penalty=l2,
                        activation=opt.activation,  
                        optimizer_name=opt.optimizer,
                        epochs=opt.epochs,
                        config_name=config_name
                    )
                    
                    results.append(result)
                    
                    # Find best configuration for this hidden size
                    if result['best_valid_acc'] > best_acc_for_width:
                        best_acc_for_width = result['best_valid_acc']
                        best_config_for_width = config_name

        print(f"BEST CONFIGURATION for Hidden {hs}: {best_config_for_width} (Val Acc: {best_acc_for_width:.4f})")

    # Print summary table
    print("\n" + "="*90)
    print(f"{'Hidden':<10} {'LR':<10} {'L2':<10} {'Dropout':<10} {'Best Val Acc':<15} {'Test Acc':<15}")
    print("="*90)
    
    results.sort(key=lambda x: (x['hidden_size'], -x['best_valid_acc']))
    
    for r in results:
        print(f"{r['hidden_size']:<10} {r['learning_rate']:<10} {r['l2_penalty']:<10} {r['dropout']:<10} {r['best_valid_acc']:.4f} {r['test_acc']:.4f}")
    print("="*90)

    # ------ (c) ------

    best_train_accs = []
    widths = sorted(hidden_sizes)

    print("\nAnalyzing best models for each width")

    for width in widths:
        width_results = [r for r in results if r['hidden_size'] == width]
        
        best_model = max(width_results, key=lambda x: x['best_valid_acc'])
        
        train_acc = best_model['final_train_acc']
        best_train_accs.append(train_acc)
        
        print(f"Width {width}: Best Config {best_model['config_name']} -> Final Train Acc: {train_acc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(widths, best_train_accs, marker='o', linestyle='-', color='b')
    plt.xscale('log', base=2) # logarithmic scale, because widths increase exponentially
    plt.xticks(widths, widths)
    plt.xlabel('Hidden Layer Width')
    plt.ylabel('Final Training Accuracy')
    plt.title('Effect of Width on Training Interpolation')
    plt.grid(True)
    plt.savefig('q2/plots/width_vs_train_acc.pdf')
    print("Plot saved to q2/plots/width_vs_train_acc.pdf")

if __name__ == '__main__':
    main()
