#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import json
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
    A: Train a single FFN configuration and return reults
    """
    print(f"\n{'='*80}")
    print(f"Training configuration: {config_name}")
    print(f"Layers: {layers}, Hidden: {hidden_size}, LR: {learning_rate}, L2: {l2_penalty}, Drop: {dropout}")
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
    final_train_acc = 0.0


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
        
        if i == epochs:
            final_train_acc = train_acc

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
        "layers" : layers,
        "learning_rate": learning_rate,
        "l2_penalty": l2_penalty,
        "dropout": dropout,
        "activation": activation,
        "optimizer": optimizer_name,
        "best_valid_acc": float(best_valid_acc),
        "test_acc": float(test_acc),
        "final_train_acc": float(final_train_acc),
        "best_epoch": int(best_epoch),
        "training_time": float(elapsed_time),
    }


def run_best_model_and_plot(train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
                          n_feats, n_classes, config, epochs):
    """
    B: Retrain the best model configuration and plot training loss and validation accuracy
    """
    print(f"\n{'#'*60}")
    print(f"B: Retraining Global Best Model for Plots")
    print(f"Config: {config['config_name']}")
    print(f"{'#'*60}")

    model = FeedforwardNetwork(
        n_classes, n_feats,
        config['hidden_size'], config['layers'], 
        config['activation'], config['dropout']
    )
    
    if config['optimizer'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_penalty'])
    else:
        optim = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_penalty'])
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    valid_accs = []
    plot_epochs = range(0, epochs + 1)

    # Initial stats
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

    plot(plot_epochs, {"Train Loss": train_losses}, filename=f'q2/plots/Q2-ffn-best-model-loss-23b.pdf')
    plot(plot_epochs, {"Valid Accuracy": valid_accs}, filename=f'q2/plots/Q2-ffn-best-model-accuracy-23b.pdf')
 
    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print(f"\nFinal Test Accuracy of Best Model: {test_acc:.4f}")

    return {
        "config": config,
        "train_loss_history": train_losses,
        "valid_acc_history": valid_accs,
        "final_valid_acc": valid_accs[-1]
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
    parser.add_argument('-scores_23a', type=str, default='q2/scores/Q2-ffn-scores-23a.json')
    parser.add_argument('-scores_23b', type=str, default='q2/scores/Q2-ffn-scores-23b.json')
    parser.add_argument('-scores_23c', type=str, default='q2/scores/Q2-ffn-scores-23c.json')

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

    # ------ (A) ------
    # Define layers
    hidden_sizes = [32]
    layers = [1, 3, 5, 7, 9]
    learning_rates = [0.1]
    l2_penalties = [0.0001]
    dropouts = [0]
    
    a_results = []

    total_configs = len(layers) * len(hidden_sizes) * len(learning_rates) * len(l2_penalties) * len(dropouts)
    print(f"\nStarting Grid Search with {total_configs} configurations")
    print(f"Optimizer: {opt.optimizer}, Activation: {opt.activation}, Epochs: {opt.epochs}")

    config_num = 0
    for layer in layers:
        print(f"\nChecking Layers Size: {layer}")
        best_acc_for_size = 0.0
        best_config_for_size = ""
        for hs in hidden_sizes:
            for lr in learning_rates:
                for l2 in l2_penalties:
                    for drop in dropouts:
                        config_num += 1
                        config_name = f"n{layer}_h{hs}_lr{lr}_l2{l2}_drop{drop}"
                        
                        print(f"\n[Configuration {config_num}/{total_configs}]")
                        
                        result = train_single_config(
                            train_dataloader=train_dataloader,
                            train_X=train_X, train_y=train_y,
                            dev_X=dev_X, dev_y=dev_y,
                            test_X=test_X, test_y=test_y,
                            n_feats=n_feats,           
                            n_classes=n_classes,        
                            hidden_size=hs,
                            layers=layer,                   
                            dropout=drop,
                            learning_rate=lr,
                            l2_penalty=l2,
                            activation=opt.activation,  
                            optimizer_name=opt.optimizer,
                            epochs=opt.epochs,
                            config_name=config_name
                        )
                        
                        a_results.append(result)
                        
                        # Find best configuration for this hidden size
                        if result['best_valid_acc'] > best_acc_for_size:
                            best_acc_for_size = result['best_valid_acc']
                            best_config_for_size = config_name

    print(f"\nBEST CONFIGURATION for Hidden layers {layer}: {best_config_for_size} (Val Acc: {best_acc_for_size:.4f})")

    # Print summary table
    print("\n" + "="*90)
    print(f"{'Layers':<10}{'Hidden':<10} {'LR':<10} {'L2':<10} {'Dropout':<10} {'Best Val Acc':<15} {'Test Acc':<15}")
    print("="*90)
    
    a_results.sort(key=lambda x: (x['layers'], -x['best_valid_acc']))
    
    for r in a_results:
        print(f"{r['layers']:<10} {r['hidden_size']:<10} {r['learning_rate']:<10} {r['l2_penalty']:<10} {r['dropout']:<10} {r['best_valid_acc']:<15.4f} {r['test_acc']:<15.4f}")
    print("="*90)

    # Save results
    Path(opt.scores_23a).parent.mkdir(parents=True, exist_ok=True)
    with open(opt.scores_23a, "w") as f:
        json.dump({"q2_3a_search": a_results}, f, indent=4)


    # ------ (B) ------
    
    global_best_config = max(a_results, key=lambda x: x['best_valid_acc'])
    
    b_results = run_best_model_and_plot(
        train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
        n_feats, n_classes, global_best_config, opt.epochs
    )

    with open(opt.scores_23b, "w") as f:
        json.dump(b_results, f, indent=4)


    # ------ (C) ------

    best_train_accs = []
    layers = sorted(layers)
    best_acc_per_depth = []

    print(f"\n{'#'*60}")
    print("C: Analyzing models for each depth")
    print(f"{'#'*60}")

    for layer in layers:
        depth_result = next(r for r in a_results if r['layers'] == layer)
        best_acc_per_depth.append(depth_result['final_train_acc'])
        

    plt.figure(figsize=(8, 6))
    plt.plot(layers, best_acc_per_depth, marker='o', linestyle='-', color='b')
    plt.xlabel('Hidden Layers')
    plt.xticks(layers, layers)
    plt.ylabel('Final Training Accuracy')
    plt.title('Training accuracy in the function of depth')
    plt.grid(True)
    plt.savefig('q2/plots/Q2-ffn-depth-vs-train-acc-23c.pdf')
    print("saved figure")

    c_data = {
        "plot_data": {
            "layers": layers,
            "train_accuracies": best_acc_per_depth
        },
    }
    
    with open(opt.scores_23c, "w") as f:
        json.dump(c_data, f, indent=4)

if __name__ == '__main__':
    main()
