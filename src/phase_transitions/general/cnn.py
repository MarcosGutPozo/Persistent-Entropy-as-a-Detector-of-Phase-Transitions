import os
import random

import numpy as np

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_cifar10_rgb_stats(root="./data"):
    # Load CIFAR-10 training images as tensors
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=T.ToTensor() # Convierte a [3, 32, 32] en rango [0, 1]
    )
    
    # Load the complete training set
    loader = DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=2)
    imgs, _ = next(iter(loader)) # [50000, 3, 32, 32]
    
    # Compute mean and standard deviation for each RGB channel
    means = imgs.mean(dim=(0, 2, 3)).tolist()
    stds = imgs.std(dim=(0, 2, 3)).tolist()
    
    return means, stds

def make_loaders(dataset="digits", batch_size=128, num_workers=4):
    dataset = dataset.lower()

    # =====================================================================
    # CASO 1: DIGITS (Scikit-Learn - Mantenemos un split estándar 80/20)
    # =====================================================================
    if dataset == "digits":
        X, y = load_digits(return_X_y=True)  # (N, 64)
        X = StandardScaler().fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=0, stratify=y
        )

        train = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        test = TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long),
        )

        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test, batch_size=batch_size * 2, shuffle=False, num_workers=0
        )

        return train_loader, test_loader, (64,), 10

    # =====================================================================
    # CASO 2: MNIST / FASHIONMNIST / CIFAR10 (Fieles al Artículo)
    # =====================================================================
    elif dataset in ["mnist", "fashionmnist", "fmnist", "cifar10"]:
        if dataset in ["fashionmnist", "fmnist"]:
            ds = torchvision.datasets.FashionMNIST
            name = "FashionMNIST"
            transform_train = T.Compose([T.ToTensor()])
            transform_test = T.Compose([T.ToTensor()])
            input_size = (1, 28, 28)

        elif dataset == "mnist":
            ds = torchvision.datasets.MNIST
            name = "MNIST"
            transform_train = T.Compose([T.ToTensor()])
            transform_test = T.Compose([T.ToTensor()])
            input_size = (1, 28, 28)

        else:
            ds = torchvision.datasets.CIFAR10
            name = "CIFAR10 (Color Experiment)"

            # 1. Obtenemos las medias y stds de los 3 canales (listas de 3 elementos)
            rgb_means, rgb_stds = compute_cifar10_rgb_stats(root="./data")

            # 2. Transformaciones de entrenamiento: Random Crop de 24x24, Flip Horizontal y Normalización RGB
            transform_train = T.Compose([
                T.RandomCrop(24), 
                T.RandomHorizontalFlip(p=0.5), 
                T.ToTensor(), 
                T.Normalize(mean=rgb_means, std=rgb_stds)
            ])

            # 3. Transformaciones de test: Center Crop de 24x24 y Normalización RGB
            transform_test = T.Compose([
                T.CenterCrop(24), 
                T.ToTensor(), 
                T.Normalize(mean=rgb_means, std=rgb_stds)
            ])
                                        
            # ¡Ojo! Cambia a 3 canales (RGB) y mantiene los 24x24 píxeles del crop
            input_size = (3, 24, 24)  

        root = "./data"

        # Cargamos pasando sus respectivas transformaciones asignadas arriba
        train_full = ds(
            root=root, train=True, download=True, transform=transform_train
        )
        test_full = ds(
            root=root, train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_full,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_full,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        num_classes = 10

        print(
            f"¡Configuración de COLOR lista! {name} -> Train: {len(train_full)} | Test: {len(test_full)}"
        )

        return train_loader, test_loader, input_size, num_classes

    else:
        raise ValueError(f"Unknown dataset: {dataset}")



def run_experiment(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    save_every=100,
    iterations=5000
):
    # Train the model and periodically save metrics and filters

    model.to(device)
    model.train()

    history = {
        "iteration": [],
        "val_acc": [],
        "filters_layer1": [],
        "filters_layer2": [],
        "loss": [],
    }

    iteration = 0
    data_iter = iter(train_loader)

    while iteration < iterations:

        # ======================
        # obtener batch
        # ======================
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # ======================
        # forward + backward
        # ======================
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        iteration += 1

        if iteration % save_every == 0:

            model.eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb).argmax(1)
                    correct += (pred == yb).sum().item()
                    total += yb.size(0)

            acc = correct / total

            with torch.no_grad():

             # -------- conv1 --------
                w1 = model.conv1.weight.data.clone()
                filters1 = w1.view(-1, 3 * 3)

                filters1 = filters1 - filters1.mean(dim=1, keepdim=True)
                filters1 = filters1 / (filters1.norm(dim=1, keepdim=True) + 1e-8)

            # -------- conv2 --------
                w2 = model.conv2.weight.data.clone()
                filters2 = w2.view(-1, 3 * 3)

                filters2 = filters2 - filters2.mean(dim=1, keepdim=True)
                filters2 = filters2 / (filters2.norm(dim=1, keepdim=True) + 1e-8)

            history["iteration"].append(iteration)
            history["val_acc"].append(acc)
            history["loss"].append(loss.item())
            history["filters_layer1"].append(filters1.cpu())
            history["filters_layer2"].append(filters2.cpu())

            print(
                 f"iter {iteration:5d} | "
                 f"loss {loss.item():.4f} | "
                 f"acc {acc:.4f} "
                )

            model.train()

    return history