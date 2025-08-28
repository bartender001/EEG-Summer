import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import time
import mlflow
import mlflow.pytorch

from src.dataset import EEGDataset
from src.model import EEG_BiGRU_CNN as EEG_CNN_Simple

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def run_kfold(csv_path, device, k=5, epochs=50, batch_size=32, lr=5e-4):
    dataset = EEGDataset(csv_path)
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_error_rates = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
            print(f'----- Fold {fold+1}/{k} -----')

            mlflow.log_param("fold", fold + 1)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("model_name", EEG_CNN_Simple.__name__)

            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = EEG_CNN_Simple(
                input_features=dataset.X.shape[2],
                num_classes=dataset.num_classes
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            best_val_acc = 0.0
            best_model_state = None

            for epoch in range(epochs):
                start_time = time.time()
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
                scheduler.step()

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed_time:.2f}s')

            fold_error = 1 - best_val_acc
            fold_error_rates.append(fold_error)
            
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("final_error_rate", fold_error)
            if best_model_state:
                model.load_state_dict(best_model_state)
                mlflow.pytorch.log_model(model, f"model_fold_{fold+1}")

            print(f'Fold {fold+1} Best Validation Accuracy: {best_val_acc:.4f}, Error Rate: {fold_error:.4f}')
            print("-" * 25)

    avg_error_rate = np.mean(fold_error_rates)
    print(f'\nAverage Error Rate across {k} folds: {avg_error_rate:.4f}')
    
    mlflow.log_metric("average_kfold_error_rate", avg_error_rate)


if __name__ == '__main__':
    CSV_FILE_PATH = "archive/BCICIV_2a_all_patients.csv" 
    NUM_EPOCHS = 50 
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    K_FOLDS = 5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    mlflow.set_experiment("EEG_Classification_KFold")
    with mlflow.start_run(run_name="KFold_Cross_Validation_Run"):
        run_kfold(
            csv_path=CSV_FILE_PATH, 
            device=DEVICE, 
            k=K_FOLDS, 
            epochs=NUM_EPOCHS, 
            batch_size=BATCH_SIZE, 
            lr=LEARNING_RATE
        )