import torch
import wandb
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import UCIHAR_Dataset
# from model import ResNetTransformer
from positional_custom_model import ResNetTransformer


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(dataloader.dataset)


def run_training(args):
    from data_loader import load_uci_har_data

    wandb.init(
        project=args.wandb_project,
        # entity="jiminit27",
        name=args.wandb_run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model": "ResNetTransformer"
        }
    )

    X_train, y_train, X_test, y_test = load_uci_har_data(args.data_dir)

    train_dataset = UCIHAR_Dataset(X_train, y_train)
    test_dataset = UCIHAR_Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device(args.device)

    model = ResNetTransformer(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = args.best_acc
    for epoch in range(args.epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_dir + "/best_model.pth")
            print(f"Best model save : Epoch :  {epoch+1}  / Best acc: {best_acc:.4f}")

