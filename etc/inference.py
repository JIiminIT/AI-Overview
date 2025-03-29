import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from data_loader import load_uci_har_data, UCIHAR_Dataset
# from model import ResNetTransformer
from positional_custom_model import ResNetTransformer


def run_inference(args):
    full_model_path = os.path.join(args.save_dir, args.model_path)
    X_train, y_train, X_test, y_test = load_uci_har_data(args.data_dir)
    test_dataset = UCIHAR_Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device(args.device)

    model = ResNetTransformer(num_classes=args.num_classes)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, target_names=args.class_labels))

    cm = confusion_matrix(y_true, y_pred)

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=args.class_labels
        )
    })

    report = classification_report(y_true, y_pred, target_names=args.class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    wandb.log({
        "classification_report": wandb.Table(dataframe=report_df)
    })

    # png 저장
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=args.class_labels,
                yticklabels=args.class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix_img": wandb.Image("confusion_matrix.png")})
    plt.show()
