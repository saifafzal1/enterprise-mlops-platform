import os, mlflow, torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR   = "data/processed/cats_dogs"
MODEL_PATH = "models/cats_vs_dogs.pt"
EPOCHS     = int(os.getenv("EPOCHS", "5"))
BATCH      = 32
LR         = 1e-4
CLASSES    = ["cats", "dogs"]

TRANSFORM = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def build_model():
    m = models.efficientnet_b0(weights="IMAGENET1K_V1")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def get_loaders():
    loaders = {}
    for split in ("train", "val"):
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=TRANSFORM[split])
        loaders[split] = DataLoader(ds, batch_size=BATCH, shuffle=(split == "train"), num_workers=2)
    return loaders


def train():
    mlflow.set_experiment("cats-vs-dogs")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    loaders   = get_loaders()

    with mlflow.start_run():
        mlflow.log_params({"epochs": EPOCHS, "lr": LR, "batch_size": BATCH, "model": "EfficientNetB0"})

        for epoch in range(EPOCHS):
            model.train()
            for imgs, labels in loaders["train"]:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                criterion(model(imgs), labels).backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in loaders["val"]:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += (model(imgs).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            acc = correct / total
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            print(f"Epoch {epoch+1}/{EPOCHS}  val_accuracy={acc:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print(f"Saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
