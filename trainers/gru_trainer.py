# trainers/gru_trainer.py
import os
import torch
from tqdm import tqdm

class GRUTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, save_path):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_loss = float('inf')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        with tqdm(self.train_loader, desc="Training", leave=False) as bar:
            for inputs, targets in bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(self.model.state_dict(), self.save_path)

        return avg_loss

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")