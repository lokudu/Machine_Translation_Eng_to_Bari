# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import TRAINING_CONFIG

class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG.learning_rate, weight_decay=TRAINING_CONFIG.weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, loader, teacher_forcing=0.5):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            src = batch['src'].to(self.device)
            tgt_in = batch['tgt_in'].to(self.device)
            tgt_out = batch['tgt_out'].to(self.device)
            src_lens = batch['src_lens']
            if src.size(1) < 2:
                continue
            self.optimizer.zero_grad()
            outputs = self.model(src, tgt_in, src_lens, teacher_forcing)
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_out = tgt_out.reshape(-1)
            loss = self.criterion(outputs, tgt_out)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.gradient_clip)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                src = batch['src'].to(self.device)
                tgt_in = batch['tgt_in'].to(self.device)
                tgt_out = batch['tgt_out'].to(self.device)
                src_lens = batch['src_lens']
                if src.size(1) < 2:
                    continue
                outputs = self.model(src, tgt_in, src_lens, teacher_forcing_ratio=0)
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_out = tgt_out.reshape(-1)
                loss = self.criterion(outputs, tgt_out)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
        return total_loss / max(num_batches, 1)
    
    def train(self, train_loader, val_loader, epochs, teacher_forcing_start=0.5, teacher_forcing_end=0.3):
        for epoch in range(epochs):
            tf_ratio = max(teacher_forcing_end, teacher_forcing_start - (teacher_forcing_start - teacher_forcing_end) * epoch / epochs)
            train_loss = self.train_epoch(train_loader, tf_ratio)
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | TF: {tf_ratio:.2f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_model.pt')
                print(f"  ✓ Saved best model")
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}