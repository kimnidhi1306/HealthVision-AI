# model_training.py

from torchinfo import summary
import torch.optim as optim
from tqdm.auto import tqdm
import os
import warnings
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_encoder_params(model):
    for param in model.encoder.parameters():
        param.requires_grad = False

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):

    model.train()
    train_loss = 0.
    train_iou = 0.

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device=DEVICE, dtype=torch.float32)
        y = y.to(device=DEVICE, dtype=torch.float32)
        optimizer.zero_grad()
        logit_mask = model(X)
        loss = loss_fn(logit_mask, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        prob_mask = logit_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(output=pred_mask.detach().cpu().long(),
                                               target=y.cpu().long(),
                                               mode="binary")

        train_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").numpy()

    train_loss = train_loss / len(dataloader)
    train_iou = train_iou / len(dataloader)

    return train_loss, train_iou

def val_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module):

    model.eval()
    val_loss = 0.
    val_iou = 0.

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device=DEVICE, dtype=torch.float32)
            y = y.to(device=DEVICE, dtype=torch.float32)
            logit_mask = model(X)
            loss = loss_fn(logit_mask, y)
            val_loss += loss.item()

            prob_mask = logit_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            tp, fp, fn, tn = smp.metrics.get_stats(output=pred_mask.detach().cpu().long(),
                                                  target=y.cpu().long(),
                                                  mode="binary")

            val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").numpy()

    val_loss = val_loss / len(dataloader)
    val_iou = val_iou / len(dataloader)

    return val_loss, val_iou

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0001, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)

        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer, early_stopping, epochs: int = 10):

    results = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_iou = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        val_loss, val_iou = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn)

        print(f'Epoch: {epoch + 1} | ',
              f'Train Loss: {train_loss:.4f} | ',
              f'Train IOU: {train_iou:.4f} | ',
              f'Val Loss: {val_loss:.4f} | ',
              f'Val IOU: {val_iou:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break

        results['train_loss'].append(train_loss)
        results['train_iou'].append(train_iou)
        results['val_loss'].append(val_loss)
        results['val_iou'].append(val_iou)

    return results
