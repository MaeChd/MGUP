import os
import math
import torch
import wandb
import random
import numpy as np
from loguru import logger
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from torch_optimizer import AdamW,Lion
from MGUP import MGUP_AdamW as MGAdamW
from MGUP import MGUP_Lion as MGLion
from MGUP import MGUP_Muon as MGMuon
from c_lion import Lion as CLion
from c_adamw import AdamW as CAdamW
from c_muon import Muon as CMuon
from muon import Muon as Muon

class WikiTextDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=1024):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data

def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "mg_adamw":
        return MGAdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "lion":
        return Lion(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.95, 0.98)
        )
    elif optimizer_name == "mg_lion":
        return MGLion(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.95, 0.98)
        )
    elif optimizer_name == "clion":
        return CLion(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.95, 0.98)  
        )
    elif optimizer_name == "cadamw":
        return CAdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "cmuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return CMuon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

    
    elif optimizer_name == "mg_muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return MGMuon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"

def get_model_and_dataloader(config_path, dataset_name, batch_size):
    # Load model config
    config = Qwen2Config.from_json_file(config_path)
    model = Qwen2ForCausalLM(config)
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "./model/Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    
    # Load dataset
    train_dataset = load_dataset("./wikitext_local.py", "wikitext-103-v1" ,\
        trust_remote_code=True,local_dir="./data/wikitext/", split="train")
    val_dataset = load_dataset("./wikitext_local.py", "wikitext-103-v1",\
        trust_remote_code=True,local_dir="./data/wikitext/", split="validation")
    train_dataset = {"train": train_dataset}
    val_dataset = {"train": val_dataset}
    
    train_dataset = WikiTextDataset(f"{dataset_name}_train", train_dataset, tokenizer)
    val_dataset = WikiTextDataset(f"{dataset_name}_val", val_dataset, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("train_loader:",len(train_loader))
    return model, train_loader, val_loader

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps
    model.train()
    return avg_loss

def train(args):
    logger.add(f"logs/train_wikitext_{args.optimizer}_lr{args.lr}.log")
    
    # Initialize wandb
    run_name = f"wikitext_{args.optimizer}_lr{args.lr}_wd{args.wd}_bs{args.batch_size}_ga{args.gradient_accumulation_steps}"
    wandb.init(
        project="qwen-wikitext-training",
        name=run_name,
        config=vars(args),
        reinit = True
    )
    
    # Setup model and dataloader
    model, train_loader, val_loader = get_model_and_dataloader(
        args.config_path,
        "wikitext-103",
        args.batch_size
    )
    
    # Setup optimizer
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.wd)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Setup learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Autocast for mixed precision training
            with autocast():
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # Scale loss and call backward
            scaler.scale(loss).backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Unscale gradients and optimizer step
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                current_step = step // args.gradient_accumulation_steps
                
                # Calculate average training loss
                avg_train_loss = total_loss / args.gradient_accumulation_steps
                
                # Evaluate on validation set every eval_steps
                if current_step > 0 and current_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    
                    # Log metrics
                    wandb.log({
                        "epoch": epoch,
                        "step": current_step,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "training_loss": avg_train_loss,
                        "validation_loss": val_loss
                    })
                    
                    logger.info(
                        f"Epoch: {epoch} Step: {current_step} "
                        f"LR: {optimizer.param_groups[0]['lr']} "
                        f"Training loss: {avg_train_loss:.4f} "
                        f"Validation loss: {val_loss:.4f}"
                    )
                else:
                    # Log only training metrics
                    wandb.log({
                        "epoch": epoch,
                        "step": current_step,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "training_loss": avg_train_loss
                    })
                    
                    logger.info(
                        f"Epoch: {epoch} Step: {current_step} "
                        f"LR: {optimizer.param_groups[0]['lr']} "
                        f"Training loss: {avg_train_loss:.4f}"
                    )
                
                total_loss = 0
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/qwen_150m.json")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    
    args = parser.parse_args()
    
    # List of optimizers to compare
    optimizers = ["adamw", "mgup_adamw", "cadamw", "muon", "c_muon", "mgup_muon"]
    
    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name} optimizer...")
        args.optimizer = optimizer_name
        train(args)
