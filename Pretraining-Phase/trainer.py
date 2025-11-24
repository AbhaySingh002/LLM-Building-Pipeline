from config import load_config
from utils import create_Dataloader, DatasetLoad
from gptv1.GPT import TinyGPT

from tqdm.auto import tqdm
import time
import torch
from torch.optim import AdamW
from torch import autocast, GradScaler
from pathlib import Path

cfg = load_config()

DATA_PATH = cfg.data.path
CONTEXT_LENGTH = cfg.data.context_length
BATCH_SIZE = cfg.data.batch_size
DEVICE = cfg.data.device if torch.cuda.is_available() else "cpu"

PRECISION = cfg.training.precision
MAX_ITERS = cfg.training.max_iters
LEARNING_RATE = cfg.training.learning_rate
GRAD_CLIP = cfg.training.grad_clip
EVAL_ITERS = cfg.training.eval_iters
WARMUP_ITERS = cfg.training.warmup_iters
MIN_LR = cfg.training.min_lr
LOG_INTERVAL = cfg.training.log_interval
DROP = cfg.model.dropout


N_BLOCK = cfg.model.n_block
N_HEAD = cfg.model.n_head
D_MODEL = cfg.model.d_model

CKPT_DIR = Path(cfg.save.ckpt_dir)
CKPT_DIR.mkdir(exist_ok=True)


def get_lr(it: int):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(decay_ratio)))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_iters=EVAL_ITERS):
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=getattr(torch, PRECISION)):
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses

def train():
    print(f"Device: {DEVICE} | Precision: {PRECISION}")
    train_loader, val_loader, dataset = create_Dataloader(DATA_PATH, BATCH_SIZE, CONTEXT_LENGTH, train=0.9)

    vocab_size = dataset.tokenizer.vocab_size
    model = TinyGPT(vocab_size=vocab_size,context_length=CONTEXT_LENGTH, n_block=N_BLOCK, n_head=N_HEAD, d_model=D_MODEL, dropout=DROP).to(DEVICE)

    if hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=cfg.training.weight_decay)
    scaler = GradScaler(enabled=(PRECISION == "float16"))

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("\nStarting Training...\n")

    iter_num = 0
    best_val_loss = float("inf")
    start_time = time.time()

    data_iters = iter(train_loader)

    pbar = tqdm(range(MAX_ITERS), desc="Training")
    for iter_num in pbar:

        lr = get_lr(iter_num)
        optimizer.param_groups[0]["lr"] = lr

        try:
            data = next(data_iters)
        except StopIteration:
            data_iters = iter(train_loader)
            data = next(data_iters)
        
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with autocast(device_type="cuda", dtype=getattr(torch, PRECISION)):
            _, loss = model(x, y)

        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % LOG_INTERVAL == 0:
            tokens_per_sec = (BATCH_SIZE * CONTEXT_LENGTH * LOG_INTERVAL) / (time.time() - start_time + 1e-8)
            start_time = time.time()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}"
            })

        if iter_num % EVAL_ITERS == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, train_loader, val_loader, DEVICE)
            print(f"\nStep {iter_num:,} | "
                  f"Train: {losses['train']:.4f} | "
                  f"Val: {losses['val']:.4f} | "
                  f"LR: {lr:.2e}")

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "iter": iter_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": best_val_loss,
                    "config": {
                        "vocab_size": vocab_size,
                        "context_length": CONTEXT_LENGTH,
                        "n_block": N_BLOCK,
                        "n_head": N_HEAD,
                        "d_model": D_MODEL,
                    }
                }, CKPT_DIR / "best_model.pt")
                print("New best model saved!")

        
        if iter_num % (MAX_ITERS // 10) == 0 and iter_num > 0:
            torch.save(model.state_dict(), CKPT_DIR / f"checkpoint_iter{iter_num}.pt")

    print(f"\nTraining finished! Best validation loss: {best_val_loss:.4f}")