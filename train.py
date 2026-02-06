import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

from models.temporal_unet import TemporalUNet
from models.scheduler import CosineNoiseSchedule
from models.gaussian_diffusion import GaussianDiffusion
from utils.trajectory import TrajectoryDataset

def load_checkpoint(checkpoint_path, model, optimizer, device):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]  # resume from the epoch after this one
    losses = checkpoint["losses"]

    print(f"Resumed from checkpoint: epoch {start_epoch}, loss {losses[-1]:.4f}")
    return start_epoch, losses


def train(model, diffusion, dataset, num_epochs=100, batch_size=32, lr=2e-4, gradient_clip=1.0, 
          checkpoint_dir="checkpoints", checkpoint_freq=10, resume_checkpoint=None, 
          device=torch.device("cuda"), dim=32, dim_mults=[1, 2, 4]
    ):
    """
    - In:
        - model: TemporalUNet model
        - diffusion: GaussianDiffusion wrapper
        - dataset: TrajectoryDataset
        - num_epochs: total number of training epochs
        - batch_size: batch size
        - lr: learning rate
        - gradient_clip: gradient clipping value
        - checkpoint_dir: directory for saving checkpoints
        - checkpoint_freq: save checkpoint every N epochs
        - resume_checkpoint: path to checkpoint to resume from (None to start fresh)
        - device: device to train on
    - Out:
        - losses: optional list of training losses
    """
    model = model.to(device)
    diffusion = diffusion.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    start_epoch = 0
    losses = []

    if resume_checkpoint is not None:
        start_epoch, losses = load_checkpoint(resume_checkpoint, model, optimizer, device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training on device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {start_epoch + 1} -> {num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(device)    # (32, 4, 128)
            
            loss = diffusion.loss(batch)
            optimizer.zero_grad()
            loss.backward()

            if gradient_clip > 0:   # clip to avoid weights blowing up
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "normalizer_mean": dataset.normalizer.mean,
                "normalizer_std": dataset.normalizer.std,
                "dim": dim,
                "dim_mults": dim_mults,
            }
            torch.save(checkpoint, checkpoint_path / f"checkpoint_{epoch + 1}.pt")
            print(f"Saved checkpoint to {checkpoint_path / f'checkpoint_{epoch + 1}.pt'}")

    # Save final model
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalizer_mean": dataset.normalizer.mean,
        "normalizer_std": dataset.normalizer.std,
        "losses": losses,
        "dim": dim,
        "dim_mults": dim_mults,
    }
    torch.save(final_checkpoint, checkpoint_path / "final_model.pt")
    print(f"Saved final model to {checkpoint_path / 'final_model.pt'}")

    return losses


def main():
    parser = argparse.ArgumentParser(description="Train Diffuser model")
    parser.add_argument("--data", type=str, default="dataset/umaze_5000.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=32, help="Base model dimension")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (e.g. checkpoints/checkpoint_10.pt)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading dataset from {args.data}...")
    dataset = TrajectoryDataset.load(args.data)
    print(f"Loaded {len(dataset)} trajectories")

    # Get dimensions from data
    sample = dataset[0]
    transition_dim, horizon = sample.shape
    print(f"Trajectory shape: ({transition_dim}, {horizon})")

    # Model instantiation
    # U-net
    model = TemporalUNet(transition_dim=transition_dim, dim=args.dim, dim_mults=[1, 2, 4])
    num_params = sum(p.numel() for p in model.parameters())
    # Noise schedule
    schedule = CosineNoiseSchedule(num_timesteps=args.diffusion_steps)
    # Complete diffusion process
    diffusion = GaussianDiffusion(model=model, schedule=schedule, state_dim=2, action_dim=2)
    print(f"Model created with {num_params:,} parameters")

    losses = train(
        model=model,
        diffusion=diffusion,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        resume_checkpoint=args.resume,
        device=device,
        dim=args.dim,
        dim_mults=[1, 2, 4]
    )

    # Train loss plotting
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.savefig(Path(args.checkpoint_dir) / "loss_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved loss results to {args.checkpoint_dir}/loss_curve.png")


if __name__ == "__main__":
    main()
