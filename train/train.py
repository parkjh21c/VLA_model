from pathlib import Path
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data.dataset import VLADataset
from data.transforms import transform
from models.policy import Policy
from train.loss import criterion

try:
    import wandb
except ImportError:
    wandb = None


def collate_fn(batch):
    images, texts, actions = zip(*batch)
    images = torch.stack(images)
    actions = torch.stack(actions)
    return images, list(texts), actions


def build_dataloaders(
    dataset_root="external/boxbrown-mydataset",
    batch_size=8,
    train_ratio=0.9,
):
    dataset = VLADataset(dataset_root=dataset_root, transform=transform)

    episode_ids = sorted(dataset.frames["episode_index"].unique().tolist())
    generator = torch.Generator().manual_seed(42)
    shuffled_order = torch.randperm(len(episode_ids), generator=generator).tolist()
    shuffled_episode_ids = [episode_ids[index] for index in shuffled_order]

    train_episode_count = int(len(shuffled_episode_ids) * train_ratio)
    train_episode_count = min(max(train_episode_count, 1), len(shuffled_episode_ids) - 1)

    train_episode_ids = set(shuffled_episode_ids[:train_episode_count])
    val_episode_ids = set(shuffled_episode_ids[train_episode_count:])

    train_indices = dataset.frames.index[
        dataset.frames["episode_index"].isin(train_episode_ids)
    ].tolist()
    val_indices = dataset.frames.index[
        dataset.frames["episode_index"].isin(val_episode_ids)
    ].tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(
        f"Episode split: train={len(train_episode_ids)} episodes, "
        f"val={len(val_episode_ids)} episodes"
    )
    print(
        f"Frame split: train={len(train_indices)} frames, "
        f"val={len(val_indices)} frames"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def run_epoch(model, dataloader, optimizer, device, training=True, max_steps=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for step, (images, texts, actions) in enumerate(dataloader):
        images = images.to(device)
        actions = actions.to(device)

        with torch.set_grad_enabled(training):
            predictions = model(images, texts)
            loss = criterion(predictions, actions)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if max_steps is not None and step + 1 >= max_steps:
            break

    denominator = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)
    return total_loss / max(denominator, 1)


def train(
    dataset_root="external/boxbrown-mydataset",
    epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    save_dir="checkpoints/exp_pretrained_distilbert",
    cache_root=".cache",
    max_train_steps=None,
    max_val_steps=None,
    use_wandb=False,
    wandb_project="vla",
    wandb_run_name=None,
    wandb_mode="online",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_path = Path(cache_root)
    (cache_path / "torch").mkdir(parents=True, exist_ok=True)
    (cache_path / "huggingface").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
    )

    model = Policy(cache_root=cache_root).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    run = None
    if use_wandb and wandb is not None:
        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            dir=str(Path.cwd()),
            mode=wandb_mode,
            config={
                "dataset_root": dataset_root,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "save_dir": save_dir,
                "cache_root": cache_root,
                "max_train_steps": max_train_steps,
                "max_val_steps": max_val_steps,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
            },
            tags=["prototype", "episode-split", "distilbert"],
        )

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
            max_steps=max_train_steps,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            training=False,
            max_steps=max_val_steps,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        checkpoint_path = save_path / f"policy_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        epoch_time_sec = time.time() - epoch_start
        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time_sec": epoch_time_sec,
                }
            )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    train()
