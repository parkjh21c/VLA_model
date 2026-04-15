import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from data.transforms import transform
from models.fusion import Fusion
from models.language_encoder import LanguageEncoder
from models.vision_encoder import VisionEncoder

# 10 tasks in LIBERO
LIBERO_10_TASKS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}


class StateEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state):
        return self.net(state)


class LiberoPolicy(nn.Module):
    def __init__(
        self,
        action_dim=7,
        state_dim=8,
        state_feature_dim=512,
        cache_root=".cache",
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(cache_dir=f"{cache_root}/torch")
        self.language_encoder = LanguageEncoder(cache_dir=f"{cache_root}/huggingface")
        self.state_encoder = StateEncoder(
            input_dim=state_dim,
            output_dim=state_feature_dim,
        )
        self.fusion = Fusion()

        fused_dim = 768 + 768 + 768 + state_feature_dim
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, image1, image2, text, state):
        vision_feat1 = self.vision_encoder(image1)
        vision_feat2 = self.vision_encoder(image2)
        lang_feat = self.language_encoder(text)
        state_feat = self.state_encoder(state)
        fused_feat = self.fusion(vision_feat1, vision_feat2, lang_feat, state_feat)
        return self.action_head(fused_feat)


class LiberoDataset(Dataset):
    def __init__(
        self,
        dataset_id="HuggingFaceVLA/libero",
        split="train",
        image_key1="observation.images.image",
        image_key2="observation.images.image2",
        state_key="observation.state",
        action_key="action",
        task_key="task_index",
        task_map=None,
        transform_fn=None,
        prompt_template="task {task_index}",
    ):
        super().__init__()

        self.dataset = load_dataset(dataset_id, split=split)
        self.image_key1 = image_key1
        self.image_key2 = image_key2
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.task_map = task_map or {}
        self.transform = transform_fn
        self.prompt_template = prompt_template

        missing_keys = [
            key
            for key in [image_key1, image_key2, state_key, action_key, task_key]
            if key not in self.dataset.column_names
        ]
        if missing_keys:
            raise KeyError(
                f"Missing dataset columns: {missing_keys}. "
                f"Available columns: {self.dataset.column_names}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image1 = sample[self.image_key1]
        image2 = sample[self.image_key2]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        task_index = int(sample[self.task_key])
        text = self.task_map.get(
            task_index,
            self.prompt_template.format(task_index=task_index),
        )
        state = torch.tensor(sample[self.state_key], dtype=torch.float32)
        action = torch.tensor(sample[self.action_key], dtype=torch.float32)

        return image1, image2, text, state, action


def collate_fn(batch):
    image1, image2, text, state, action = zip(*batch)
    return (
        torch.stack(image1),
        torch.stack(image2),
        list(text),
        torch.stack(state),
        torch.stack(action),
    )


def split_by_episode(dataset, train_ratio=0.9, seed=42):
    if "episode_index" not in dataset.dataset.column_names:
        raise KeyError("LIBERO dataset must contain 'episode_index' for episode split.")

    episode_ids = sorted(set(dataset.dataset["episode_index"]))
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(episode_ids), generator=generator).tolist()
    shuffled_episode_ids = [episode_ids[i] for i in shuffled]

    train_episode_count = int(len(shuffled_episode_ids) * train_ratio)
    train_episode_count = min(max(train_episode_count, 1), len(shuffled_episode_ids) - 1)

    train_episode_ids = set(shuffled_episode_ids[:train_episode_count])
    val_episode_ids = set(shuffled_episode_ids[train_episode_count:])

    train_indices = [
        idx
        for idx, episode_index in enumerate(dataset.dataset["episode_index"])
        if episode_index in train_episode_ids
    ]
    val_indices = [
        idx
        for idx, episode_index in enumerate(dataset.dataset["episode_index"])
        if episode_index in val_episode_ids
    ]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        len(train_episode_ids),
        len(val_episode_ids),
    )


def build_dataloaders(args):
    dataset = LiberoDataset(
        dataset_id=args.dataset_id,
        split=args.split,
        image_key1=args.image_key1,
        image_key2=args.image_key2,
        state_key=args.state_key,
        action_key=args.action_key,
        task_key=args.task_key,
        task_map=LIBERO_10_TASKS,
        transform_fn=transform,
        prompt_template=args.prompt_template,
    )

    train_dataset, val_dataset, train_episodes, val_episodes = split_by_episode(
        dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(
        f"Episode split: train={train_episodes} episodes, "
        f"val={val_episodes} episodes"
    )
    print(
        f"Frame split: train={len(train_dataset)} frames, "
        f"val={len(val_dataset)} frames"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def load_pretrained_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    filtered_state = {
        key: value
        for key, value in checkpoint.items()
        if key.startswith("vision_encoder.") or key.startswith("language_encoder.")
    }

    load_result = model.load_state_dict(filtered_state, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded keys: {len(filtered_state)}")
    print(f"Missing keys: {load_result.missing_keys[:20]}")
    print(f"Unexpected keys: {load_result.unexpected_keys[:20]}")


def set_trainable_params(model, freeze_backbone=True):
    if not freeze_backbone:
        return

    for parameter in model.vision_encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.language_encoder.parameters():
        parameter.requires_grad = False


def run_epoch(model, dataloader, optimizer, criterion, device, training=True, max_steps=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for step, (image1, image2, text, state, action) in enumerate(dataloader):
        image1 = image1.to(device)
        image2 = image2.to(device)
        state = state.to(device)
        action = action.to(device)

        with torch.set_grad_enabled(training):
            prediction = model(image1, image2, text, state)
            loss = criterion(prediction, action)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if max_steps is not None and step + 1 >= max_steps:
            break

    denominator = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)
    return total_loss / max(denominator, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VLA checkpoint on LIBERO.")
    parser.add_argument(
        "--dataset-id",
        default="HuggingFaceVLA/libero",
        help="Hugging Face dataset id for LIBERO-style data.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-key1", default="observation.images.image")
    parser.add_argument("--image-key2", default="observation.images.image2")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--task-key", default="task_index")
    parser.add_argument("--prompt-template", default="task {task_index}")
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/exp_pretrained_distilbert/policy_epoch_5.pt",
    )
    parser.add_argument("--cache-root", default=".cache")
    parser.add_argument("--save-dir", default="checkpoints/libero_finetune")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_path = Path(args.cache_root)
    (cache_path / "torch").mkdir(parents=True, exist_ok=True)
    (cache_path / "huggingface").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)

    model = LiberoPolicy(cache_root=args.cache_root).to(device)
    load_pretrained_backbone(model, args.checkpoint_path)
    set_trainable_params(model, freeze_backbone=args.freeze_backbone)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            training=True,
            max_steps=args.max_train_steps,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            training=False,
            max_steps=args.max_val_steps,
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"time_sec={time.time() - epoch_start:.2f}"
        )

        checkpoint_path = save_dir / f"libero_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
