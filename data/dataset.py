import io
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class VLADataset(Dataset):
    def __init__(
        self,
        dataset_root="external/boxbrown-mydataset",
        camera_name="camera1",
        transform=None,
        excluded_episode_ids=(0, 271),
        default_prompt="stack the colored blocks",
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.camera_name = camera_name
        self.transform = transform
        self.excluded_episode_ids = set(excluded_episode_ids)
        self.default_prompt = default_prompt

        self.info = self._load_info()
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self.frames = self._load_frames()

    def _load_info(self):
        info_path = self.dataset_root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _load_tasks(self):
        task_path = self.dataset_root / "meta" / "tasks.parquet"
        task_df = pd.read_parquet(task_path)

        if "task_index" in task_df.columns and task_df.index.name is None:
            # In this dataset the task sentence is stored in the index.
            return {int(row["task_index"]): str(index) for index, row in task_df.iterrows()}

        index_column = self._pick_first_existing_column(
            task_df.columns,
            ["task_index", "index", "id"],
        )
        text_column = self._pick_first_existing_column(
            task_df.columns,
            ["task", "task_name", "text", "instruction"],
        )

        if index_column is None or text_column is None:
            return {}

        return {
            int(row[index_column]): str(row[text_column])
            for _, row in task_df.iterrows()
        }

    def _load_episodes(self):
        episode_dir = self.dataset_root / "meta" / "episodes"
        episode_files = sorted(episode_dir.glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode parquet files found under {episode_dir}")

        episode_df = pd.concat(
            [pd.read_parquet(path) for path in episode_files],
            ignore_index=True,
        )
        episode_df = episode_df[~episode_df["episode_index"].isin(self.excluded_episode_ids)].copy()
        episode_df["video_path"] = episode_df.apply(self._build_video_path, axis=1)

        return episode_df.set_index("episode_index")

    def _load_frames(self):
        data_dir = self.dataset_root / "data"
        data_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No frame parquet files found under {data_dir}")

        frame_df = pd.concat(
            [pd.read_parquet(path) for path in data_files],
            ignore_index=True,
        )
        frame_df = frame_df[~frame_df["episode_index"].isin(self.excluded_episode_ids)].copy()
        frame_df["task_text"] = frame_df["task_index"].map(
            lambda task_index: self.tasks.get(int(task_index), self.default_prompt)
        )
        frame_df["video_path"] = frame_df["episode_index"].map(
            lambda episode_index: self.episodes.loc[int(episode_index), "video_path"]
        )

        if frame_df.empty:
            raise ValueError("No valid frames found after applying episode filters.")

        return frame_df.reset_index(drop=True)

    def _build_video_path(self, row):
        chunk_column = f"videos/observation.images.{self.camera_name}/chunk_index"
        file_column = f"videos/observation.images.{self.camera_name}/file_index"

        if chunk_column not in row or file_column not in row:
            raise KeyError(
                f"Camera '{self.camera_name}' was not found in episode metadata. "
                f"Expected columns '{chunk_column}' and '{file_column}'."
            )

        chunk_index = int(row[chunk_column])
        file_index = int(row[file_column])
        return (
            self.dataset_root
            / "videos"
            / f"observation.images.{self.camera_name}"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )

    def _pick_first_existing_column(self, columns, candidates):
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    def _extract_frame_image(self, video_path, timestamp):
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{float(timestamp):.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]

        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        return image

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        row = self.frames.iloc[idx]

        image = self._extract_frame_image(row["video_path"], row["timestamp"])
        if self.transform is not None:
            image = self.transform(image)

        text = row["task_text"]
        action = torch.tensor(row["action"], dtype=torch.float32)

        return image, text, action
