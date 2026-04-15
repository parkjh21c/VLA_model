import argparse
import json
import os
from pathlib import Path
import sys

import imageio
import numpy as np
import torch
from PIL import Image

from data.transforms import transform
from generic_finetune import VLAFineTunePolicy


def ensure_libero_on_path():
    script_root = Path(__file__).resolve().parent
    candidate_paths = [
        script_root / "LIBERO",
        script_root.parent / "LIBERO",
    ]

    for candidate in candidate_paths:
        if (candidate / "libero" / "libero" / "__init__.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
ensure_libero_on_path()


def quat_xyzw_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    quat = quat / norm
    x, y, z, w = quat
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))

    if sin_half < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = np.array([x, y, z], dtype=np.float32) / sin_half
    return axis * angle


def build_state(obs):
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    eef_axis_angle = quat_xyzw_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, eef_axis_angle, gripper], axis=0).astype(np.float32)


def preprocess_image(image_array, flip_images=True):
    if flip_images:
        image_array = image_array[::-1]
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    return transform(pil_image)


def format_frame(image_array, flip_images=True):
    if flip_images:
        return image_array[::-1].astype(np.uint8)
    return image_array.astype(np.uint8)


def save_episode_video(frames, video_path, fps=20):
    if not frames:
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def load_model(args, device):
    model = VLAFineTunePolicy(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        state_feature_dim=args.state_feature_dim,
        cache_root=args.cache_root,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def get_suite_and_tasks(args):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite_name]()

    if args.task_ids:
        task_ids = [int(task_id) for task_id in args.task_ids.split(",")]
    else:
        task_ids = list(range(task_suite.n_tasks))

    return task_suite, task_ids


def load_task_init_states(task_suite, task_id):
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_states_path, map_location="cpu", weights_only=False)


def rollout_one_episode(
    model,
    env,
    init_state,
    instruction,
    device,
    max_steps,
    flip_images,
    record_video=False,
    video_path=None,
    video_camera="agentview_image",
    video_fps=20,
):
    env.reset()
    obs = env.set_init_state(init_state)

    dummy_action = np.zeros(7, dtype=np.float32)
    frames = []
    for _ in range(5):
        obs, _, _, _ = env.step(dummy_action)
        if record_video:
            frames.append(format_frame(obs[video_camera], flip_images=flip_images))

    success = False
    for _ in range(max_steps):
        image1 = preprocess_image(obs["agentview_image"], flip_images=flip_images).unsqueeze(0).to(device)
        image2 = preprocess_image(obs["robot0_eye_in_hand_image"], flip_images=flip_images).unsqueeze(0).to(device)
        state = torch.from_numpy(build_state(obs)).unsqueeze(0).to(device)
        text = [instruction]

        with torch.no_grad():
            action = model(image1, image2, text, state)[0].detach().cpu().numpy()

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, reward, done, info = env.step(action)
        if record_video:
            frames.append(format_frame(obs[video_camera], flip_images=flip_images))

        success = bool(done)
        if isinstance(info, dict) and "task_success" in info:
            success = bool(info["task_success"])

        if success:
            break

    if record_video and video_path is not None:
        save_episode_video(frames, video_path, fps=video_fps)

    return success


def evaluate_task(model, task_suite, task_id, args, device):
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    instruction = task.language
    env_args = {
        "bddl_file_name": str(task_suite.get_task_bddl_file_path(task_id)),
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    init_states = load_task_init_states(task_suite, task_id)
    successes = 0
    episode_results = []

    for episode_idx in range(args.n_episodes):
        init_state = init_states[episode_idx % len(init_states)]
        video_path = None
        if args.save_video:
            safe_task_name = task.name.replace("/", "_")
            video_path = (
                Path(args.video_dir)
                / args.suite_name
                / f"task_{task_id:02d}_{safe_task_name}"
                / f"episode_{episode_idx:02d}.mp4"
            )
        success = rollout_one_episode(
            model=model,
            env=env,
            init_state=init_state,
            instruction=instruction,
            device=device,
            max_steps=args.max_steps,
            flip_images=args.flip_images,
            record_video=args.save_video,
            video_path=video_path,
            video_camera=args.video_camera,
            video_fps=args.video_fps,
        )
        successes += int(success)
        episode_results.append(
            {
                "episode_index": episode_idx,
                "success": bool(success),
                "video_path": str(video_path) if video_path is not None else None,
            }
        )

    env.close()
    success_rate = successes / args.n_episodes
    return {
        "task_id": task_id,
        "task_name": task.name,
        "instruction": instruction,
        "successes": successes,
        "episodes": args.n_episodes,
        "success_rate": success_rate,
        "episode_results": episode_results,
    }


def evaluate_suite(model, task_suite, task_ids, args, device):
    results = []
    for task_id in task_ids:
        result = evaluate_task(model, task_suite, task_id, args, device)
        print(
            f"[task {result['task_id']}] "
            f"{result['task_name']} -> "
            f"success_rate={result['success_rate']:.3f}"
        )
        results.append(result)

    average = float(np.mean([result["success_rate"] for result in results])) if results else 0.0
    return {
        "suite_name": args.suite_name,
        "n_episodes": args.n_episodes,
        "average_success_rate": average,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned checkpoint on LIBERO.")
    parser.add_argument(
        "--checkpoint-path",
        default="../checkpoints/libero_finetune/libero_epoch_5.pt",
    )
    parser.add_argument("--suite-name", default="libero_10")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids, e.g. 0,1,2")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--cache-root", default=".cache")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--state-feature-dim", type=int, default=512)
    parser.add_argument("--flip-images", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="../eval_logs/videos")
    parser.add_argument(
        "--video-camera",
        default="agentview_image",
        choices=["agentview_image", "robot0_eye_in_hand_image"],
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--output-json", default="../eval_logs/libero_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args, device)
    task_suite, task_ids = get_suite_and_tasks(args)
    summary = evaluate_suite(model, task_suite, task_ids, args, device)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Suite: {summary['suite_name']}")
    print(f"Average success rate: {summary['average_success_rate']:.3f}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
