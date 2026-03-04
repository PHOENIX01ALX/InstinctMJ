"""Visualize AMASS motion data by playing retargetted motions on the robot (mjlab).

NOTE: This script is a *data visualization* utility.  It loads retargetted
motion files (or runs SMPL IK on raw AMASS data) and drives the robot joint
positions directly — no RL policy is involved.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import mjlab
from mjlab.utils.lab_api import math as math_utils
from instinct_mjlab.assets.unitree_g1 import G1_MJCF_PATH
from instinct_mjlab.utils.humanoid_ik import HumanoidSmplRotationalIK

# --------------------------------------------------------------------------- #
# Constants (same as InstinctLab original)
# --------------------------------------------------------------------------- #
DECIMATION = 4


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VisualizeConfig:
    """Configuration for AMASS motion visualization."""

    # Path to AMASS dataset root or a single npz file
    motion_path: str = os.path.expanduser("~/Datasets/AMASS/")
    # Optional YAML with selected motion files
    selection_yaml: str | None = None
    # Number of environments
    num_envs: int = 4
    # Viewer backend
    viewer: Literal["native", "viser", "none"] = "native"
    # Video output path (set to enable recording)
    video: str | None = None
    # Video FPS (derived from sim_dt if not set)
    video_fps: float = 50.0
    # Device
    device: str = "cuda:0"
    # Debug mode (attach debugger)
    debug: bool = False
    # Live plot mode
    live_plot: bool = False
    # URDF/MJCF path for the robot chain
    urdf_path: str = G1_MJCF_PATH


def _load_motion_files(cfg: VisualizeConfig) -> list[str]:
    """Collect motion file paths from the configured motion_path."""
    import yaml

    root = os.path.abspath(cfg.motion_path)

    if cfg.selection_yaml is not None:
        with open(cfg.selection_yaml) as f:
            selection = yaml.safe_load(f)
        return [os.path.join(root, f) for f in selection.get("selected_files", [])]

    if os.path.isfile(root):
        return [root]

    motion_files: list[str] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.endswith("poses.npz") or filename.endswith("retargetted.npz"):
                motion_files.append(os.path.join(dirpath, filename))
    return sorted(motion_files)


def _load_single_motion(filepath: str, cfg: VisualizeConfig):
    """Load and retarget a single motion file, returning (root_pos, root_quat, joint_pos, framerate)."""
    import pytorch_kinematics as pk

    with open(cfg.urdf_path, mode="rb") as f:
        urdf_str = f.read()
    robot_chain = pk.build_chain_from_urdf(urdf_str)

    if filepath.endswith("poses.npz"):
        motion = np.load(filepath)
        framerate = float(motion["mocap_framerate"])
        poses = torch.from_numpy(motion["poses"]).to(torch.float32)
        poses = poses[:, : 24 * 3].reshape(-1, 24, 3)
        root_trans = torch.from_numpy(motion["trans"]).to(torch.float32)

        retargetting_func = HumanoidSmplRotationalIK(
            robot_chain=robot_chain,
            smpl_root_in_robot_link_name="pelvis",
            translation_scaling=0.75,
            translation_height_offset=0.0,
        )
        robot_joint_pos, robot_root_poses = retargetting_func(poses, root_trans)
        root_pos = robot_root_poses[:, :3]
        root_quat = robot_root_poses[:, 3:]
    elif filepath.endswith("retargetted.npz"):
        motion = np.load(filepath, allow_pickle=True)
        framerate = float(motion["framerate"])
        root_pos = torch.from_numpy(motion["base_pos_w"]).to(torch.float32)
        root_quat = torch.from_numpy(motion["base_quat_w"]).to(torch.float32)
        robot_joint_pos = torch.from_numpy(motion["joint_pos"]).to(torch.float32)

        # re-order joint positions to match the order in robot chain
        retargetted_joint_names = (
            motion["joint_names"] if isinstance(motion["joint_names"], list) else motion["joint_names"].tolist()
        )
        chain_joint_names = [j.name for j in robot_chain.get_joints()]
        robot_joint_pos = robot_joint_pos[:, [retargetted_joint_names.index(name) for name in chain_joint_names]]
    else:
        raise ValueError(f"Unsupported motion file format: {filepath}")

    return root_pos, root_quat, robot_joint_pos, framerate


def run_visualize(cfg: VisualizeConfig) -> None:
    """Run the AMASS motion visualization loop."""
    # wait for attach if in debug mode
    if cfg.debug:
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    motion_files = _load_motion_files(cfg)
    if not motion_files:
        print("[ERROR] No motion files found.")
        return

    print(f"[INFO] Found {len(motion_files)} motion files.")

    # Iterate through motions and visualize
    for motion_idx, motion_file in enumerate(motion_files):
        print(f"[INFO] Loading motion {motion_idx + 1}/{len(motion_files)}: {motion_file}")
        root_pos, root_quat, joint_pos, framerate = _load_single_motion(motion_file, cfg)

        num_frames = root_pos.shape[0]
        print(
            f"  Framerate: {framerate}, Frames: {num_frames}, "
            f"Duration: {num_frames / framerate:.2f}s"
        )

        # Print basic statistics
        root_height_min = root_pos[:, 2].min().item()
        root_height_max = root_pos[:, 2].max().item()
        root_height_mean = root_pos[:, 2].mean().item()
        print(
            f"  Root height: min={root_height_min:.3f}, max={root_height_max:.3f}, "
            f"mean={root_height_mean:.3f}"
        )

        # Compute root velocity
        if num_frames > 1:
            root_vel = (root_pos[1:] - root_pos[:-1]) * framerate
            max_linvel = torch.max(torch.norm(root_vel, dim=-1)).item()
            print(f"  Max linear velocity: {max_linvel:.3f} m/s")

        # Compute joint range
        joint_range = joint_pos.max(dim=0).values - joint_pos.min(dim=0).values
        print(f"  Joint range (mean): {joint_range.mean().item():.3f} rad")

    print(f"\n[INFO] Finished processing {len(motion_files)} motion files.")


def main() -> None:
    """Entry point for AMASS visualization."""
    import tyro

    cfg = tyro.cli(VisualizeConfig, config=mjlab.TYRO_FLAGS)
    run_visualize(cfg)


if __name__ == "__main__":
    main()
