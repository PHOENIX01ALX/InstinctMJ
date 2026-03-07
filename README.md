# Instinct_MJ

[![mjlab](https://img.shields.io/badge/framework-mjlab-4C7AF2.svg)](https://github.com/mujocolab/mjlab)
[![MuJoCo](https://img.shields.io/badge/simulator-MuJoCo-silver.svg)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-orange.svg)](https://releases.ubuntu.com/)
[![instinct_rl](https://img.shields.io/badge/training-instinct__rl-brightgreen.svg)](https://github.com/project-instinct/instinct_rl)

## Overview

`Instinct_MJ` is a `mjlab`-native port of InstinctLab tasks for humanoid whole-body reinforcement learning on MuJoCo.
It keeps task semantics aligned with InstinctLab while expressing environments, managers, scenes, and task registration in native `mjlab` style so the package plugs directly into `instinct_rl`.

**Key Features:**

- `Behavior-preserving migration` Ports InstinctLab locomotion, shadowing, perceptive, and parkour tasks with matching task IDs and training workflow.
- `mjlab-native integration` Uses `mjlab` task registration, manager configs, scene wiring, and MuJoCo assets instead of compatibility shims.
- `instinct_rl workflow` Supports the same train / play / export loop used by the Project-Instinct ecosystem.
- `Installable task package` Registers tasks through the `mjlab.tasks` entry-point so environments are discoverable after editable install.

## Task Suite

Registered task IDs:

- `Instinct-Locomotion-Flat-G1-v0`
- `Instinct-Locomotion-Flat-G1-Play-v0`
- `Instinct-BeyondMimic-Plane-G1-v0`
- `Instinct-BeyondMimic-Plane-G1-Play-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-v0`
- `Instinct-Perceptive-Shadowing-G1-Play-v0`
- `Instinct-Perceptive-Vae-G1-v0`
- `Instinct-Perceptive-Vae-G1-Play-v0`
- `Instinct-Parkour-Target-Amp-G1-v0`
- `Instinct-Parkour-Target-Amp-G1-Play-v0`

Use the CLI to inspect the full list at any time:

```bash
instinct-list-envs
instinct-list-envs shadowing
```

## Installation

Clone the required repositories as sibling directories:

```bash
mkdir -p ~/Project-Instinct
cd ~/Project-Instinct

git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/project-instinct/instinct_rl.git
git clone https://github.com/cmjang/Instinct_MJ.git
```

Install with `uv`:

```bash
cd Instinct_MJ
uv sync
```

Or install editable packages with `pip`:

```bash
pip install -e ../mjlab
pip install -e ../instinct_rl
pip install -e .
```

## Quick Start

Train:

```bash
instinct-train Instinct-Locomotion-Flat-G1-v0
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

Play (`--load-run` is required):

```bash
instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

Export ONNX for parkour:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

Module form is also available when console scripts are not on `PATH`:

```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Locomotion-Flat-G1-v0
python -m instinct_mj.scripts.instinct_rl.play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
python -m instinct_mj.scripts.list_envs
```

## Repository Layout

- `src/instinct_mj/tasks` — task registration and family-specific configs.
- `src/instinct_mj/envs` — environment wrappers, manager extensions, and shared MDP terms.
- `src/instinct_mj/motion_reference` — motion data loaders, buffers, and reference managers.
- `src/instinct_mj/assets` — MuJoCo robot assets and resource files.
- `src/instinct_mj/scripts` — train, play, visualization, and data-processing entry points.

## Data and Outputs

- Override dataset root with `INSTINCT_DATASETS_ROOT` when needed.
- Training logs are written to `logs/instinct_rl/<experiment_name>/<timestamp_run>/`.
- Play videos are saved under `videos/play/` in the selected run directory.

## Relationship to InstinctLab

`Instinct_MJ` is the MuJoCo / `mjlab` counterpart to InstinctLab in the Project-Instinct ecosystem.

Reference links:

- Original repository: `https://github.com/project-instinct/InstinctLab`
- Original README: `https://github.com/project-instinct/InstinctLab/blob/main/README.md`
- Local reference in this workspace: `../InstinctLab`

## License and Contributing

- License: `CC BY-NC 4.0`, see `LICENSE`
- Contribution guide: `CONTRIBUTING.md`
- Contributor agreement: `CONTRIBUTOR_AGREEMENT.md`

## Task Documentation

- Shadowing: `src/instinct_mj/tasks/shadowing/README.md`
- BeyondMimic: `src/instinct_mj/tasks/shadowing/beyondmimic/README.md`
- Parkour: `src/instinct_mj/tasks/parkour/README.md`
