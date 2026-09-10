# ArtGym

ArtGym contains Isaac Gym environments and scripts for articulated-object grasping, manipulation, teacher training, student distillation, and simulation evaluation.

## Clone

Clone the repo with submodules:

```bash
git clone --recursive git@github.com:youngcv/artgym.git
```

If you already cloned without `--recursive`, initialize the submodules with:

```bash
git submodule update --init --recursive
```

## Installation

see [install.md](install.md) for installation.

## Submodules

Additional documentation lives in the linked submodules:

- [make_data/README.md](make_data/README.md)
- [func_lygra/README.md](func_lygra/README.md)

## Prerequisites
- Using [make_data](make_data/) to generate ./assets/objects/...
- Using [func_lygra](func_lygra/) to generate ./caches/initial_grasp/...

Verify the file structure matches:

```
./assets/hands/...
./assets/objects/...
./caches/initial_grasp/...
```


## Pipeline

This is the main workflow:

- validate grasps in sim
- train teacher
- evaluate teacher
- save a `success` grasp pool
- select a grasp and run teacher inference in sim
- distill and evaluate the student in sim

### 1. Validate Grasps

Validate all instances of one object class:

```bash
scripts/validate_all_instances.sh \
  sharpa knife \
  --asset-dir knife_30 \
  --pipeline cpu \
  --num-envs 500 \
  --episode-length 30 \
  --rot-threshold 0.1 \
  --pos-threshold 0.01 \
  --no-split \
  --headless \
  --camera
```

Save a deduplicated grasp pool with `--unique`:

```bash
scripts/validate_all_instances.sh \
  sharpa knife \
  --asset-dir knife_30 \
  --pipeline cpu \
  --num-envs 500 \
  --episode-length 30 \
  --rot-threshold 0.1 \
  --pos-threshold 0.01 \
  --unique \
  --unique-pos-threshold 0.005 \
  --unique-rot-threshold 0.05 \
  --no-split \
  --headless \
  --camera
```

Validate one instance:

```bash
python -m isaacgymenvs.valid_grasp \
  --pipeline cpu \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 000 \
  --num-envs 500 \
  --episode-length 30 \
  --rot-threshold 0.1 \
  --pos-threshold 0.01 \
  --headless \
  --camera
```

The unique filter removes a grasp only when both are true:

- object position distance is below `--unique-pos-threshold`
- object rotation distance is below `--unique-rot-threshold`

### 2. Train Teacher

```bash
python -m isaacgymenvs.train \
  task=artmanip \
  hand=sharpa \
  object=knife \
  asset_dir=knife_30 \
  train=artmanipSAPGPrivLSTMPPO \
  task.env.numEnvs=16000 \
  experiment=knife_sapg \
  headless=True \
  task.env.graspSplit=valid
```

Single-machine multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m isaacgymenvs.train \
  task=artmanip \
  hand=sharpa \
  object=knife \
  asset_dir=knife_30 \
  headless=True \
  train=artmanipSAPGPrivLSTMPPO \
  task.env.numEnvs=16000 \
  experiment=knife_sapg \
  task.env.graspSplit=valid \
  multi_gpu=True
```

`task.env.numEnvs` is per GPU.

### 3. Evaluate Consecutive Teacher Success
Evaluate one instance:

```bash
python -m isaacgymenvs.eval_consecutive \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 000 \
  --grasp-split valid \
  --max-steps 1200 \
  --episodes-per-grasp 100 \
  --goal-switch-interval-secs 1.5 \
  --save-success-cycle-threshold 1.0 \
  --save-success-cycle-metric mean \
  --save-success-output-split success \
  --deterministic \
  --headless \
  --randomize true
```

Use this to rank grasps by how many open-close cycles they sustain before failing.

`--grasp-split` can be `train`, `test`, `valid`, `select`, or `success`.

This writes:

- `caches/initial_grasp/sharpa/<asset_dir>/<instance_id>/consecutive_eval_summary.json`

With threshold-based saving, it also writes:

- `caches/initial_grasp/sharpa/<asset_dir>/<instance_id>/success/valid_grasps.npy`
- `caches/initial_grasp/sharpa/<asset_dir>/<instance_id>/success/summary.json`

Batch-evaluate all instances:

```bash
bash scripts/eval_consecutive_all_instances.sh \
  sharpa knife \
  --asset-dir knife_30 \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --grasp-split valid \
  --max-steps 1200 \
  --episodes-per-grasp 100 \
  --goal-switch-interval-secs 1.0 \
  --save-success-cycle-threshold 0 \
  --save-success-cycle-metric mean \
  --save-success-output-split success \
  --deterministic \
  --headless \
  --randomize true
```

This saves:

- `caches/initial_grasp/sharpa/<asset_dir>/<instance_id>/consecutive_eval_summary.json`
- `caches/initial_grasp/sharpa/<asset_dir>/consecutive_eval_asset_summary.json`


### 4. Infer The Teacher In Sim
Choose one cached grasp before inference:
```bash
python select_grasp.py \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 000 \
  --group success \
  --select-idx 0
```
```bash
python -m isaacgymenvs.infer teacher \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --deterministic \
  --randomize false \
  --goal-switch-interval-secs 1.5 \
  --max-steps 300 \
  --save-cur-targets cur_targets.npy \
  --save-video infer.mp4 \
  --headless \
  --checkpoint runs/knife_sapg/best/model.pth \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 000
```

`--goal-switch-interval-secs` is the success-hold time before infer switches between goals.


### 5. Distill Student

```bash
python -m isaacgymenvs.distill \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --num-envs 8000 \
  --updates 10000 \
  --rollout-steps 16 \
  --lr 1e-4 \
  --cosine-coef 0.1 \
  --deterministic \
  --expl-block-idx 0 \
  --headless \
  --save-every-updates 100 \
  --save-best-after-updates 500 \
  --grasp-split valid \
  --custom_tcn
```

When using RTX 50s gpu, --custom_tcn is necessary.

Single-machine multi-GPU distillation:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m isaacgymenvs.distill \
  --multi-gpu \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --num-envs 8000 \
  --updates 100000 \
  --rollout-steps 16 \
  --lr 1e-4 \
  --cosine-coef 0.1 \
  --deterministic \
  --expl-block-idx 0 \
  --headless \
  --save-every-updates 100 \
  --save-best-after-updates 500 \
  --grasp-split valid \
  --custom_tcn
```

`--num-envs` is per GPU, and only rank 0 writes checkpoints, summaries, and periodic eval outputs.

Distillation saves:

- best-loss distilled checkpoint at the resolved student output path
- best-reward checkpoint beside it, named like `proprio_only_best_reward.pth`
- optional periodic checkpoints like `proprio_only_update0100.pth` when `--save-every-updates` is enabled

Use `--save-best-after-updates 50` to delay best-loss and best-reward checkpoint tracking.

### 2. Evaluate Student

```bash
python -m isaacgymenvs.eval_consecutive \
  --student-artifact runs/knife_sapg/nn/student/proprio_only_update0100.pth \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 030 \
  --grasp-split valid \
  --episodes-per-grasp 100 \
  --goal-switch-interval-secs 1.5 \
  --max-steps 1200 \
  --deterministic \
  --torch-deterministic true \
  --headless \
  --randomize true \
  --progress-interval-sec 2
```

evaluate all instances:

```bash
bash scripts/eval_consecutive_all_instances.sh \
  sharpa knife \
  --asset-dir knife_30 \
  --student-artifact runs/knife_sapg/nn/student/proprio_only_update0100.pth \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --grasp-split valid \
  --episodes-per-grasp 100 \
  --goal-switch-interval-secs 1.0 \
  --max-steps 1200 \
  --deterministic \
  --headless \
  --randomize true \
  --save-success-cycle-threshold 0 \
  --save-success-cycle-metric mean \
  --save-success-output-split success_student
```

### 3. Infer Student

```bash
python -m isaacgymenvs.infer student \
  --student-artifact runs/knife_sapg/nn/student/proprio_only_update0100.pth \
  --checkpoint runs/knife_sapg/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --asset-dir knife_30 \
  --instance-id 000 \
  --deterministic \
  --torch-deterministic true \
  --randomize false \
  --goal-switch-interval-secs 1.5 \
  --max-steps 300 \
  --save-cur-targets student_cur_targets.npy \
  --save-video student_infer.mp4 \
  --headless
```

## PPO Variant

To run PPO instead of SAPG, use the same commands and change only the train config name from `train=artmanipSAPGPrivLSTMPPO` to `train=artmanipPrivLSTMPPO`.

## Real-World Deployment

refer to [deploy.md](deploy.md) for details
