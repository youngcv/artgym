# Real Robot Deploy Commands


## Select One Grasp

Choose one cached grasp before running real-hand commands:

```bash
python select_grasp.py \
  --hand sharpa \
  --object knife \
  --asset-dir knife_real \
  --instance-id 000 \
  --group valid \
  --select-idx 0
```

This writes:

```text
caches/initial_grasp/sharpa/<asset_dir>/<instance_id>/selected_grasps.npy
```
`--grasp-split` can be `train`, `test`, `valid`, `select`, or `success`.

## Load Selected Grasp On Sharpa

```bash
python -m isaacgymenvs.deploy test-grasp \
  --hand sharpa \
  --object knife \
  --asset-dir knife_real \
  --grasp-instance-id 000 \
  --robot-kwargs-json '{"sdk_root":"/opt/sharpa-wave-sdk","hand_side":"left","zero_state_on_connect":true}'
```

## Replay Saved Cur Targets On Sharpa

```bash
python -m isaacgymenvs.deploy replay \
  --hand sharpa \
  --cur-targets-npy cur_targets.npy \
  --robot-kwargs-json '{"sdk_root":"/opt/sharpa-wave-sdk","hand_side":"left","zero_state_on_connect":true}' \
  --replay-hz 30 \
  --init-grasp-settle-sec 1.0 \
  --max-steps 0
```

Replay only commands the saved hand trajectory.

## Start Student Policy Server

```bash
python -m isaacgymenvs.deploy server \
  --student-artifact runs/knife/student/converged.pth \
  --checkpoint runs/knife/best/model.pth \
  --train artmanipSAPGPrivLSTMPPO \
  --hand sharpa \
  --object knife \
  --rl-device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

Deployment goal switching is manual-only. The server always opens a goal prompt; press Enter in the server terminal to switch to the next configured goal, or type a numeric goal offset to override it.

## Run Student On Real Hand Through The RPC Client

```bash
python -m isaacgymenvs.deploy client \
  --host 127.0.0.1 \
  --port 5555 \
  --task artmanip \
  --hand sharpa \
  --object knife \
  --asset-dir knife_real \
  --robot-kwargs-json '{"sdk_root":"/opt/sharpa-wave-sdk","hand_side":"left","zero_state_on_connect":true}' \
  --grasp-instance-id 000 \
  --deterministic \
  --print-every 10 \
  --goal-offset 0.04 \
  --use-measured-init-hand-qpos true \
  --use-measured-init-fingertip-pos true
```

For a proprio-only student, cached init state is usually closer to the sim reset distribution:

```text
--use-measured-init-hand-qpos false
--use-measured-init-fingertip-pos false
```

## Notes

- `server` and `client` may use different conda env.
- To add another hand, create `isaacgymenvs/deploy/<hand>/robot.py` and `isaacgymenvs/deploy/<hand>/observation_provider.py` implementing the general APIs, then run with `--hand <hand>`.

