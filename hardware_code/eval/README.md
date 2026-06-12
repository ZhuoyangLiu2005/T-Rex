# Policy inference client

`eval_trex_async.py` runs a learned policy on the robot by talking to the
T-Rex ZMQ inference server (`T-Rex/scripts/test.py`). The model runs
server-side (any GPU machine); this client only needs the teleop stack's
dependencies.

## Slow / fast protocol

The server is a stateful ZMQ **REP** endpoint; this client orchestrates the
cadence over a **REQ** socket:

- `mode: "slow_and_fast"` — full VLA forward at a chunk start: the server
  caches latent/action KV at the τ-split and returns the action chunk.
- `mode: "fast"` — a small tactile-only payload between chunk starts: the
  server continues flow matching on the cached KV with fresh tactile and
  returns a refined action chunk.
- `mode: "slow"` — full forward without the tactile refinement path.

Typical cadence: slow every 16 robot steps, fast at offsets 0/4/8/12 within
the chunk. ZMQ REP is single-threaded, so a fast request arriving mid-slow
naturally waits until the slow pass finishes.

## Control mode

The client supports one control mode: per action step, the policy outputs a
**delta end-effector pose relative to the chunk-start EEF pose** (3 local xyz
+ 6 rot6d per arm) plus **absolute hand joint targets** (22 per hand) —
62-D dual-arm or 31-D single-arm — resolved to joint commands via
differential IK. Other control modes (absolute EEF, delta/absolute joint
space) are straightforward variations of the chunk-execution loop if your
policy head differs.

## Running

1. Start the inference server (on the GPU machine) — see the T-Rex top-level
   README, "Post-training & inference".
2. Bring up the robot-side processes exactly as for teleop (cameras, hands;
   no Vive/gloves needed) — see the main README.
3. Everything is driven by the YAML config (`inference:` section plus the
   shared robot/cameras/hands/environment sections — see
   `config/default.yaml`):

   ```bash
   cd eval && python eval_trex_async.py --config ../config/default.yaml \
       --task-description "Pour the sugar from the filled cup to the empty cup."
   ```

   Hotkeys during execution: `p` pause/resume, `r` reset trajectory, `q` quit.

> **Reproducing the paper evals:** the released checkpoints were evaluated
> on a bench with `environment.table_height: 0.76` and back/right walls at
> `0.60`/`0.75` m, and the head crop box must match what the checkpoint was
> trained with — set these in your config copy. The torso pose is the config
> default (`[0.9, 1.57, 0.1]`, same as the dataset).

The observation order (proprio → images → tactile-from-buffer) deliberately
matches `main_teleop.py`'s recording order so inference-time inputs are
consistent with the training data.
