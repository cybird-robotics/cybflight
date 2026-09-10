"""Export the RL gate-racing policy into cybflight_sim test fixtures.

Usage — run from the RL project root, with its conda env active:

    cd ../../optimal_quad_control_RL
    OUT_DIR=<cybflight>/crates/cybflight_sim/tests/fixtures/rl_race \
        python3 <cybflight>/tools/export_rl_policy.py

Emits into $OUT_DIR:
  policy.bin    flat <f4 weights: per layer [W(out,in) row-major, then b(out)]
  samples.bin   N x (16 world_state_ned + 20 obs + 4 raw_action + 1 gate), <f4
  traj.bin      M x (16 world_state_ned + 4 action + 1 gate), <f4
  meta.json     shapes, params, track, gate-pass events, provenance

The samples deliberately pair each NED world state with the observation the
Python env built from it. That lets the Rust side verify its ENU-native
observation pipeline (and the ENU->NED adapter for these NED-trained
weights) against ground truth instead of against its own re-derivation --
which is what makes `observation_matches_python_ground_truth` a real test
rather than a tautology.

The weight layout is `nn.Linear.weight.numpy().tobytes()` verbatim: no
transpose here and none in `cybflight_core::nn::mlp`, so the two cannot
silently disagree about row- vs column-major.
"""

import os, sys, json
import numpy as np

import numpy.core as _npcore
for _sub in ['', '.numeric', '.multiarray', '.umath', '._multiarray_umath',
             '.numerictypes', '._dtype', '.overrides']:
    try:
        sys.modules['numpy._core' + _sub] = __import__('numpy.core' + _sub, fromlist=['_'])
    except Exception:
        pass

sys.path.insert(0, os.path.abspath('.'))
import torch, torch.nn as nn
from stable_baselines3 import PPO
from quad_race_env import Quadcopter3DGates, gate_pos, gate_yaw, start_pos, f_func
from randomization import randomization_fixed_params_5inch, params_5inch

OUT = os.environ['OUT_DIR']
os.makedirs(OUT, exist_ok=True)
CKPT = 'models/my_session/run0_5inch_10_percent/98000000.zip'
W_MAX_N, W_MIN_N, DT, GATE_SIZE = 3000.0, 0.0, 0.01, 1.5

model = PPO.load(CKPT, device='cpu')
seq = nn.Sequential(*(list(model.policy.mlp_extractor.policy_net) +
                      [model.policy.action_net])).cpu()
layers = [l for l in seq if isinstance(l, nn.Linear)]

# ---- weights -------------------------------------------------------------
blob, shapes = bytearray(), []
for l in layers:
    w = l.weight.data.cpu().numpy().astype('<f4')
    b = l.bias.data.cpu().numpy().astype('<f4')
    shapes.append([int(w.shape[1]), int(w.shape[0])])
    blob += w.tobytes(); blob += b.tobytes()
open(os.path.join(OUT, 'policy.bin'), 'wb').write(bytes(blob))

def fwd(obs):
    with torch.no_grad():
        return seq(torch.tensor(obs[None, :], dtype=torch.float32, device='cpu')).numpy()[0]

env = Quadcopter3DGates(
    num_envs=1, gates_pos=gate_pos, gate_yaw=gate_yaw, start_pos=start_pos,
    randomization=randomization_fixed_params_5inch, gates_ahead=1,
    initialize_at_random_gates=False)

w_hover = float(np.sqrt(9.81 / (4 * params_5inch['k_w'])))
wn = 2 * (w_hover - W_MIN_N) / (W_MAX_N - W_MIN_N) - 1

def reset():
    env.reset()
    env.target_gates[:] = 0
    env.step_counts[:] = 0
    env.world_states[0] = np.array(
        [start_pos[0], start_pos[1], start_pos[2], 0, 0, 0,
         0, 0, gate_yaw[0], 0, 0, 0, wn, wn, wn, wn], dtype=np.float32)
    env.update_states()

# ---- rollout, capturing (world_state, obs, action) -----------------------
reset()
samples, traj, events, passes = [], [], [], 0
for k in range(1200):
    ws = env.world_states[0].copy()
    obs = env.states[0].copy()
    raw = fwd(obs)
    act = np.clip(raw, -1.0, 1.0)
    tg = int(env.target_gates[0]) % env.num_gates

    samples.append(np.concatenate([ws, obs, raw, [tg]]))
    traj.append(np.concatenate([ws, act, [tg]]))

    env.actions = act[None, :]
    new = env.world_states + DT * f_func(env.world_states.T, env.actions.T, env.params.T).T

    pg, yg = env.gate_pos[tg], env.gate_yaw[tg]
    nx, ny = np.cos(yg), np.sin(yg)
    p_old = (ws[0] - pg[0]) * nx + (ws[1] - pg[1]) * ny
    p_new = (new[0, 0] - pg[0]) * nx + (new[0, 1] - pg[1]) * ny
    crossed = (p_old < 0) and (p_new > 0)
    within = bool(np.all(np.abs(new[0, 0:3] - pg) < GATE_SIZE / 2))

    env.world_states = new
    if crossed and within:
        passes += 1
        env.target_gates[0] = (env.target_gates[0] + 1) % env.num_gates
        events.append(['pass', k, tg])
    elif crossed:
        events.append(['collision', k, tg]); break
    if new[0, 2] > 0:
        events.append(['ground', k, -1]); break
    if np.any(np.abs(new[0, 0:2]) > 5) or new[0, 2] < -7:
        events.append(['bounds', k, -1]); break
    env.update_states()

samples = np.array(samples, dtype='<f4')
traj = np.array(traj, dtype='<f4')
# every 8th sample is plenty to pin the observation pipeline
samples[::8].tofile(os.path.join(OUT, 'samples.bin'))
traj.tofile(os.path.join(OUT, 'traj.bin'))

meta = dict(
    checkpoint=CKPT, layer_shapes=shapes,
    n_samples=int(samples[::8].shape[0]), sample_stride=8,
    sample_cols=dict(world_state_ned=16, obs=20, raw_action=4, target_gate=1),
    n_traj=int(traj.shape[0]), traj_cols=dict(world_state_ned=16, action=4, target_gate=1),
    gate_passes=int(passes), events=events,
    params_5inch={k: float(v) for k, v in params_5inch.items()},
    gate_pos_ned=gate_pos.tolist(), gate_yaw_ned=gate_yaw.tolist(),
    start_pos_ned=start_pos.tolist(), start_yaw_ned=float(gate_yaw[0]),
    hover_omega_rad_s=w_hover, hover_omega_norm=float(wn),
    dt=DT, w_min_n=W_MIN_N, w_max_n=W_MAX_N, gate_size=GATE_SIZE,
    log_std=model.policy.log_std.exp().detach().cpu().numpy().tolist(),
)
json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=2)
print(f'{passes} passes, {len(traj)} steps, samples {samples[::8].shape}, shapes {shapes}')
print('last events:', events[-3:])
