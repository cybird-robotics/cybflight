"""Export the trained ACMPC policy into cybflight_sim test fixtures.

Usage — run from the cybflight-isaac project root, with its conda env active:

    conda activate acmpc
    cd ../../cybflight-isaac
    OUT_DIR=<cybflight>/crates/cybflight_sim/tests/fixtures/acmpc_race \
        python3 <cybflight>/tools/export_acmpc_policy.py

Emits into $OUT_DIR:
  cost_net.bin  flat <f4 weights of the cost network's 4 nn.Linear layers,
                per layer [W(out,in) row-major, then b(out)]
  samples.bin   N x (16 world_state_ned + 30 obs + 140 cost_logits
                     + 4 mpc_u0 + 4 action + 1 gate), <f4
  traj.bin      M x (16 world_state_ned + 4 action + 1 gate), <f4
  meta.json     shapes, hyperparameters, track, gate-pass events, provenance

The columns are laid out so the Rust ladder can localize a failure to one
stage: `cost_logits` is the cost network *before* its sigmoid head (the
sharpest possible check on the GELU tower — the sigmoid would compress a
divergence away), `mpc_u0` is the box-DDP's first control before the CTBR
normalization, and `action` is what SB3's deterministic `predict` returns.

The rollout is deliberately NOT BatchedRaceEnv.step: the env auto-resets on
termination, and the fixture must record a single uninterrupted flight from
a fixed start. It reuses the env's own pieces (`ctbr_to_motor`,
`model_derivatives`, `RaceCourse`) so the physics and rules cannot drift.

The weight layout is `nn.Linear.weight.numpy().tobytes()` verbatim: no
transpose here and none in `cybflight_core::nn::mlp`.
"""

import json
import os

import numpy as np
import torch as th
from stable_baselines3 import PPO
import yaml

from cybflight import ctbr
from cybflight.config import CtbrConfig, RaceTrackConfig
from cybflight.course import RaceCourse
import cybflight.dynamics as dyn
from cybflight.policy_acmpc import MpcActorCriticPolicy  # noqa: F401  (unpickling)

OUT = os.environ["OUT_DIR"]
CKPT = os.environ.get("CKPT", "train_out/cmp_acmpc_final.zip")
TRACK = os.environ.get("TRACK", "examples/conf/track/figure8.yaml")
STEPS, SAMPLE_STRIDE, OBS_PREFIX = 1200, 8, 20

os.makedirs(OUT, exist_ok=True)
th.set_num_threads(1)

model = PPO.load(CKPT, device="cpu")
extractor = model.policy.mlp_extractor
cost_net, acfg, ccfg = extractor.cost_net, extractor.cfg, extractor.ctbr_cfg

# ---- weights -------------------------------------------------------------
linears = [m for m in cost_net.net if isinstance(m, th.nn.Linear)]
blob, shapes = bytearray(), []
for layer in linears:
    w = layer.weight.data.numpy().astype("<f4")
    shapes.append([int(w.shape[1]), int(w.shape[0])])
    blob += w.tobytes()
    blob += layer.bias.data.numpy().astype("<f4").tobytes()
open(os.path.join(OUT, "cost_net.bin"), "wb").write(bytes(blob))

# The cost head without its sigmoid: the raw tower output.
logit_net = th.nn.Sequential(*list(cost_net.net)[:-1])

course = RaceCourse(RaceTrackConfig.from_dict(yaml.safe_load(open(TRACK).read())))
params = dyn.PARAMS_5INCH


def evaluate(obs):
    """(30,) obs -> (cost_logits, mpc_u0, deterministic action)."""
    with th.no_grad():
        f = th.tensor(obs[None], dtype=th.float32)
        logits = logit_net(f[:, :OBS_PREFIX])
        c_mat, c_vec = cost_net(f[:, :OBS_PREFIX])
        u0 = extractor._solve_chunk(f[:, OBS_PREFIX:], c_mat, c_vec)
        action = extractor.forward_actor(f).numpy()[0]
    return logits.numpy()[0], u0.numpy()[0], np.clip(action, -1.0, 1.0)


# ---- rollout, capturing (world_state, obs, cost, u0, action) -------------
# Same initial condition as tests/nn_gate_race.rs: 1 m behind gate 0, level,
# facing it, rotors at hover — so the two methods are compared from an
# identical state, not merely on an identical course.
w_hover = float(np.sqrt(dyn.G / (4 * params.k_w)))
wn = 2 * (w_hover - dyn.W_MIN_N) / (dyn.W_MAX_N - dyn.W_MIN_N) - 1
state = np.zeros((1, 16))
state[0, 0:3] = course.start_pos
state[0, 8] = course.gate_yaw[0]
state[0, 12:16] = wn

samples, traj, events, passes = [], [], [], 0
target = 0
for k in range(STEPS):
    ws = state[0].copy()
    obs = course.build_obs(state, np.array([target]), include_raw_state=True)
    logits, u0, action = evaluate(obs)

    samples.append(np.concatenate([ws, obs, logits, u0, action, [target]]))
    traj.append(np.concatenate([ws, action, [target]]))

    u = ctbr.ctbr_to_motor(
        action[None], state[:, 9:12], state[:, 12:16], params, ccfg
    )
    new = state + dyn.DT * dyn.model_derivatives(state, u, params)
    new[:, 12:16] = np.clip(new[:, 12:16], -1.0, 1.0)

    p_old, p_new = (
        np.array([ws[1], ws[0], -ws[2]]),
        np.array([new[0, 1], new[0, 0], -new[0, 2]]),
    )
    passed, collided = course.gate_passed(p_old, p_new, target)
    state = new
    if passed:
        passes += 1
        target = (target + 1) % course.num_gates
        events.append(["pass", k, int(target)])
    elif collided:
        events.append(["collision", k, int(target)])
        break
    if p_new[2] < 0.0:
        events.append(["ground", k, -1])
        break
    if abs(p_new[0]) > 5.0 or abs(p_new[1]) > 5.0:
        events.append(["bounds", k, -1])
        break

samples = np.array(samples, dtype="<f4")[::SAMPLE_STRIDE]
traj = np.array(traj, dtype="<f4")
samples.tofile(os.path.join(OUT, "samples.bin"))
traj.tofile(os.path.join(OUT, "traj.bin"))

json.dump(
    dict(
        checkpoint=CKPT,
        num_timesteps=int(model.num_timesteps),
        layer_shapes=shapes,
        n_samples=int(samples.shape[0]),
        sample_stride=SAMPLE_STRIDE,
        sample_cols=dict(
            world_state_ned=16, obs=30, cost_logits=int(2 * acfg.horizon * 14),
            mpc_u0=4, action=4, target_gate=1,
        ),
        n_traj=int(traj.shape[0]),
        traj_cols=dict(world_state_ned=16, action=4, target_gate=1),
        gate_passes=int(passes),
        events=events,
        acmpc=dict(
            horizon=acfg.horizon, mpc_dt=acfg.mpc_dt, lqr_iter=acfg.lqr_iter,
            range_q=acfg.range_q, range_p=acfg.range_p,
            linesearch_decay=acfg.linesearch_decay,
            max_linesearch_iter=acfg.max_linesearch_iter,
        ),
        ctbr=dict(
            omega_max=list(ccfg.omega_max), rate_gains=list(ccfg.rate_gains),
            newton_iters=ccfg.newton_iters, a_max=ctbr.A_MAX,
        ),
        params_5inch=params._asdict(),
        gate_pos_ned=course.gate_pos.tolist(),
        gate_yaw_ned=course.gate_yaw.tolist(),
        start_pos_ned=course.start_pos.tolist(),
        start_yaw_ned=float(course.gate_yaw[0]),
        hover_omega_rad_s=w_hover,
        dt=dyn.DT, w_min_n=dyn.W_MIN_N, w_max_n=dyn.W_MAX_N,
        gate_size=course.gate_size, gravity=dyn.G,
    ),
    open(os.path.join(OUT, "meta.json"), "w"),
    indent=2,
)
print(f"{passes} passes, {len(traj)} steps, samples {samples.shape}, shapes {shapes}")
print("last events:", events[-3:])
