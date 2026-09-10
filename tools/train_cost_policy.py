#!/usr/bin/env python3
"""Train the situation-conditioned NMPC cost policy (docs/learned_mpc_cost.md, PLAN B).

The environment is the Rust `cybflight_rl.VecEnv` (recovery episodes on
procedural maneuver primitives, stepped by the real SQP-RTI + INDI + plant).
PPO comes from Stable-Baselines3; the actor is exported to the flat format
`cybflight_sim::controller::OwnedCostPolicy::from_file` reads.

    # build the bridge once (from the repo root):
    maturin develop --release -m crates/cybflight_rl/Cargo.toml --target x86_64-unknown-linux-gnu
    # train (≈ 10 M env steps, minutes on a 24-core host):
    python3 tools/train_cost_policy.py --steps 10_000_000 --out target/cost_policy
    # the constant-weight control (observation zeroed):
    python3 tools/train_cost_policy.py --blind --steps 3_000_000 --out target/cost_policy_blind

Outputs in --out: `ppo.zip` (SB3 checkpoint), `policy.bin` (Rust export),
`log.csv` (rollout statistics), `probe.json` (weight-vs-situation sweeps).
"""

import argparse
import csv
import json
import os
import struct
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from gymnasium import spaces

import cybflight_rl

OBS = cybflight_rl.OBS_DIM
NZ = cybflight_rl.NZ
# Checkpoint header, mirrored in crates/cybflight/build.rs.
MAGIC = 0x50434643  # b"CFCP" little-endian
FORMAT = 1
MAP_VERSION = cybflight_rl.COST_MAP_VERSION


class RustVecEnv(SB3VecEnv):
    """SB3 VecEnv over the Rust batch environment (auto-reset inside Rust)."""

    def __init__(self, n, vehicle, seed, **kw):
        self.env = cybflight_rl.VecEnv(n, vehicle, seed=seed, **kw)
        super().__init__(
            n,
            spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
            spaces.Box(-1.0, 1.0, (NZ,), np.float32),
        )
        self.actions = None
        self.ep_ret = np.zeros(n, np.float32)
        self.ep_len = np.zeros(n, np.int32)
        self.stats = []  # per finished episode: (return, length, crashed, kind, peak_thrust)

    def reset(self):
        self.ep_ret[:] = 0
        self.ep_len[:] = 0
        return self.env.reset()

    def step_async(self, actions):
        self.actions = np.ascontiguousarray(actions, dtype=np.float32)

    def step_wait(self):
        obs, r, done, info = self.env.step(self.actions)
        self.ep_ret += r
        self.ep_len += 1
        infos = []
        for i in range(self.num_envs):
            d = {"e_geom": float(info["e_geom"][i]), "e_time": float(info["e_time"][i])}
            if done[i]:
                d["terminal_observation"] = info["terminal_obs"][i]
                d["TimeLimit.truncated"] = bool(info["truncated"][i])
                self.stats.append(
                    (self.ep_ret[i], self.ep_len[i], bool(info["crashed"][i]), int(info["kind"][i]),
                     float(info["peak_thrust"][i]))
                )
                self.ep_ret[i] = 0
                self.ep_len[i] = 0
            infos.append(d)
        return obs, r, done, infos

    def close(self):
        pass

    # SB3 VecEnv abstract surface we do not need.
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs


def export_policy(model: PPO, path: str):
    """Actor MLP (policy_net + action_net) → flat `nn::Mlp` file.

    The header stamps the cost-map revision the policy was trained
    against. The firmware bake asserts it, because the shape checks it
    can otherwise make (input/output width, layer chain) are blind to a
    map change that leaves the dimensions alone — which is exactly what
    a fence-constant bump does.
    """
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, th.nn.Linear)]
    layers.append(pol.action_net)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", MAGIC, FORMAT, MAP_VERSION))
        f.write(struct.pack("<I", len(layers)))
        for l in layers:
            f.write(struct.pack("<II", l.in_features, l.out_features))
        for l in layers:
            f.write(l.weight.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(l.bias.detach().cpu().numpy().astype("<f4").tobytes())
    return [(l.in_features, l.out_features) for l in layers]


def probe(model: PPO, path: str):
    """Sweep one situation input at a time from a level-flight baseline and
    record the deterministic z — the check that the learned map is
    situational (docs/learned_mpc_cost.md §3, verification 1)."""
    base = np.zeros(OBS, np.float32)
    base[8] = 1.0   # vehicle body z = +b̂ (level)
    base[11] = 1.0  # reference body z = +b̂
    base[21] = 0.5  # ‖v_ref‖ = 5 m/s
    for j in range(5):
        b = 22 + 7 * j
        base[b] = 1.0        # cos θ_ref = 1
        base[b + 3] = 1.0    # ref body z = +b̂
        base[b + 5] = 0.6    # thrust margin
        base[b + 6] = 0.8    # rate margin
    base[57] = 0.25
    sweeps = {
        "contour_error_m": (1, np.linspace(-1.5, 1.5, 13)),
        "lag_error_m": (0, np.linspace(-1.5, 1.5, 13)),
        "vel_error_along_m_s": (3, np.linspace(-1.0, 1.0, 13)),  # /5
        "ref_tilt_cos_preview10": (22 + 7 * 2, np.linspace(1.0, -1.0, 13)),
        "thrust_margin_preview0": (27, np.linspace(1.0, -0.5, 13)),
        "att_error_roll_rad": (12, np.linspace(-1.5, 1.5, 13)),
    }
    out = {}
    for name, (idx, vals) in sweeps.items():
        rows = []
        for v in vals:
            o = base.copy()
            o[idx] = v
            z, _ = model.predict(o[None], deterministic=True)
            rows.append({"x": float(v), "z": np.clip(z[0], -1, 1).tolist()})
        out[name] = rows
    json.dump(out, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="vehicles/sakura_bench_leader_1khz.yaml")
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--out", default="target/cost_policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true", help="zero the observation (constant-z control)")
    ap.add_argument("--disturb", type=float, default=1.0)
    ap.add_argument("--log-std-init", type=float, default=-1.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--w-time", type=float, default=1.0)
    ap.add_argument("--w-geom", type=float, default=10.0)
    ap.add_argument("--w-dz", type=float, default=0.1)
    ap.add_argument("--w-du-rate", type=float, default=25.0)
    ap.add_argument("--gyro-sigma", type=float, default=0.03)
    ap.add_argument("--mass-jitter", type=float, default=0.2)
    ap.add_argument("--rate-nominal", type=float, default=5.0)
    ap.add_argument("--sampler-max-lag", type=float, default=0.1)
    ap.add_argument("--model-drag", action="store_true")
    ap.add_argument("--body-drag", type=float, default=0.0)
    ap.add_argument("--domain-rand", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.set_num_threads(4)

    env = RustVecEnv(args.envs, args.vehicle, args.seed, blind=args.blind, disturb=args.disturb,
                     mass_jitter=args.mass_jitter,
                     w_time=args.w_time, w_geom=args.w_geom, w_dz=args.w_dz,
                     w_du_rate=args.w_du_rate, gyro_sigma=args.gyro_sigma, rate_weight_nominal=args.rate_nominal, sampler_max_lag=args.sampler_max_lag,
                     model_drag=args.model_drag, body_drag=args.body_drag,
                     domain_rand=args.domain_rand)
    if args.resume:
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.envs * args.n_steps // 8,
            n_epochs=6,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=args.lr,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=th.nn.Tanh,
                log_std_init=args.log_std_init,
            ),
            seed=args.seed,
            verbose=0,
            device="cpu",
        )

    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["steps", "wall_s", "episodes", "mean_return", "mean_len", "crash_rate", "mean_e_geom_by_kind"])
    t0 = time.time()
    per_iter = args.envs * args.n_steps
    done = 0
    best = -np.inf
    while done < args.steps:
        model.learn(total_timesteps=per_iter, reset_num_timesteps=False)
        done += per_iter
        st = env.stats
        env.stats = []
        if st:
            ret = np.array([s[0] for s in st])
            ln = np.array([s[1] for s in st])
            cr = np.array([s[2] for s in st], dtype=float)
            kinds = np.array([s[3] for s in st])
            by_kind = {int(k): float(ret[kinds == k].mean()) for k in np.unique(kinds)}
            row = [done, round(time.time() - t0, 1), len(st), round(float(ret.mean()), 3),
                   round(float(ln.mean()), 1), round(float(cr.mean()), 4), json.dumps(by_kind)]
            w.writerow(row)
            log.flush()
            print(f"steps={done:>9d} t={time.time()-t0:6.0f}s eps={len(st):5d} "
                  f"ret={ret.mean():8.3f} len={ln.mean():6.1f} crash={cr.mean():.3f} "
                  f"std={float(th.exp(model.policy.log_std).mean()):.3f}", flush=True)
            if ret.mean() > best:
                best = ret.mean()
                model.save(os.path.join(args.out, "ppo_best.zip"))
                export_policy(model, os.path.join(args.out, "policy_best.bin"))
    model.save(os.path.join(args.out, "ppo.zip"))
    shapes = export_policy(model, os.path.join(args.out, "policy.bin"))
    probe(model, os.path.join(args.out, "probe.json"))
    print("exported", shapes, "→", args.out)


if __name__ == "__main__":
    main()
