from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from custom_envs.joint_1dof_env import Joint1DofEnv
from pipeline.train import _resolve_device
from project_config import ensure_dir, get


def _chirp(t: torch.Tensor, T: float, f0_hz: float, f1_hz: float, amp: float, phase: torch.Tensor) -> torch.Tensor:
    k = (float(f1_hz) - float(f0_hz)) / max(1e-6, float(T))
    ph = 2.0 * np.pi * (float(f0_hz) * t + 0.5 * k * (t * t)) + phase
    return amp * torch.sin(ph)


@torch.no_grad()
def generate_sim_log(cfg: Dict[str, Any], out_npz: str) -> None:
    device = _resolve_device(cfg)
    env_cfg = {
        "num_envs": int(get(cfg, "data.sim.num_trajectories")),
        "frame_dt": float(get(cfg, "sim.frame_dt")),
        "sim_substeps": int(get(cfg, "sim.sim_substeps")),
        "asset_mjcf": str(get(cfg, "paths.asset_mjcf")),
        "mjcf_override": dict(get(cfg, "sim.mjcf_override")),
        "runs_dir": str(get(cfg, "paths.runs_dir")),
    }
    env = Joint1DofEnv(env_cfg, device=device)

    T = int(get(cfg, "data.sim.steps"))
    profile = str(get(cfg, "data.sim.profile", required=False, default="random_uniform"))
    lo = float(get(cfg, "data.sim.torque_low", required=False, default=-0.5))
    hi = float(get(cfg, "data.sim.torque_high", required=False, default=0.5))
    tau_slew = float(get(cfg, "data.sim.tau_slew_nm_s", required=False, default=0.0))
    dt = float(get(cfg, "sim.frame_dt"))

    state = env.reset()  # [K, 2] (q_out, qd_out)
    states = []
    actions = []
    tau_prev = torch.zeros(env.num_envs, 1, device=device)

    phase0 = 2.0 * np.pi * torch.rand(env.num_envs, 1, device=device)
    chirp_cfg = get(cfg, "data.sim.chirp", required=False, default={})
    f0 = float(chirp_cfg.get("f0_hz", 0.1))
    f1 = float(chirp_cfg.get("f1_hz", 5.0))
    amp = float(chirp_cfg.get("amplitude_nm", min(abs(lo), abs(hi))))
    for _ in range(T):
        if profile == "chirp":
            step_idx = len(actions)
            t_s = torch.full((env.num_envs, 1), float(step_idx) * dt, device=device)
            tau = _chirp(t_s, T=float(T) * dt, f0_hz=f0, f1_hz=f1, amp=amp, phase=phase0)
        else:
            tau = (torch.rand(env.num_envs, 1, device=device) * (hi - lo)) + lo

        tau = torch.clamp(tau, min=lo, max=hi)
        if tau_slew > 0:
            max_step = float(tau_slew) * dt
            tau = torch.clamp(tau, min=tau_prev - max_step, max=tau_prev + max_step)
            tau_prev = tau
        states.append(state)
        actions.append(tau)
        state = env.step(tau)

    states_t = torch.stack(states, dim=1).detach().cpu().numpy()  # [K, T, 2]
    actions_t = torch.stack(actions, dim=1).detach().cpu().numpy().squeeze(-1)  # [K, T]

    q_out = states_t[..., 0]
    qd_out = states_t[..., 1]
    tau_cmd = actions_t
    t = np.arange(T, dtype=np.float64) * dt

    ensure_dir(get(cfg, "paths.runs_dir"))
    np.savez(out_npz, t=t, q_out=q_out, qd_out=qd_out, tau_cmd=tau_cmd)
