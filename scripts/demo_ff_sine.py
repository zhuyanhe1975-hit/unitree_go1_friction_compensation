from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from pipeline.config import load_cfg
from project_config import ensure_dir, get


@dataclass
class TrialResult:
    mode: str
    path: str


class UnitreeMotor:
    """
    Minimal Unitree GO-M8010-6 interface for position tracking with torque feedforward.
    Uses MotorCmd fields: q, dq, kp, kd, tau (feedforward).
    """

    def __init__(self, cfg: dict):
        def _try_add_sdk_path(p: str) -> bool:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                # Make sure dependent .so (libUnitreeMotorSDK_*.so) can be found at import time.
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                if p not in ld.split(":"):
                    os.environ["LD_LIBRARY_PATH"] = (p + (":" + ld if ld else ""))
                sys.path.insert(0, p)
                return True
            return False

        # Priority:
        # 1) env UNITREE_ACTUATOR_SDK_LIB
        # 2) config real.unitree_sdk_lib
        # 3) best-effort known local path
        env_sdk = os.environ.get("UNITREE_ACTUATOR_SDK_LIB", "").strip()
        sdk_lib = get(cfg, "real.unitree_sdk_lib", required=False, default=None)
        ok = False
        if env_sdk:
            ok = _try_add_sdk_path(env_sdk)
        if (not ok) and sdk_lib:
            ok = _try_add_sdk_path(str(sdk_lib))
        if not ok:
            local_sdk = "/home/yhzhu/Industrial Robot/unitree_actuator_sdk/lib"
            _try_add_sdk_path(local_sdk)

        try:
            import unitree_actuator_sdk as u  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import unitree_actuator_sdk. Set env UNITREE_ACTUATOR_SDK_LIB or "
                "set `real.unitree_sdk_lib` in config.json, or build it using scripts/build_unitree_sdk.py. "
                f"Original error: {e}"
            ) from e

        self.u = u
        port = str(get(cfg, "real.serial_port"))
        try:
            self.serial = u.SerialPort(port)
        except RuntimeError as e:
            msg = str(e)
            if "Permission denied" in msg or "IO Exception (13)" in msg:
                raise RuntimeError(
                    f"Failed to open serial port {port} (permission denied).\n"
                    f"- Quick test: run with sudo: `sudo {sys.executable} scripts/demo_ff_sine.py ...`\n"
                    f"- Proper fix: add user to dialout then re-login:\n"
                    f"    `sudo usermod -a -G dialout $USER`\n"
                    f"Original error: {msg}"
                ) from e
            raise

        self.cmd = u.MotorCmd()
        self.data = u.MotorData()
        self.data.motorType = u.MotorType.GO_M8010_6
        self.cmd.motorType = u.MotorType.GO_M8010_6
        self.cmd.mode = u.queryMotorMode(u.MotorType.GO_M8010_6, u.MotorMode.FOC)
        self.cmd.id = int(get(cfg, "real.motor_id"))

        # prime
        self.cmd.q = 0.0
        self.cmd.dq = 0.0
        self.cmd.kp = 0.0
        self.cmd.kd = 0.0
        self.cmd.tau = 0.0
        self.serial.sendRecv(self.cmd, self.data)

    def read(self) -> tuple[float, float, float, float, float]:
        # q, dq, tau, temp, merror
        return float(self.data.q), float(self.data.dq), float(self.data.tau), float(self.data.temp), float(self.data.merror)

    def send(self, q_ref: float, dq_ref: float, kp: float, kd: float, tau_ff: float) -> None:
        self.cmd.q = float(q_ref)
        self.cmd.dq = float(dq_ref)
        self.cmd.kp = float(kp)
        self.cmd.kd = float(kd)
        self.cmd.tau = float(tau_ff)
        self.serial.sendRecv(self.cmd, self.data)

    def shutdown(self) -> None:
        try:
            self.cmd.tau = 0.0
            self.cmd.kp = 0.0
            self.cmd.kd = 0.0
            self.serial.sendRecv(self.cmd, self.data)
        except Exception:
            pass


class FrictionPredictor:
    """
    Online friction-like torque predictor:
      input: history of [sin(q), cos(q), qd, (temp?)]
      output: tau_fric (Nm)
    """

    def __init__(self, model_path: str, dataset_npz: str, device: str = "cpu"):
        import torch
        from pipeline.model import CausalTransformer
        from pipeline.features import state_to_features

        self._torch = torch
        self._state_to_features = state_to_features

        ds = dict(np.load(dataset_npz, allow_pickle=True))
        self.x_mean = ds["x_mean"].astype(np.float32)
        self.x_std = ds["x_std"].astype(np.float32)
        self.y_mean = ds["y_mean"].astype(np.float32)
        self.y_std = ds["y_std"].astype(np.float32)
        # History length used for normalization stats; the model checkpoint may override this.
        self.H = int(ds["x"].shape[1])
        self.D = int(self.x_mean.shape[0])

        ckpt = torch.load(model_path, map_location=torch.device(device))
        self.model = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(ckpt.get("embed_dim", 64)),
            num_layers=int(ckpt.get("num_layers", 2)),
            num_heads=int(ckpt.get("num_heads", 4)),
            history_len=int(ckpt.get("history_len", self.H)),
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device

        self.H = int(ckpt.get("history_len", self.H))
        self.buf = deque(maxlen=self.H)
        # Preallocate a CPU buffer to avoid per-step allocations; small but helps jitter.
        self._x_buf = np.zeros((self.H, self.D), dtype=np.float32)
        # Warm up a few calls to reduce first-use spikes.
        with torch.no_grad():
            dummy = torch.zeros((1, self.H, int(ckpt["input_dim"])), dtype=torch.float32, device=device)
            for _ in range(5):
                _ = self.model(dummy)

    def reset(self) -> None:
        self.buf.clear()

    def update_and_predict(self, q: float, qd: float, temp: float | None = None) -> float | None:
        feat = self._state_to_features(np.array([q], dtype=np.float64), np.array([qd], dtype=np.float64)).astype(np.float32).reshape(-1)
        if self.D == 4:
            if temp is None:
                temp = 0.0
            feat = np.concatenate([feat, np.array([float(temp)], dtype=np.float32)], axis=0)
        self.buf.append(feat.astype(np.float32))
        if len(self.buf) < self.buf.maxlen:
            return None
        # Copy ring-buffer into contiguous array without allocating.
        for i, row in enumerate(self.buf):
            self._x_buf[i] = row
        x_n = (self._x_buf - self.x_mean) / self.x_std
        with self._torch.no_grad():
            xb = self._torch.from_numpy(x_n[None, ...]).to(self.device)
            y_n = self.model(xb).detach().cpu().numpy().reshape(-1)
        y = float(y_n[0] * self.y_std[0] + self.y_mean[0])
        return y


class TorqueDeltaPredictor:
    """
    Online torque-delta predictor:
      input: history of [sin(q), cos(q), qd, (temp?), tau_out] ending at (k-1)
      output: delta_tau_out[k] = tau_out[k] - tau_out[k-1]

    Then a one-step torque estimate can be constructed:
      tau_out_hat[k] = tau_out[k-1] + delta_tau_pred[k]
    """

    def __init__(self, model_path: str, dataset_npz: str, device: str = "cpu"):
        import torch
        from pipeline.model import CausalTransformer
        from pipeline.features import state_error_to_features, state_to_features

        self._torch = torch
        self._state_to_features = state_to_features
        self._state_error_to_features = state_error_to_features

        ds = dict(np.load(dataset_npz, allow_pickle=True))
        self.x_mean = ds["x_mean"].astype(np.float32)
        self.x_std = ds["x_std"].astype(np.float32)
        self.y_mean = ds["y_mean"].astype(np.float32)
        self.y_std = ds["y_std"].astype(np.float32)
        self.H = int(ds["x"].shape[1])
        self.D = int(self.x_mean.shape[0])

        ckpt = torch.load(model_path, map_location=torch.device(device))
        self.model = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(ckpt.get("embed_dim", 64)),
            num_layers=int(ckpt.get("num_layers", 2)),
            num_heads=int(ckpt.get("num_heads", 4)),
            history_len=int(ckpt.get("history_len", self.H)),
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device

        self.H = int(ckpt.get("history_len", self.H))
        self.buf = deque(maxlen=self.H)
        self._x_buf = np.zeros((self.H, self.D), dtype=np.float32)

        with torch.no_grad():
            dummy = torch.zeros((1, self.H, int(ckpt["input_dim"])), dtype=torch.float32, device=device)
            for _ in range(5):
                _ = self.model(dummy)

    def reset(self) -> None:
        self.buf.clear()

    def update_and_predict(
        self,
        q: float,
        qd: float,
        tau_out: float,
        temp: float | None = None,
        q_ref: float | None = None,
        qd_ref: float | None = None,
    ) -> float | None:
        """
        For v1 datasets:
          D=4: [sin, cos, qd, tau_out]
          D=5: [sin, cos, qd, temp, tau_out]
        For v2 datasets:
          D=6: [sin, cos, qd, e_q, e_qd, tau_out]
          D=7: [sin, cos, qd, e_q, e_qd, temp, tau_out]
        """
        if self.D in (6, 7):
            if q_ref is None or qd_ref is None:
                raise ValueError("torque-delta v2 requires q_ref and qd_ref for controller-context features")
            feat = (
                self._state_error_to_features(
                    np.array([q], dtype=np.float64),
                    np.array([qd], dtype=np.float64),
                    np.array([q_ref], dtype=np.float64),
                    np.array([qd_ref], dtype=np.float64),
                )
                .astype(np.float32)
                .reshape(-1)
            )
            if self.D == 7:
                if temp is None:
                    temp = 0.0
                feat = np.concatenate(
                    [feat, np.array([float(temp)], dtype=np.float32), np.array([float(tau_out)], dtype=np.float32)], axis=0
                )
            else:
                feat = np.concatenate([feat, np.array([float(tau_out)], dtype=np.float32)], axis=0)
        else:
            feat = self._state_to_features(np.array([q], dtype=np.float64), np.array([qd], dtype=np.float64)).astype(np.float32).reshape(-1)
            # Torque-delta dataset always appends tau_out as the last channel; temp may optionally exist.
            if self.D == 5:
                if temp is None:
                    temp = 0.0
                feat = np.concatenate(
                    [feat, np.array([float(temp)], dtype=np.float32), np.array([float(tau_out)], dtype=np.float32)], axis=0
                )
            else:
                feat = np.concatenate([feat, np.array([float(tau_out)], dtype=np.float32)], axis=0)

        self.buf.append(feat.astype(np.float32))
        if len(self.buf) < self.buf.maxlen:
            return None

        for i, row in enumerate(self.buf):
            self._x_buf[i] = row
        x_n = (self._x_buf - self.x_mean) / self.x_std
        with self._torch.no_grad():
            xb = self._torch.from_numpy(x_n[None, ...]).to(self.device)
            y_n = self.model(xb).detach().cpu().numpy().reshape(-1)
        delta = float(y_n[0] * self.y_std[0] + self.y_mean[0])
        return delta


def _sine_ref(t: float, q0: float, amp: float, freq_hz: float) -> tuple[float, float]:
    w = 2.0 * np.pi * float(freq_hz)
    q_ref = float(q0 + amp * np.sin(w * t))
    dq_ref = float(amp * w * np.cos(w * t))
    return q_ref, dq_ref


def _run_trial(
    cfg: dict,
    mode: str,
    motor: UnitreeMotor,
    friction: FrictionPredictor | None,
    torque_delta: TorqueDeltaPredictor | None,
    out_npz: str,
    duration_s: float,
    dt: float,
    q0: float,
    amp: float,
    freq_hz: float,
    kp: float,
    kd: float,
    tau_ff_limit: float,
    tau_ff_slew: float,
    tau_ff_scale: float,
    ff_update_div: int,
    speed_limit: float,
    temp_limit_c: float,
    abort_on_merror: bool,
) -> None:
    ensure_dir(os.path.dirname(out_npz) or ".")
    if friction is not None:
        friction.reset()
    if torque_delta is not None:
        torque_delta.reset()

    logs: dict[str, list[float]] = {
        k: []
        for k in [
            "t",
            "q",
            "qd",
            "q_ref",
            "qd_ref",
            "e_q",
            "e_qd",
            "kp",
            "kd",
            "tau_cmd_ff",
            "tau_ff",
            "tau_out",
            "temp",
            "merror",
            "loop_dt",
        ]
    }

    tau_ff_prev = 0.0
    start = time.perf_counter()
    next_t = start
    last_loop_t = start

    stop = False

    def _sigint(_sig, _frame):
        nonlocal stop
        stop = True

    old = signal.signal(signal.SIGINT, _sigint)
    try:
        while True:
            now = time.perf_counter()
            t = now - start
            if stop or t >= duration_s:
                break

            q, qd, tau_out, temp, merror = motor.read()
            if abort_on_merror and int(merror) != 0:
                raise RuntimeError(f"motor merror={int(merror)}")
            if float(temp) >= temp_limit_c:
                raise RuntimeError(f"motor temp={float(temp):.1f}C >= limit={temp_limit_c:.1f}C")
            if speed_limit > 0.0 and abs(qd) > speed_limit:
                raise RuntimeError(f"qd={qd:+.3f} rad/s exceeds limit {speed_limit:+.3f} rad/s")

            q_ref, dq_ref = _sine_ref(t, q0=q0, amp=amp, freq_hz=freq_hz)

            tau_ff = 0.0
            if mode == "ff" and friction is not None:
                # Update the feedforward at a reduced rate if requested; hold otherwise.
                if ff_update_div <= 1 or (len(logs["t"]) % ff_update_div == 0):
                    pred = friction.update_and_predict(q=q, qd=qd, temp=temp)
                    if pred is not None:
                        tau_ff = float(tau_ff_scale) * float(pred)
                    else:
                        tau_ff = 0.0
                else:
                    tau_ff = float(tau_ff_prev)
            elif mode == "ff" and torque_delta is not None:
                if ff_update_div <= 1 or (len(logs["t"]) % ff_update_div == 0):
                    delta = torque_delta.update_and_predict(q=q, qd=qd, tau_out=tau_out, temp=temp, q_ref=q_ref, qd_ref=dq_ref)
                    if delta is not None:
                        # User-requested construction: tau_ff[k] = tau_out[k-1] + delta_tau_out_pred[k]
                        # Here we approximate tau_out[k-1] with the latest measured tau_out (since we update before send).
                        tau_hat = float(tau_out) + float(delta)
                        tau_ff = float(tau_ff_scale) * float(tau_hat)
                    else:
                        tau_ff = 0.0
                else:
                    tau_ff = float(tau_ff_prev)

            # clamp + slew
            tau_ff = float(np.clip(tau_ff, -tau_ff_limit, tau_ff_limit))
            max_step = max(0.0, tau_ff_slew) * dt
            tau_ff = float(np.clip(tau_ff, tau_ff_prev - max_step, tau_ff_prev + max_step))
            tau_ff_prev = tau_ff

            motor.send(q_ref=q_ref, dq_ref=dq_ref, kp=kp, kd=kd, tau_ff=tau_ff)

            logs["t"].append(float(t))
            logs["q"].append(float(q))
            logs["qd"].append(float(qd))
            logs["q_ref"].append(float(q_ref))
            logs["qd_ref"].append(float(dq_ref))
            logs["e_q"].append(float(q_ref - q))
            logs["e_qd"].append(float(dq_ref - qd))
            logs["kp"].append(float(kp))
            logs["kd"].append(float(kd))
            logs["tau_cmd_ff"].append(float(tau_ff))
            logs["tau_ff"].append(float(tau_ff))
            logs["tau_out"].append(float(tau_out))
            logs["temp"].append(float(temp))
            logs["merror"].append(float(merror))
            now2 = time.perf_counter()
            logs["loop_dt"].append(float(now2 - last_loop_t))
            last_loop_t = now2

            next_t += dt
            # Sleep with a small busy-wait tail to reduce oversleep jitter (important for fair comparison).
            while True:
                rem = next_t - time.perf_counter()
                if rem <= 0:
                    break
                if rem > 0.002:
                    time.sleep(rem - 0.001)
                else:
                    # busy wait for the final ~2ms
                    pass
    finally:
        signal.signal(signal.SIGINT, old)

    out = {k: np.asarray(v, dtype=np.float64) for k, v in logs.items()}

    # Derived signals from q/t for analysis: qd_from_q, qdd_from_q.
    t_arr = out["t"].reshape(-1)
    q_arr = out["q"].reshape(-1)
    if len(t_arr) >= 2:
        dt_arr = np.diff(t_arr)
        # Avoid division by 0 in rare cases; fallback to configured dt.
        dt_safe = np.where(dt_arr > 1e-9, dt_arr, float(dt))
        qd_from_q = np.zeros_like(q_arr)
        qd_from_q[1:] = (q_arr[1:] - q_arr[:-1]) / dt_safe
        qdd_from_q = np.zeros_like(q_arr)
        qdd_from_q[1:] = (qd_from_q[1:] - qd_from_q[:-1]) / dt_safe
    else:
        qd_from_q = np.zeros_like(q_arr)
        qdd_from_q = np.zeros_like(q_arr)

    out["qd_from_q"] = qd_from_q.astype(np.float64)
    out["qdd_from_q"] = qdd_from_q.astype(np.float64)

    np.savez(out_npz, **out)


def _metrics(path: str) -> dict[str, float]:
    ds = dict(np.load(path, allow_pickle=True))
    e = ds["e_q"].astype(np.float64).reshape(-1)
    loop_dt = ds.get("loop_dt", np.array([], dtype=np.float64)).astype(np.float64).reshape(-1)
    tau_out = ds["tau_out"].astype(np.float64).reshape(-1)
    tau_ff = ds["tau_ff"].astype(np.float64).reshape(-1)
    return {
        "rmse_e_q": float(np.sqrt(np.mean(e**2))) if len(e) else float("nan"),
        "maxabs_e_q": float(np.max(np.abs(e))) if len(e) else float("nan"),
        "meanabs_tau_out": float(np.mean(np.abs(tau_out))) if len(tau_out) else float("nan"),
        "meanabs_tau_ff": float(np.mean(np.abs(tau_ff))) if len(tau_ff) else float("nan"),
        "loop_dt_median": float(np.median(loop_dt)) if len(loop_dt) else float("nan"),
        "loop_dt_p90": float(np.quantile(loop_dt, 0.9)) if len(loop_dt) else float("nan"),
    }


def _plot_compare(base_npz: str, ff_npz: str, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    b = dict(np.load(base_npz, allow_pickle=True))
    f = dict(np.load(ff_npz, allow_pickle=True))

    tb = b["t"].reshape(-1)
    tf = f["t"].reshape(-1)

    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.plot(tb, b["q_ref"], label="q_ref", color="k", alpha=0.5)
    plt.plot(tb, b["q"], label="q (baseline)", alpha=0.8)
    plt.plot(tf, f["q"], label="q (ff)", alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(tb, b["e_q"], label="e_q baseline", alpha=0.8)
    plt.plot(tf, f["e_q"], label="e_q ff", alpha=0.8)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(tb, b["tau_out"], label="tau_out baseline", alpha=0.8)
    plt.plot(tf, f["tau_out"], label="tau_out ff", alpha=0.8)
    plt.plot(tf, f["tau_ff"], label="tau_ff (ff)", color="r", alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend()

    ensure_dir(os.path.dirname(out_png) or ".")
    plt.tight_layout()
    plt.savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["baseline", "ff", "both"], default="both")
    ap.add_argument("--ff_type", choices=["friction", "torque_delta"], default="friction")
    ap.add_argument("--friction_model", default="runs/friction_model.pt")
    ap.add_argument("--friction_dataset", default="runs/friction_dataset.npz")
    ap.add_argument("--torque_delta_model", default="runs/torque_delta_model.pt")
    ap.add_argument("--torque_delta_dataset", default="runs/torque_delta_dataset.npz")
    ap.add_argument("--kp", type=float, default=10.0)
    ap.add_argument("--kd", type=float, default=0.2)
    ap.add_argument("--amp", type=float, default=0.5, help="sine amplitude (rad)")
    ap.add_argument("--freq", type=float, default=0.5, help="sine frequency (Hz)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=None, help="control dt (default: real.dt from config)")
    ap.add_argument("--rest_s", type=float, default=2.0, help="rest between trials (seconds)")
    ap.add_argument("--tau_ff_limit", type=float, default=0.3)
    ap.add_argument("--tau_ff_slew", type=float, default=50.0, help="Nm/s")
    ap.add_argument("--tau_ff_scale", type=float, default=1.0, help="scale applied to predicted feedforward torque")
    ap.add_argument("--ff_update_div", type=int, default=1, help="update feedforward every N control steps (hold in between)")
    ap.add_argument("--ff_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--speed_limit", type=float, default=None, help="rad/s (default: real.max_speed_rad_s)")
    ap.add_argument("--temp_limit_c", type=float, default=None)
    ap.add_argument("--abort_on_merror", action="store_true", default=True)
    ap.add_argument("--no_abort_on_merror", dest="abort_on_merror", action="store_false")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dt = float(args.dt) if args.dt is not None else float(get(cfg, "real.dt", required=False, default=0.01))
    speed_limit = float(args.speed_limit) if args.speed_limit is not None else float(get(cfg, "real.max_speed_rad_s", required=False, default=0.0))
    temp_limit_c = float(args.temp_limit_c) if args.temp_limit_c is not None else float(get(cfg, "real.temp_limit_c", required=False, default=80.0))

    # Set center position for sine at current q
    motor = UnitreeMotor(cfg)
    try:
        q_now, qd_now, _tau, _temp, _merror = motor.read()
        q0 = float(q_now)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = str(get(cfg, "paths.runs_dir"))
        ensure_dir(out_dir)

        friction = None
        torque_delta = None
        if args.mode in ("ff", "both"):
            # Use CPU by default; can be switched later if needed.
            if args.ff_type == "friction":
                if not (os.path.exists(args.friction_model) and os.path.exists(args.friction_dataset)):
                    raise FileNotFoundError(
                        f"missing friction model/dataset: {args.friction_model} / {args.friction_dataset}. "
                        "Train it first:\n"
                        "  PYTHONPATH=. python3 inverse_torque/prepare_friction.py\n"
                        "  PYTHONPATH=. python3 inverse_torque/train_friction.py\n"
                    )
            else:
                if not (os.path.exists(args.torque_delta_model) and os.path.exists(args.torque_delta_dataset)):
                    raise FileNotFoundError(
                        f"missing torque-delta model/dataset: {args.torque_delta_model} / {args.torque_delta_dataset}. "
                        "Train it first:\n"
                        "  PYTHONPATH=. python3 inverse_torque/prepare.py\n"
                        "  PYTHONPATH=. python3 inverse_torque/train.py\n"
                    )
            # Reduce scheduling jitter from PyTorch thread pools.
            try:
                import torch

                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            if args.ff_type == "friction":
                friction = FrictionPredictor(args.friction_model, dataset_npz=args.friction_dataset, device=str(args.ff_device))
            else:
                torque_delta = TorqueDeltaPredictor(
                    args.torque_delta_model, dataset_npz=args.torque_delta_dataset, device=str(args.ff_device)
                )

        results: list[TrialResult] = []
        if args.mode in ("baseline", "both"):
            p = os.path.join(out_dir, f"ff_demo_baseline_{stamp}.npz")
            print("[demo] running baseline...")
            _run_trial(
                cfg,
                mode="baseline",
                motor=motor,
                friction=None,
                torque_delta=None,
                out_npz=p,
                duration_s=float(args.duration),
                dt=dt,
                q0=q0,
                amp=float(args.amp),
                freq_hz=float(args.freq),
                kp=float(args.kp),
                kd=float(args.kd),
                tau_ff_limit=0.0,
                tau_ff_slew=0.0,
                tau_ff_scale=0.0,
                ff_update_div=1,
                speed_limit=speed_limit,
                temp_limit_c=temp_limit_c,
                abort_on_merror=bool(args.abort_on_merror),
            )
            results.append(TrialResult(mode="baseline", path=p))
            time.sleep(max(0.0, float(args.rest_s)))

        if args.mode in ("ff", "both"):
            p = os.path.join(out_dir, f"ff_demo_ff_{stamp}.npz")
            print("[demo] running feedforward (friction prediction)...")
            _run_trial(
                cfg,
                mode="ff",
                motor=motor,
                friction=friction,
                torque_delta=torque_delta,
                out_npz=p,
                duration_s=float(args.duration),
                dt=dt,
                q0=q0,
                amp=float(args.amp),
                freq_hz=float(args.freq),
                kp=float(args.kp),
                kd=float(args.kd),
                tau_ff_limit=float(args.tau_ff_limit),
                tau_ff_slew=float(args.tau_ff_slew),
                tau_ff_scale=float(args.tau_ff_scale),
                ff_update_div=int(max(1, args.ff_update_div)),
                speed_limit=speed_limit,
                temp_limit_c=temp_limit_c,
                abort_on_merror=bool(args.abort_on_merror),
            )
            results.append(TrialResult(mode="ff", path=p))

        # Report
        if len(results) >= 1:
            out_md = os.path.join(out_dir, f"ff_demo_report_{stamp}.md")
            lines = []
            lines.append("# Feedforward Torque Compensation Demo")
            lines.append("")
            lines.append(f"amp(rad)={args.amp:g}, freq(Hz)={args.freq:g}, duration(s)={args.duration:g}, dt(s)={dt:g}")
            lines.append(f"kp={args.kp:g}, kd={args.kd:g}, tau_ff_limit={args.tau_ff_limit:g}, tau_ff_slew={args.tau_ff_slew:g}")
            lines.append(f"tau_ff_scale={args.tau_ff_scale:g}")
            lines.append(f"ff_update_div={int(max(1,args.ff_update_div))}, ff_device={args.ff_device}")
            lines.append(f"ff_type={args.ff_type}")
            if args.ff_type == "friction":
                lines.append(f"friction_model={args.friction_model}, friction_dataset={args.friction_dataset}")
            else:
                lines.append(f"torque_delta_model={args.torque_delta_model}, torque_delta_dataset={args.torque_delta_dataset}")
            lines.append("")
            for r in results:
                m = _metrics(r.path)
                lines.append(f"## {r.mode}")
                lines.append(f"- log: {os.path.basename(r.path)}")
                lines.append(f"- rmse_e_q(rad): {m['rmse_e_q']:.6f}")
                lines.append(f"- maxabs_e_q(rad): {m['maxabs_e_q']:.6f}")
                lines.append(f"- meanabs_tau_out(Nm): {m['meanabs_tau_out']:.6f}")
                lines.append(f"- meanabs_tau_ff(Nm): {m['meanabs_tau_ff']:.6f}")
                lines.append(f"- loop_dt_median(s): {m['loop_dt_median']:.6f}")
                lines.append(f"- loop_dt_p90(s): {m['loop_dt_p90']:.6f}")
                lines.append("")

            if any(r.mode == "baseline" for r in results) and any(r.mode == "ff" for r in results):
                base_path = [r.path for r in results if r.mode == "baseline"][0]
                ff_path = [r.path for r in results if r.mode == "ff"][0]
                out_png = os.path.join(out_dir, f"ff_demo_compare_{stamp}.png")
                _plot_compare(base_path, ff_path, out_png=out_png)
                lines.append(f"compare plot: {os.path.basename(out_png)}")

            with open(out_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print("[demo] saved report:", out_md)
    finally:
        motor.shutdown()


if __name__ == "__main__":
    main()
