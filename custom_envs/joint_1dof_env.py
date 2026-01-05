import os
import torch
import numpy as np

# If Warp cache directory was previously created by sudo, user runs may fail with EACCES.
# Use a repo-local cache by default.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WARP_CACHE = os.path.join(_REPO_ROOT, "runs", "warp_cache")
os.makedirs(_WARP_CACHE, exist_ok=True)
os.environ.setdefault("WARP_CACHE_PATH", _WARP_CACHE)

import warp as wp
import warp.sim

# local
from project_config import ConfigError
import xml.etree.ElementTree as ET

# 在文件头部加入
try:
    from tqdm import tqdm
except ImportError:
    # 如果没装 tqdm，做一个简单的伪装
    def tqdm(iterable, desc=""):
        return iterable
    
# ==============================================================================
# 1. 极其轻量的 Warp 环境基类 (Warp 1.8.0 Control Object Fix)
# ==============================================================================
class MinimalWarpEnv:
    def __init__(self, cfg, device):
        self.device = device
        if not isinstance(cfg, dict):
            raise ConfigError("env cfg must be a dict (loaded from config.json)")
        self.cfg = cfg
        self.frame_dt = float(cfg["frame_dt"])
        self.sim_substeps = int(cfg["sim_substeps"])
        self.num_envs = int(cfg["num_envs"])
        
        with wp.ScopedDevice(self.device):
            self.builder = wp.sim.ModelBuilder()
            self.load_assets()
            
            try:
                self.model = self.builder.finalize(device=self.device)
            except TypeError:
                self.model = self.builder.finalize()
            
            self.model.ground = False 
            self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
            
            # 1. 初始化 State (位置/速度)
            self.state = self.model.state()
            
            # 2. 【关键修复】初始化 Control (力矩/动作)
            # Warp 1.8.0 将控制信号剥离到了 Control 对象中
            self.control = self.model.control()
            
            # 计算 DOF
            total_dof = self.state.joint_q.shape[0]
            if self.num_envs > 0:
                self.dof_per_env = total_dof // self.num_envs
            else:
                self.dof_per_env = 0
            
            print(f"[WarpEnv] Built {self.num_envs} envs. Total DOF: {total_dof} ({self.dof_per_env}/env)")

            # FK
            wp.sim.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)
            wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)

    def load_assets(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        with wp.ScopedDevice(self.device):
            dt = self.frame_dt / self.sim_substeps
            
            # 1. 准备力矩数据
            total_dof = self.state.joint_q.shape[0]
            full_torque = torch.zeros(total_dof, device=self.device)
            
            if actions.shape[1] == 1 and self.dof_per_env > 0:
                indices = torch.arange(0, total_dof, step=self.dof_per_env, device=self.device, dtype=torch.long)
                target_indices = indices + 0 
                full_torque.index_add_(0, target_indices, actions.flatten())
            else:
                full_torque = actions.flatten()

            # Warp's MJCF parser currently does not apply MuJoCo joint `damping`/`frictionloss` reliably.
            # Apply a minimal dissipative model explicitly in torque space:
            #   tau_eff = tau_cmd - b*qd - tau_c*sign(qd)
            mj = self.cfg.get("mjcf_override", {}) if isinstance(getattr(self, "cfg", {}), dict) else {}
            b = float(mj.get("damping", 0.0))
            tau_c = float(mj.get("frictionloss", 0.0))
            if b != 0.0 or tau_c != 0.0:
                qd_flat = wp.to_torch(self.state.joint_qd)
                if tau_c != 0.0:
                    eps = 1.0e-3
                    sgn = qd_flat / (torch.abs(qd_flat) + eps)
                else:
                    sgn = 0.0
                full_torque = full_torque - b * qd_flat - tau_c * sgn

            wp_act = wp.from_torch(full_torque)

            # 2. 积分循环
            for _ in range(self.sim_substeps):
                self.state.clear_forces()
                
                # 【关键修复】将力矩赋值给 Control 对象
                # Control 对象里依然保留了 joint_act 这个属性名
                self.control.joint_act.assign(wp_act)
                
                # 【关键修复】将 control 对象传给 simulate
                self.integrator.simulate(
                    self.model, 
                    self.state, 
                    self.state, 
                    dt, 
                    control=self.control  # <--- 这里！
                )
            
            # 3. 返回状态
            q_flat = wp.to_torch(self.state.joint_q)
            qd_flat = wp.to_torch(self.state.joint_qd)
            
            q = q_flat.view(self.num_envs, -1)
            qd = qd_flat.view(self.num_envs, -1)
            
            return torch.cat([q, qd], dim=-1)

# ==============================================================================
# 2. 具体的 1-DOF 关节环境
# ==============================================================================
class Joint1DofEnv(MinimalWarpEnv):
    # ... (其他方法不变)
    def load_assets(self):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_file_dir, ".."))
        asset_path_cfg = self.cfg.get("asset_mjcf", "assets/joint_1dof.xml") if hasattr(self, "cfg") else "assets/joint_1dof.xml"
        if os.path.isabs(asset_path_cfg):
            self.asset_path = asset_path_cfg
        else:
            self.asset_path = os.path.abspath(os.path.join(repo_root, asset_path_cfg))

        if not os.path.exists(self.asset_path):
            raise FileNotFoundError(f"XML file not found at: {self.asset_path}")

        # Runtime patch MJCF default joint params from cfg (so frictionloss can be updated without relying
        # on Warp parse_mjcf kwargs support).
        mjcf_override = self.cfg.get("mjcf_override", {}) if hasattr(self, "cfg") else {}
        if any(k in mjcf_override for k in ["damping", "armature", "frictionloss"]):
            runs_dir = self.cfg.get("runs_dir", os.path.join(repo_root, "runs"))
            os.makedirs(runs_dir, exist_ok=True)
            patched = os.path.join(runs_dir, "mjcf_patched.xml")
            self._write_patched_mjcf(self.asset_path, patched, mjcf_override)
            self.asset_path = patched

        print(f"[Joint1DofEnv] Parsing MJCF {self.num_envs} times...")

        stiffness = float(mjcf_override.get("stiffness", 0.0))
        damping = float(mjcf_override.get("damping", 0.05))
        armature = float(mjcf_override.get("armature", 0.005))
        contact_ke = float(mjcf_override.get("contact_ke", 1.0e4))
        contact_kd = float(mjcf_override.get("contact_kd", 1.0e2))
        
        # 【关键修改】添加进度条，让你看到它在动
        for i in tqdm(range(self.num_envs), desc="Building Envs"):
            try:
                wp.sim.parse_mjcf(
                    self.asset_path, self.builder,
                    stiffness=stiffness, damping=damping, armature=armature,
                    contact_ke=contact_ke, contact_kd=contact_kd
                )
            except AttributeError:
                wp.sim.load_mjcf(
                    self.asset_path, self.builder,
                    stiffness=stiffness, damping=damping, armature=armature,
                    contact_ke=contact_ke, contact_kd=contact_kd
                )

    def reset(self):
        with wp.ScopedDevice(self.device):
            total_dof = self.state.joint_q.shape[0]
            
            q_np = np.zeros(total_dof, dtype=np.float32)
            qd_np = np.zeros(total_dof, dtype=np.float32)
            
            if self.dof_per_env > 0:
                rand_pos = np.random.uniform(-1.5, 1.5, size=self.num_envs)
                for env_i in range(self.num_envs):
                    idx = env_i * self.dof_per_env
                    q_np[idx] = rand_pos[env_i]
            
            q_wp = wp.from_numpy(q_np, device=self.device, dtype=wp.float32)
            qd_wp = wp.from_numpy(qd_np, device=self.device, dtype=wp.float32)
            
            self.state.joint_q.assign(q_wp)
            self.state.joint_qd.assign(qd_wp)
            
            # 清零 Control 力矩
            self.control.joint_act.zero_()
            
            q_flat = wp.to_torch(self.state.joint_q)
            qd_flat = wp.to_torch(self.state.joint_qd)
            
            q = q_flat.view(self.num_envs, -1)
            qd = qd_flat.view(self.num_envs, -1)
            
            return torch.cat([q, qd], dim=-1)

    def __init__(self, cfg, device="cuda"):
        # Keep a copy for asset path override.
        self.cfg = cfg if isinstance(cfg, dict) else {}
        super().__init__(cfg, device=device)

    @staticmethod
    def _write_patched_mjcf(src_path: str, dst_path: str, mjcf_override: dict) -> None:
        tree = ET.parse(src_path)
        root = tree.getroot()
        default = root.find("default")
        if default is None:
            default = ET.SubElement(root, "default")
        joint = default.find("joint")
        if joint is None:
            joint = ET.SubElement(default, "joint")

        if "damping" in mjcf_override:
            joint.set("damping", str(float(mjcf_override["damping"])))
        if "armature" in mjcf_override:
            joint.set("armature", str(float(mjcf_override["armature"])))
        if "frictionloss" in mjcf_override:
            joint.set("frictionloss", str(float(mjcf_override["frictionloss"])))

        tree.write(dst_path, encoding="utf-8", xml_declaration=False)
