import os
import torch
import numpy as np
import warp as wp
import warp.sim

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
        self.frame_dt = cfg.frame_dt
        self.sim_substeps = cfg.sim_substeps
        self.num_envs = cfg.num_envs
        
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
                wp_act = wp.from_torch(full_torque)
            else:
                wp_act = wp.from_torch(actions.flatten())

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
        self.asset_path = os.path.abspath(os.path.join(current_file_dir, "../assets/joint_1dof.xml"))
        
        if not os.path.exists(self.asset_path):
            raise FileNotFoundError(f"XML file not found at: {self.asset_path}")

        print(f"[Joint1DofEnv] Parsing MJCF {self.num_envs} times...")
        
        # 【关键修改】添加进度条，让你看到它在动
        for i in tqdm(range(self.num_envs), desc="Building Envs"):
            try:
                wp.sim.parse_mjcf(
                    self.asset_path, self.builder,
                    stiffness=0.0, damping=0.05, armature=0.005,
                    contact_ke=1.0e4, contact_kd=1.0e2
                )
            except AttributeError:
                wp.sim.load_mjcf(
                    self.asset_path, self.builder,
                    stiffness=0.0, damping=0.05, armature=0.005,
                    contact_ke=1.0e4, contact_kd=1.0e2
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