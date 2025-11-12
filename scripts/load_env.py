# ================================================================
# Rocket URDF Test Script (Isaac Lab)
# Loads a lunar rocket URDF with 2-DOF gimbal joints and runs sim.
# ================================================================

import argparse
import numpy as np
import torch
import os

from isaaclab.app import AppLauncher

# ------------------------------------------------
# 1. CLI args
# ------------------------------------------------
parser = argparse.ArgumentParser(description="Run the custom lunar rocket URDF in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ------------------------------------------------
# 2. Launch Isaac Lab app
# ------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab core imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import UrdfFileCfg

# ------------------------------------------------
# 3. Rocket + environment configuration
# ------------------------------------------------

# Path to your rocket URDF and moon USD
ROCKET_URDF_PATH = "space_robot/assets/rocket/rocket.urdf"
MOON_USD_PATH = "space_robot/assets/moon/moon_terrain.usd"

# --- Rocket configuration ---
ROCKET_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=None,  # we’ll import from URDF, not an existing USD
        urdf_path=ROCKET_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        density=2000.0,
        scale=1.0,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
    actuators={
        "gimbal_actuators": ImplicitActuatorCfg(
            joint_names_expr=["pitch_joint", "yaw_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
            stiffness=2000.0,
            damping=20.0,
        ),
    },
)

# --- Scene configuration ---
class RocketSceneCfg(InteractiveSceneCfg):
    """Custom scene with rocket and moon surface."""
    ground = AssetBaseCfg(prim_path="/World/MoonSurface", spawn=sim_utils.UsdFileCfg(usd_path=MOON_USD_PATH))
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )
    rocket = ROCKET_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Rocket")

# ------------------------------------------------
# 4. Simulation + control loop
# ------------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    print("[INFO] Rocket DOFs:", scene["Rocket"].data.joint_names)

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = scene["Rocket"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["Rocket"].write_root_pose_to_sim(root_state[:, :7])
            scene["Rocket"].write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = scene["Rocket"].data.default_joint_pos.clone()
            joint_vel = scene["Rocket"].data.default_joint_vel.clone()
            scene["Rocket"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO] Resetting rocket state...")

        # Simple gimbal oscillation
        pitch_angle = 0.1 * np.sin(2 * np.pi * 0.25 * sim_time)
        yaw_angle = 0.1 * np.cos(2 * np.pi * 0.25 * sim_time)
        gimbal_action = torch.tensor([[pitch_angle, yaw_angle]], dtype=torch.float32)
        scene["Rocket"].set_joint_position_target(gimbal_action)

        # Step sim
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

# ------------------------------------------------
# 5. Main
# ------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([8.0, 0.0, 4.0], [0.0, 0.0, 1.0])

    scene_cfg = RocketSceneCfg(args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Rocket simulation setup complete.")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
