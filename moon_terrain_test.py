# moon_terrain_test.py
#
# Minimal Isaac Lab test using ONLY IsaacLab APIs.
# Visualizes your Moon Terrain in Isaac Sim.

import argparse
import os
import sys

# Add IsaacLab/scripts to PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Test Moon Terrain Generator")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IsaacLab imports
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.utils import configclass

# Import your custom terrain config
from space_robot.terrains.moon_terrain_cfg import MoonTerrainCfg


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class MoonSceneCfg(InteractiveSceneCfg):

    moon_terrain_generator = TerrainGeneratorCfg(
        num_rows=1,
        num_cols=1,
        size=(20.0, 20.0),
        difficulty_range=(0.0, 1.0),
        border_width=1.0,
        border_height=0.2,
        curriculum=False,
        sub_terrains={
            "moon": MoonTerrainCfg(
                proportion=1.0,
                flat_patch_sampling=None
            )
        },
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=moon_terrain_generator,
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
@configclass
class MoonEnvCfg(ManagerBasedEnvCfg):
    scene: MoonSceneCfg = MoonSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=1.0
    )

    def __post_init__(self):
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print("\n[INFO] Generating Moon Terrain...\n")

    env_cfg = MoonEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()

    print("[INFO] Simulation started. View the cratered Moon Terrain in Isaac Sim.\n")

    while simulation_app.is_running():
        env.step(None)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
