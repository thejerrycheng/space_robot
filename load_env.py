# ================================================================
# Lunar Rocket Environment (Isaac Lab 2025)
# Author: Jerry Cheng | Space Robotics | NYU Fall 2025
# ================================================================
# ================================================================
# Isaac Sim Environment Loader
# Author: Jerry Cheng | NYU Space Robotics | Fall 2025
# ================================================================

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext

#  ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")

ROCKET_URDF_PATH = os.path.join(ASSET_ROOT, "rocket", "rocket.urdf")
MOON_USD_PATH = os.path.join(ASSET_ROOT, "moon", "moon_terrain.usd")

print("The path is: ", MOON_USD_PATH)

def design_scene():
    """Designs the scene by spawning the Moon terrain and lighting."""

    # --- Remove default ground plane ---
    # (Don't create GroundPlaneCfg)

    # --- Spawn Moon terrain instead ---
    cfg_moon_surface = sim_utils.UsdFileCfg(
        usd_path=MOON_USD_PATH,
        scale=(1.0, 1.0, 1.0),
        visible=True,
    )
    cfg_moon_surface.func("/World/MoonSurface", cfg_moon_surface)

    print(f"[INFO] Loaded Moon surface from: {MOON_USD_PATH}")

    # --- Add lighting ---
    cfg_dome_light = sim_utils.DomeLightCfg(
    intensity=4000.0,
    color=(1.0, 1.0, 1.0),
    )
    cfg_dome_light.func("/World/DomeLight", cfg_dome_light)

    cfg_distant_light = sim_utils.DistantLightCfg(
        intensity=2000.0,
        color=(1.0, 1.0, 0.9),
    )
    cfg_distant_light.func("/World/SunLight", cfg_distant_light, translation=(0, 0, 50))


    print("[INFO] Moon environment and lighting initialized.")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    #design the scene 
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready! 
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
