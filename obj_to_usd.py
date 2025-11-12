# ================================================================
# OBJ → USD Converter for Moon Terrain
# Author: Jerry Cheng | NYU Space Robotics | Fall 2025
# ================================================================

import argparse
import os
from isaaclab.app import AppLauncher

# ------------------------------------------------------------
# 1. Launch Isaac Lab app
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert Moon OBJ terrain to USD inside Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# 2. Now Omniverse context is active, import the converter
# ------------------------------------------------------------
from isaaclab.utils.assets import MeshConverterCfg, import_from_mesh

# ------------------------------------------------------------
# 3. Paths
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "moon"))
OBJ_PATH = os.path.join(PROJECT_ROOT, "moon_terrain.obj")
USD_OUTPUT_DIR = PROJECT_ROOT

print(f"[INFO] Converting OBJ to USD\n  OBJ: {OBJ_PATH}")

# ------------------------------------------------------------
# 4. Create config
# ------------------------------------------------------------
cfg = MeshConverterCfg(
    mesh_path=OBJ_PATH,
    usd_dir=USD_OUTPUT_DIR,
    scale=(1.0, 1.0, 1.0),
    density=1500.0,
    collision=True,
    visual=True,
    merge_meshes=True,
)

# ------------------------------------------------------------
# 5. Convert
# ------------------------------------------------------------
usd_path = import_from_mesh(cfg)
print(f"[INFO] ✅ Conversion complete → {usd_path}")

# ------------------------------------------------------------
# 6. Close the app cleanly
# ------------------------------------------------------------
simulation_app.close()
