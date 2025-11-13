from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from .rocket_asset_cfg import ROCKET_CFG


class RocketSceneCfg(InteractiveSceneCfg):
    """Scene that spawns the rocket."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    Rocket = ROCKET_CFG.replace(prim_path="{ENV_REGEX_NS}/Rocket")
