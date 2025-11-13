from dataclasses import dataclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

@dataclass
class MoonTerrainCfg(SubTerrainBaseCfg):
    min_craters: int = 3
    max_craters: int = 15
    min_radius: float = 0.3
    max_radius: float = 2.0
    min_depth: float = 0.05
    max_depth: float = 0.6

    min_roughness: float = 0.0
    max_roughness: float = 0.3

    crater_smoothness: float = 1.5

    # assigned later in moon_terrain.py
    function = None
