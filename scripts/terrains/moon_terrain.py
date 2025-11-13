import numpy as np
from noise import pnoise2
from isaaclab.utils.mesh import height_field_to_trimesh


def generate_moon_terrain(difficulty: float, cfg):
    """Procedural cratered moon terrain for IsaacLab."""
    length, width = cfg.size
    resolution = 256
    hf = np.zeros((resolution, resolution), dtype=np.float32)

    # Difficulty -> number of craters, roughness
    num_craters = int(np.interp(difficulty, [0, 1],
                                [cfg.min_craters, cfg.max_craters]))
    roughness = np.interp(difficulty, [0, 1],
                          [cfg.min_roughness, cfg.max_roughness])

    # Precompute meshgrid
    xx, yy = np.meshgrid(np.arange(resolution), np.arange(resolution))

    # Add craters
    for _ in range(num_craters):
        cx = np.random.uniform(0, resolution)
        cy = np.random.uniform(0, resolution)

        radius = np.random.uniform(cfg.min_radius, cfg.max_radius)
        depth  = np.random.uniform(cfg.min_depth, cfg.max_depth)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        crater_profile = np.exp(-(dist / radius) ** cfg.crater_smoothness)

        hf -= depth * crater_profile

    # Add perlin roughness
    for i in range(resolution):
        for j in range(resolution):
            hf[i, j] += roughness * pnoise2(
                i / 40.0, j / 40.0,
                octaves=3, persistence=0.4, lacunarity=2.0,
                repeatx=1024, repeaty=1024, base=42
            )

    mesh = height_field_to_trimesh(hf, size=(length, width))
    origin = np.array([0.0, 0.0, float(np.min(hf))], dtype=np.float32)

    return [mesh], origin


# Register the terrain function
from .moon_terrain_cfg import MoonTerrainCfg
MoonTerrainCfg.function = generate_moon_terrain
