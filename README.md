# TINTIN: Thrust-Integrated Neural Touchdown

This project builds the **Tintin lunar rocket landing simulator** in **Isaac Lab**, following the modeling described in *Space_Robot_Course_Project_1-6.pdf*.  
It implements a 6-DoF variable-mass rocket with a gimbaled engine for RL-based lunar touchdown.

---

## Directory Overview

- **agents/** – RL or PID control agents for training and evaluation.  
- **assets/** – 3D models of the rocket and terrain.  
  - `rocket/` – Rocket meshes (.obj, .usd, .urdf).  
  - `moon/` – Lunar surface meshes (.obj, .usd).  
- **cfg/** – Configuration files.  
  - `lunar_env_cfg.yaml` – Simulation setup (Moon gravity, contact params).  
  - `rocket_asset_cfg.yaml` – Rocket parameters (mass, inertia, engine offset).  
  - `rl_train_cfg.yaml` – RL training configuration and reward weights.  
- **envs/** – Custom Isaac Lab environment definitions.  
- **logs/** – Training logs and rollout data.  
- **scripts/** – Main scripts.  
  - `load_env.py` – Loads the lunar landing environment in Isaac Lab.  
  - `obj_to_usd.py` – Converts OBJ models to USD format for Isaac Lab.  
  - `__init__.py` – Marks this directory as a package.

---

## Simulation Summary

- **States:** position, velocity, quaternion, angular velocity, mass  
- **Controls:** thrust `T`, pitch `θp`, yaw `θy`  
- **Physics:** lunar gravity (1.62 m/s²), variable mass, gimbaled thrust at nozzle  
- **Goal:** achieve a soft, accurate landing with minimal propellant

---

## Run

```bash
# Launch the Isaac Lab environment
python scripts/load_env.py

# Convert assets to USD
python scripts/obj_to_usd.py
