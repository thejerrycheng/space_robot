# Copyright ...
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------
# Launch Isaac Sim
# ------------------------------------------------------------
import argparse
import math
import time
 
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rocket with event-based carb keyboard thrust + flame viz.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--dt", type=float, default=0.01, help="Timestep [s]")
parser.add_argument("--thrust_max", type=float, default=400.0, help="Max thrust (arbitrary units)")
parser.add_argument("--thrust_step", type=float, default=10.0, help="Thrust increment per key press")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# Imports AFTER app launch
# ------------------------------------------------------------
import carb
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf

# ------------------------------------------------------------
# Simple USD helpers
# ------------------------------------------------------------

def ensure_translate(stage, path):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    t_op = next((op for op in ops if op.GetOpName() == "xformOp:translate"), None)
    if t_op is None:
        t_op = xf.AddTranslateOp(opSuffix="")
        t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return t_op

def set_translate(stage, path, p):
    ensure_translate(stage, path).Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

# ------------------------------------------------------------
# Engine / Flame visualization
# ------------------------------------------------------------

def make_rocket_engine(stage, parent_path,
                       base_height=0.0,
                       nozzle_h=0.3,
                       nozzle_r=0.08,
                       flame_len_init=0.1,
                       flame_r_init=0.12):
    """
    Creates /World/Rocket/Engine with:
      - Nozzle cone pointing DOWN (-Z in world)
      - FlameMain / FlameMid / FlameFar cones also pointing down.
    We do this by rotating Engine 180° around X so its local +Z = world -Z.
    """
    engine_path = f"{parent_path}/Engine"
    engine = UsdGeom.Xform.Define(stage, engine_path)
    xf = UsdGeom.Xformable(engine.GetPrim())

    # Rotate 180 degrees around X so +Z(local) points in -Z(world)
    rot_op = xf.AddRotateXYZOp(opSuffix="")
    rot_op.Set(Gf.Vec3f(180.0, 0.0, 0.0))

    # Place engine root roughly at bottom of rocket body
    # (your body center is around z ~ 2; legs go down to ~1.6)
    # We'll anchor engine at z = 1.5 in world, plus any offset you want.
    trans_op = xf.AddTranslateOp()
    trans_op.Set(Gf.Vec3d(0.0, 0.0, 1.5 + base_height))

    # Nozzle (fixed cone)
    noz = UsdGeom.Cone.Define(stage, f"{engine_path}/Nozzle")
    noz.GetHeightAttr().Set(float(nozzle_h))
    noz.GetRadiusAttr().Set(float(nozzle_r))
    UsdGeom.Gprim(noz.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(0.82, 0.84, 0.88)])
    # Along +Z of engine (which is world -Z)
    set_translate(stage, f"{engine_path}/Nozzle", (0.0, 0.0, nozzle_h * 0.5))

    # Flames
    for name, color in [
        ("FlameMain", Gf.Vec3f(1.00, 0.55, 0.10)),
        ("FlameMid",  Gf.Vec3f(1.00, 0.75, 0.20)),
        ("FlameFar",  Gf.Vec3f(1.00, 0.90, 0.35)),
    ]:
        cone = UsdGeom.Cone.Define(stage, f"{engine_path}/{name}")
        cone.GetHeightAttr().Set(float(flame_len_init))
        cone.GetRadiusAttr().Set(float(flame_r_init))
        UsdGeom.Gprim(cone.GetPrim()).CreateDisplayColorAttr().Set([color])

    # Initial positioning for the flames
    update_rocket_flame(
        stage,
        parent_path=parent_path,
        thrust=0.0,
        thrust_max=args_cli.thrust_max,
        nozzle_h=nozzle_h,
        flame_len_min=flame_len_init,
        flame_len_max=2.0,
        flame_r=flame_r_init,
        spread_max=2.25,
        scale_nozzle=False,
    )

def update_rocket_flame(stage, parent_path, thrust, thrust_max,
                        nozzle_h=0.3,
                        flame_len_min=0.05,
                        flame_len_max=2.0,
                        flame_r=0.12,
                        spread_max=2.25,
                        scale_nozzle=False):
    """
    Scale flame length and radius with thrust; position as consecutive segments
    after the nozzle, all pointing DOWN (-Z in world).
    """
    engine_path = f"{parent_path}/Engine"

    if thrust_max <= 0.0:
        k = 0.0
    else:
        k = max(0.0, min(1.0, thrust / thrust_max))

    # Lengths
    L_core = flame_len_min + (flame_len_max - flame_len_min) * k
    L_mid  = max(L_core * 0.75, flame_len_min * 0.8)
    L_far  = max(L_core * 0.55, flame_len_min * 0.6)

    # Radii (spread)
    R_core = flame_r * (1.0 + (spread_max - 1.0) * k)
    R_mid  = flame_r * (1.0 + 0.8 * (spread_max - 1.0) * k)
    R_far  = flame_r * (1.0 + 0.6 * (spread_max - 1.0) * k)

    # Apply sizes
    for name, H, R in [
        ("FlameMain", L_core, R_core),
        ("FlameMid",  L_mid,  R_mid),
        ("FlameFar",  L_far,  R_far),
    ]:
        cone = UsdGeom.Cone.Get(stage, f"{engine_path}/{name}")
        if cone:
            cone.GetHeightAttr().Set(float(H))
            cone.GetRadiusAttr().Set(float(R))

    # Position flames along +Z in engine space (which is -Z in world)
    z0 = nozzle_h
    set_translate(stage, f"{engine_path}/FlameMain", (0.0, 0.0, z0 + 0.5 * L_core))
    set_translate(stage, f"{engine_path}/FlameMid",  (0.0, 0.0, z0 + L_core + 0.5 * L_mid + 0.02))
    set_translate(stage, f"{engine_path}/FlameFar",  (0.0, 0.0, z0 + L_core + L_mid + 0.5 * L_far + 0.04))

    # Optional nozzle “puff”
    if scale_nozzle:
        noz = UsdGeom.Cone.Get(stage, f"{engine_path}/Nozzle")
        if noz:
            base_r = noz.GetRadiusAttr().Get()
            noz.GetRadiusAttr().Set(float(base_r * (1.0 + 0.15 * k)))

# ------------------------------------------------------------
# Rocket control data container
# ------------------------------------------------------------

class RocketControl:
    def __init__(self):
        self.thrust = 0.0  # upwards (+Z) effective thrust

# ------------------------------------------------------------
# Global keyboard event hook (carb.input)
# ------------------------------------------------------------

_key_sub = None

def install_keyboard_listener(rocket: RocketControl, thrust_step: float, thrust_max: float):
    """
    Subscribe to global keyboard events.
    UP   -> increase thrust
    DOWN -> decrease thrust
    """
    input_iface = carb.input.acquire_input_interface()

    def _on_key(event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.UP:
                rocket.thrust = min(thrust_max, rocket.thrust + thrust_step)
            elif event.input == carb.input.KeyboardInput.DOWN:
                rocket.thrust = max(0.0, rocket.thrust - thrust_step)

    global _key_sub
    if _key_sub is None:
        _key_sub = input_iface.subscribe_to_keyboard_events(None, _on_key)

# ------------------------------------------------------------
# Scene Construction (rocket geometry)
# ------------------------------------------------------------

def design_scene(sim):
    """ 
    Builds the rocket as USD geometry under /World/Rocket.
    NOTE: We do NOT give rigid_props here; the rocket is moved kinematically
    by our own integrator, not PhysX forces.
    """
    stage = sim.stage

    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Root Xform for rocket
    prim_utils.create_prim("/World/Rocket", "Xform")

    # --------------------------------------------------------
    # Rocket Nose  (Rigid Body + Collider + Visual)
    # --------------------------------------------------------
    cfg_rocket_nose = sim_utils.ConeCfg(
        radius=0.1,
        height=0.3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # <-- enable rigid body
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    )
    cfg_rocket_nose.func(
        "/World/Rocket/Nose",
        cfg_rocket_nose,
        translation=(0.0, 0.0, 2.45),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    # --------------------------------------------------------
    # Rocket Body (Rigid Body + Collider + Visual)
    # --------------------------------------------------------
    cfg_body = sim_utils.CylinderCfg(
        radius=0.1,
        height=0.8,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # <-- enable rigid body
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
    )
    cfg_body.func(
        "/World/Rocket/Body",
        cfg_body,
        translation=(0.0, 0.0, 2.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    # --------------------------------------------------------
    # Thruster housing (Rigid Body + Collider + Visual)
    # --------------------------------------------------------
    cfg_thruster_geom = sim_utils.CylinderCfg(
        radius=0.05,
        height=0.15,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # <-- enable rigid body
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
    )
    cfg_thruster_geom.func(
        "/World/Rocket/Thruster",
        cfg_thruster_geom,
        translation=(0.0, 0.0, 1.55),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    # --------------------------------------------------------
    # Rocket Legs
    # --------------------------------------------------------
    leg_positions = [
        (0.12, 0.0),
        (-0.06, 0.1),
        (-0.06, -0.1),
    ]

    for i, (lx, ly) in enumerate(leg_positions):

        # horizontal section
        cfg_leg_h = sim_utils.CylinderCfg(
            radius=0.03,
            height=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )

        angle = math.atan2(ly, lx)
        qz = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))

        cfg_leg_h.func(
            f"/World/Rocket/Leg{i}_H",
            cfg_leg_h,
            translation=(lx / 2.0, ly / 2.0, 1.8),
            orientation=qz,
        )

        # vertical downward section
        cfg_leg_v = sim_utils.CylinderCfg(
            radius=0.03,
            height=0.25,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )

        cfg_leg_v.func(
            f"/World/Rocket/Leg{i}_V",
            cfg_leg_v,
            translation=(lx, ly, 1.6),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    # Create engine + flames under /World/Rocket
    make_rocket_engine(stage, parent_path="/World/Rocket")

# ------------------------------------------------------------
# Main: kinematic rocket integration + flame viz
# ------------------------------------------------------------

def main():
    # Simulation context (we use it mainly for dt + stepping + camera)
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([4.0, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Build rocket + scene
    design_scene(sim)
    sim.reset()

    print("[INFO] Rocket with event-based carb keyboard thrust + flame viz ready.")
    print("[INFO] Controls: UP = increase thrust, DOWN = decrease thrust.")

    # Rocket kinematic state (world)
    rocket = RocketControl()
    rocket_mass = 50.0  # arbitrary mass units for acceleration scaling
    pos_z = 0.0         # translation on /World/Rocket Xform
    vel_z = 0.00
    dt = args_cli.dt

    # Set initial rocket transform (root)
    set_translate(sim.stage, "/World/Rocket", (0.0, 0.0, pos_z))

    # Install keyboard listener for thrust
    install_keyboard_listener(
        rocket=rocket,
        thrust_step=args_cli.thrust_step,
        thrust_max=args_cli.thrust_max,
    )

    last_log = 0.0

    while simulation_app.is_running():
        # Simple 1D integration: thrust -> accel -> vel -> pos
        # Thrust acts upwards (+Z). No gravity here; you can subtract g if desired.
        accel_z = rocket.thrust / rocket_mass
        vel_z += accel_z * dt
        pos_z += vel_z * dt

        # Optional: don't let rocket go below ground visually
        if pos_z < 0.0:
            pos_z = 0.0
            vel_z = 0.0

        # Update rocket root transform
        set_translate(sim.stage, "/World/Rocket", (0.0, 0.0, pos_z))

        # Update flame visualization based on thrust
        update_rocket_flame(
            sim.stage,
            parent_path="/World/Rocket",
            thrust=rocket.thrust,
            thrust_max=args_cli.thrust_max,
            nozzle_h=0.3,
            flame_len_min=0.05,
            flame_len_max=2.0,
            flame_r=0.12,
            spread_max=2.25,
            scale_nozzle=False,
        )

        # Occasionally log thrust to terminal
        now = time.time()
        if now - last_log > 0.5:
            level = rocket.thrust / args_cli.thrust_max if args_cli.thrust_max > 0 else 0.0
            print(f"[THRUST] {rocket.thrust:.1f} / {args_cli.thrust_max:.1f}  ({level:.0%})")
            last_log = now

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
