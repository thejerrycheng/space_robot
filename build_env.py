# Copyright ...
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math
# import keyboard

import carb
input = carb.input.acquire_input_interface()

if input.is_key_down(carb.input.KeyboardInput.W):
    thrust += 0.5

from isaaclab.app import AppLauncher
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from omni.isaac.core.physics_context import PhysicsContext

# ------------------------------------------------------------
# Launch Isaac Sim
# ------------------------------------------------------------

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ------------------------------------------------------------
# Scene Construction
# ------------------------------------------------------------

def design_scene():

    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Rocket", "Xform")

    # --------------------------------------------------------
    # Rocket Nose
    # --------------------------------------------------------
    cfg_rocket_nose = sim_utils.ConeCfg(
        radius=0.1,
        height=0.3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
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
    # Rocket Body
    # --------------------------------------------------------
    cfg_body = sim_utils.CylinderCfg(
        radius=0.1,
        height=0.8,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.2),
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
    # Thruster Nozzle
    # --------------------------------------------------------
    cfg_thruster = sim_utils.CylinderCfg(
        radius=0.05,
        height=0.15,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
    )
    cfg_thruster.func(
        "/World/Rocket/Thruster",
        cfg_thruster,
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )

        angle = math.atan2(ly, lx)
        qz = (math.cos(angle/2), 0.0, 0.0, math.sin(angle/2))

        cfg_leg_h.func(
            f"/World/Rocket/Leg{i}_H",
            cfg_leg_h,
            translation=(lx/2, ly/2, 1.8),
            orientation=qz,
        )

        # vertical downward section
        cfg_leg_v = sim_utils.CylinderCfg(
            radius=0.03,
            height=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )

        cfg_leg_v.func(
            f"/World/Rocket/Leg{i}_V",
            cfg_leg_v,
            translation=(lx, ly, 1.6),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    # --------------------------------------------------------
    # FIXED JOINTS (NO BREAKING APART)
    # --------------------------------------------------------
    sim_utils.JointCfg.fix(
        parent="/World/Rocket/Body",
        child="/World/Rocket/Nose",
    ).func("/World/Rocket/J_Nose")

    for i in range(3):
        sim_utils.JointCfg.fix(
            parent="/World/Rocket/Body",
            child=f"/World/Rocket/Leg{i}_H",
        ).func(f"/World/Rocket/J_Leg{i}_H")

        sim_utils.JointCfg.fix(
            parent=f"/World/Rocket/Leg{i}_H",
            child=f"/World/Rocket/Leg{i}_V",
        ).func(f"/World/Rocket/J_Leg{i}_V")

    # --------------------------------------------------------
    # Disable internal collisions (using collision groups)
    # --------------------------------------------------------
    physx = PhysicsContext.instance()
    ROCKET_GROUP = 2

    rocket_parts = [
        "/World/Rocket/Nose",
        "/World/Rocket/Body",
        "/World/Rocket/Thruster",
    ]
    for i in range(3):
        rocket_parts += [
            f"/World/Rocket/Leg{i}_H",
            f"/World/Rocket/Leg{i}_V",
        ]

    for p in rocket_parts:
        physx.set_collision_group(p, ROCKET_GROUP)
        physx.set_collision_group_mask(p, 0)  # disable collisions within group


# ------------------------------------------------------------
# Main Loop With Keyboard Control
# ------------------------------------------------------------

def main():

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    design_scene()
    sim.reset()

    print("[INFO] Rocket with keyboard thruster control ready.")

    thrust = 0.0
    yaw_force = 0.0
    pitch_force = 0.0

    while simulation_app.is_running():

        # -------------------------------
        # Keyboard Controls
        # -------------------------------
        if keyboard.is_pressed("w"):
            thrust += 0.5
        if keyboard.is_pressed("s"):
            thrust -= 0.5

        if keyboard.is_pressed("a"):
            yaw_force = 40.0
        elif keyboard.is_pressed("d"):
            yaw_force = -40.0
        else:
            yaw_force = 0.0

        if keyboard.is_pressed("up"):
            pitch_force = 40.0
        elif keyboard.is_pressed("down"):
            pitch_force = -40.0
        else:
            pitch_force = 0.0

        thrust = max(0.0, min(thrust, 400.0))

        # Main thrust straight upward
        sim.apply_force_at_pos(
            "/World/Rocket/Body",
            (0.0, 0.0, thrust),
            (0.0, 0.0, 2.0),
        )

        # Yaw control (left-right)
        sim.apply_force_at_pos(
            "/World/Rocket/Body",
            (yaw_force, 0.0, 0.0),
            (0.0, 0.0, 2.0),
        )

        # Pitch control (forward-backward)
        sim.apply_force_at_pos(
            "/World/Rocket/Body",
            (0.0, pitch_force, 0.0),
            (0.0, 0.0, 2.0),
        )

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
