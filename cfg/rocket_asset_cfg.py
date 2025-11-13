import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_BASE_DIR

ROCKET_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_BASE_DIR}/space_robot/assets/rocket/rocket.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={},
    ),
    actuators={
        "rocket_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=200.0,
            stiffness=20000.0,
            damping=200.0,
        )
    },
)
