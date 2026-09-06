"""Self-contained MuJoCo crowd simulator for the crowd-nav controller.

Kinematic by design: the crowd is integrated by a social-force model and the robot base is
placed from the controller's velocity command, so there is no gait controller to tune and no
physics blow-ups. MuJoCo supplies the scene, the collision geometry and the renderer.
"""
__all__ = ["scene", "crowd", "simulate"]
