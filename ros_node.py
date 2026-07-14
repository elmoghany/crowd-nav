#!/usr/bin/env python3
"""
Example ROS node wiring the PeRoI controller to a real robot. Works with ROS 1 (rospy) or ROS 2
(rclpy) — the block that imports is selected at runtime. This is a REFERENCE you adapt to your stack;
the only project-specific parts are (a) how you get tracked people, and (b) your base's command type.

Inputs it expects (remap to your topics):
  - robot pose in a fixed frame (e.g. `map`): from /odom (nav_msgs/Odometry) or TF.
  - tracked people as a list of (id, x, y) in the SAME frame: adapt `people_cb` to your detector/tracker
    (spencer_tracking_msgs/TrackedPersons, people_msgs/People, a MarkerArray, etc.).
  - a goal (geometry_msgs/PoseStamped) on /peroi_goal, in the same frame.
Output:
  - geometry_msgs/Twist on /cmd_vel. World-frame (vx,vy) from the controller is rotated into the base
    frame using the robot yaw; if your base is differential-drive, see `to_diff_drive`.

Run:  ROS1:  rosrun peroi ros_node.py _ckpt:=/abs/path/residual_predictor.pt
      ROS2:  ros2 run peroi ros_node --ros-args -p ckpt:=/abs/path/residual_predictor.pt
"""
import math
import numpy as np
from peroi_controller import PeRoIController

# --- optional: convert a world-frame (vx,vy) command into a differential-drive (v, omega) ---
def to_diff_drive(vx, vy, yaw, k_omega=1.5, v_max=0.8, w_max=1.2):
    """For a robot that can't strafe: drive forward along its heading, turn toward the desired velocity
    direction. Holonomic bases should instead rotate (vx,vy) into the base frame (see cmd builder)."""
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        return 0.0, 0.0
    desired_yaw = math.atan2(vy, vx)
    err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
    v = max(0.0, min(v_max, speed * math.cos(err)))     # slow down when not yet facing the direction
    w = max(-w_max, min(w_max, k_omega * err))
    return v, w


def build_ros1(ckpt, v_max, holonomic):
    import rospy
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry

    ctrl = PeRoIController(ckpt, v_max=v_max); ctrl.reset()
    state = {"robot": None, "yaw": 0.0, "goal": None, "peds": {}}

    def odom_cb(msg):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        state["robot"] = (p.x, p.y)
        state["yaw"] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def goal_cb(msg):
        state["goal"] = (msg.pose.position.x, msg.pose.position.y)

    def people_cb(msg):
        # ADAPT: turn your tracker message into {track_id: (x, y)} in the robot's frame.
        state["peds"] = {i: (m.pose.position.x, m.pose.position.y) for i, m in enumerate(msg.markers)}

    rospy.init_node("peroi_controller")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/odom", Odometry, odom_cb)
    rospy.Subscriber("/peroi_goal", PoseStamped, goal_cb)
    from visualization_msgs.msg import MarkerArray
    rospy.Subscriber("/tracked_people", MarkerArray, people_cb)   # ADAPT topic + type
    rate = rospy.Rate(8); dt = 1.0 / 8
    while not rospy.is_shutdown():
        if state["robot"] and state["goal"]:
            vx, vy = ctrl.step(state["robot"], state["goal"], state["peds"], dt=dt)
            t = Twist()
            if holonomic:                      # rotate world (vx,vy) into the base frame
                c, s = math.cos(-state["yaw"]), math.sin(-state["yaw"])
                t.linear.x = c * vx - s * vy; t.linear.y = s * vx + c * vy
            else:
                t.linear.x, t.angular.z = to_diff_drive(vx, vy, state["yaw"], v_max=v_max)
            pub.publish(t)
        rate.sleep()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="residual_predictor.pt")
    ap.add_argument("--v_max", type=float, default=0.6)
    ap.add_argument("--holonomic", action="store_true", help="base can strafe (vx,vy); else diff-drive")
    args, _ = ap.parse_known_args()
    build_ros1(args.ckpt, args.v_max, args.holonomic)
