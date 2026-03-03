#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from rclpy.action import ActionClient

import tf2_ros
#from tf_transformations import euler_from_quaternion
import math


@dataclass
class FrontierCluster:
    cells: List[Tuple[int, int]]          # (mx, my) grid indices
    centroid_m: Tuple[float, float]       # (x, y) in map frame meters
    size: int


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FrontierExplorer(Node):
    """
    Frontier-based exploration node (Active SLAM driver) for ROS 2 + Nav2.

    Inputs:
      - /map (nav_msgs/OccupancyGrid)
      - TF map -> base_frame (base_link / base_footprint)

    Outputs:
      - Nav2 NavigateToPose goals to /navigate_to_pose

    Requirements:
      - Nav2 running and providing NavigateToPose action server
      - A SLAM/mapping node publishing /map and TF map->odom
      - Your robot localization/odometry providing TF odom->base
    """

    def __init__(self):
        super().__init__("frontier_explorer")

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("base_frame", "base_link")   # try base_footprint if your tree uses it
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("nav2_action_name", "navigate_to_pose")
        self.declare_parameter("plan_timeout_sec", 180.0)   # max time to wait for a goal to finish

        self.declare_parameter("update_period_sec", 1.0)
        self.declare_parameter("min_frontier_cluster_size", 15)  # filter tiny noisy frontiers
        self.declare_parameter("frontier_score_w_dist", 1.0)     # higher => penalize far
        self.declare_parameter("frontier_score_w_size", 0.3)     # higher => prefer big frontier
        self.declare_parameter("goal_clearance_m", 0.35)         # step back from frontier into free space
        self.declare_parameter("unknown_value", -1)
        self.declare_parameter("free_value_max", 5)              # treat 0..5 as free-ish (some maps use 0 only)
        self.declare_parameter("occupied_value_min", 65)         # treat 65..100 as occupied
        self.declare_parameter("max_goal_attempts_per_target", 2)
        self.declare_parameter("blacklist_radius_m", 0.6)        # if goal fails, blacklist nearby goals

        self.map_topic = self.get_parameter("map_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.action_name = self.get_parameter("nav2_action_name").value

        self.update_period = float(self.get_parameter("update_period_sec").value)
        self.min_cluster = int(self.get_parameter("min_frontier_cluster_size").value)
        self.w_dist = float(self.get_parameter("frontier_score_w_dist").value)
        self.w_size = float(self.get_parameter("frontier_score_w_size").value)
        self.goal_clearance = float(self.get_parameter("goal_clearance_m").value)
        self.unknown_value = int(self.get_parameter("unknown_value").value)
        self.free_value_max = int(self.get_parameter("free_value_max").value)
        self.occ_value_min = int(self.get_parameter("occupied_value_min").value)
        self.plan_timeout = float(self.get_parameter("plan_timeout_sec").value)
        self.max_attempts = int(self.get_parameter("max_goal_attempts_per_target").value)
        self.blacklist_radius = float(self.get_parameter("blacklist_radius_m").value)

        # -----------------------
        # Map subscription (use transient local to get last map immediately)
        # -----------------------
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, qos
        )

        # -----------------------
        # TF
        # -----------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -----------------------
        # Nav2 Action client
        # -----------------------
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)

        # -----------------------
        # State
        # -----------------------
        self.map: Optional[OccupancyGrid] = None
        self.map_data = None
        self.map_w = 0
        self.map_h = 0
        self.res = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.exploring = False
        self.current_goal: Optional[Tuple[float, float]] = None
        self.goal_attempts = 0
        self.blacklist: List[Tuple[float, float]] = []  # failed goals (x,y)

        # Periodic loop
        self.timer = self.create_timer(self.update_period, self.loop)

        self.get_logger().info(
            "FrontierExplorer started. Waiting for /map and Nav2 action server..."
        )

    # -----------------------
    # Callbacks and main loop
    # -----------------------
    def on_map(self, msg: OccupancyGrid):
        self.map = msg
        self.map_data = msg.data
        self.map_w = msg.info.width
        self.map_h = msg.info.height
        self.res = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

    def loop(self):
        if self.map is None:
            return

        if not self.nav_client.wait_for_server(timeout_sec=0.0):
            # Don’t spam logs
            return

        pose = self.get_robot_pose_map()
        if pose is None:
            return
        rx, ry, ryaw = pose

        # If we are currently navigating, don't enqueue new goals
        if self.exploring:
            return

        # Compute frontiers
        frontier_cells = self.detect_frontier_cells()
        if not frontier_cells:
            self.get_logger().info("No frontier cells found. Exploration may be complete.")
            return

        clusters = self.cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if c.size >= self.min_cluster]
        if not clusters:
            self.get_logger().info("Frontiers exist but all are too small/noisy. Done.")
            return

        # Score and select best target
        best_goal = self.select_best_goal(clusters, (rx, ry))
        if best_goal is None:
            self.get_logger().info("All candidate goals are blacklisted or invalid.")
            return

        gx, gy = best_goal
        self.send_nav_goal(gx, gy, ryaw)

    # -----------------------
    # Pose
    # -----------------------
    def get_robot_pose_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            # Common when TF not ready or base_frame name differs
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = yaw_from_quat(tf.transform.rotation)
        return x, y, yaw

    # -----------------------
    # Frontier detection
    # -----------------------
    def idx(self, mx: int, my: int) -> int:
        return my * self.map_w + mx

    def in_bounds(self, mx: int, my: int) -> bool:
        return 0 <= mx < self.map_w and 0 <= my < self.map_h

    def cell(self, mx: int, my: int) -> int:
        return int(self.map_data[self.idx(mx, my)])

    def is_free(self, v: int) -> bool:
        return 0 <= v <= self.free_value_max

    def is_unknown(self, v: int) -> bool:
        return v == self.unknown_value

    def is_occupied(self, v: int) -> bool:
        return v >= self.occ_value_min

    def detect_frontier_cells(self) -> List[Tuple[int, int]]:
        """
        Frontier cell: unknown cell adjacent (4-neighbor) to a free cell.
        """
        frontiers = []
        # Skip borders
        for my in range(1, self.map_h - 1):
            row_off = my * self.map_w
            for mx in range(1, self.map_w - 1):
                v = int(self.map_data[row_off + mx])
                if not self.is_unknown(v):
                    continue

                # Check if any neighbor is free
                if (self.is_free(self.cell(mx + 1, my)) or
                    self.is_free(self.cell(mx - 1, my)) or
                    self.is_free(self.cell(mx, my + 1)) or
                    self.is_free(self.cell(mx, my - 1))):
                    frontiers.append((mx, my))
        return frontiers

    def cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[FrontierCluster]:
        """
        BFS clustering on frontier grid cells (8-connected).
        """
        frontier_set = set(frontier_cells)
        visited = set()
        clusters: List[FrontierCluster] = []

        neighbors8 = [(-1, -1), (0, -1), (1, -1),
                      (-1,  0),          (1,  0),
                      (-1,  1), (0,  1), (1,  1)]

        for start in frontier_cells:
            if start in visited:
                continue
            q: Deque[Tuple[int, int]] = deque([start])
            visited.add(start)
            cells = []

            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dx, dy in neighbors8:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

            # centroid in meters
            cx_m, cy_m = self.cluster_centroid_m(cells)
            clusters.append(FrontierCluster(cells=cells, centroid_m=(cx_m, cy_m), size=len(cells)))

        return clusters

    def grid_to_map(self, mx: int, my: int) -> Tuple[float, float]:
        x = self.origin_x + (mx + 0.5) * self.res
        y = self.origin_y + (my + 0.5) * self.res
        return x, y

    def map_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        mx = int((x - self.origin_x) / self.res)
        my = int((y - self.origin_y) / self.res)
        return mx, my

    def cluster_centroid_m(self, cells: List[Tuple[int, int]]) -> Tuple[float, float]:
        sx = 0.0
        sy = 0.0
        for mx, my in cells:
            x, y = self.grid_to_map(mx, my)
            sx += x
            sy += y
        n = max(1, len(cells))
        return sx / n, sy / n

    # -----------------------
    # Goal selection
    # -----------------------
    def is_blacklisted(self, x: float, y: float) -> bool:
        for bx, by in self.blacklist:
            if (x - bx) ** 2 + (y - by) ** 2 <= self.blacklist_radius ** 2:
                return True
        return False

    def select_best_goal(self, clusters: List[FrontierCluster], robot_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        rx, ry = robot_xy
        best = None
        best_score = float("inf")

        # Sort clusters by (distance - size bonus) score
        for c in clusters:
            fx, fy = c.centroid_m
            if self.is_blacklisted(fx, fy):
                continue

            # Push goal slightly back from frontier into free space (heuristic)
            gx, gy = self.pull_back_into_free((fx, fy), (rx, ry), self.goal_clearance)
            if gx is None:
                continue
            if self.is_blacklisted(gx, gy):
                continue

            dist = math.hypot(gx - rx, gy - ry)
            # Score: prefer closer + bigger
            score = self.w_dist * dist - self.w_size * float(c.size)

            if score < best_score:
                best_score = score
                best = (gx, gy)

        return best

    def pull_back_into_free(self, frontier_xy: Tuple[float, float], robot_xy: Tuple[float, float], back_m: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Move goal from frontier centroid slightly toward the robot (into explored/free space),
        and verify it's in a free-ish cell.
        """
        fx, fy = frontier_xy
        rx, ry = robot_xy
        vx, vy = (rx - fx), (ry - fy)
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return None, None

        # Step toward robot
        ux, uy = vx / norm, vy / norm
        gx = fx + ux * back_m
        gy = fy + uy * back_m

        # Validate goal cell is free
        mx, my = self.map_to_grid(gx, gy)
        if not self.in_bounds(mx, my):
            return None, None
        v = self.cell(mx, my)
        if not self.is_free(v):
            # Try smaller pullbacks
            for scale in [0.75, 0.5, 0.25, 0.1]:
                gx2 = fx + ux * (back_m * scale)
                gy2 = fy + uy * (back_m * scale)
                mx2, my2 = self.map_to_grid(gx2, gy2)
                if self.in_bounds(mx2, my2) and self.is_free(self.cell(mx2, my2)):
                    return gx2, gy2
            return None, None

        return gx, gy

    # -----------------------
    # Nav2 action
    # -----------------------
    def send_nav_goal(self, x: float, y: float, yaw_hint: float):
        goal_msg = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)

        # Keep yaw: point roughly forward (use robot yaw as a safe default)
        # You can set yaw toward the frontier, but yaw isn't critical for reaching a point.
        qz = math.sin(yaw_hint * 0.5)
        qw = math.cos(yaw_hint * 0.5)
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        goal_msg.pose = ps

        self.get_logger().info(f"Sending goal: x={x:.2f}, y={y:.2f} (map frame)")
        self.exploring = True
        self.current_goal = (x, y)

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2.")
            self._mark_goal_failed()
            return

        result_future = goal_handle.get_result_async()
        # Add a timeout via a timer check
        self._goal_start_time = time.time()
        result_future.add_done_callback(self._on_nav_result)

        # Also create a watchdog timer
        self._watchdog_timer = self.create_timer(0.5, lambda: self._watchdog(goal_handle))

    def _watchdog(self, goal_handle):
        if not self.exploring:
            try:
                self._watchdog_timer.cancel()
            except Exception:
                pass
            return

        if (time.time() - self._goal_start_time) > self.plan_timeout:
            self.get_logger().warn("Navigation timeout; canceling goal.")
            cancel_future = goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: None)
            self._mark_goal_failed()

            try:
                self._watchdog_timer.cancel()
            except Exception:
                pass

    def _on_nav_result(self, future):
        try:
            result = future.result().result
            status = future.result().status
        except Exception:
            self.get_logger().warn("Failed to get Nav2 result.")
            self._mark_goal_failed()
            return

        # Nav2 status codes: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (commonly)
        if status == 4:
            self.get_logger().info("Goal reached.")
            self.goal_attempts = 0
            self.exploring = False
            self.current_goal = None
        else:
            self.get_logger().warn(f"Goal failed with status={status}.")
            self._mark_goal_failed()

    def _mark_goal_failed(self):
        self.goal_attempts += 1

        if self.current_goal is not None:
            gx, gy = self.current_goal
            # After enough tries, blacklist near this goal
            if self.goal_attempts >= self.max_attempts:
                self.get_logger().warn(f"Blacklisting goal area near x={gx:.2f}, y={gy:.2f}")
                self.blacklist.append((gx, gy))
                self.goal_attempts = 0

        self.exploring = False
        self.current_goal = None

    def _mark_goal_rejected(self):
        self._mark_goal_failed()


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

