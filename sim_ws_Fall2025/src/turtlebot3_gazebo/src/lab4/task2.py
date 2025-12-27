#!/usr/bin/env python3
import math
import heapq
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf_transformations import euler_from_quaternion


#  Utility functions 

def get_yaw_from_quaternion(q):
    quat = [q.x, q.y, q.z, q.w]
    _, _, yaw = euler_from_quaternion(quat)
    return yaw


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# graph classes (A*) 

class GraphNode:
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        self.costs = []

    def add_edges(self, nums, ws):
        self.neighbors.extend(nums)
        self.costs.extend(ws)


class GridGraph:
    def __init__(self):
        self.nodes = {}


class AStar:
    def __init__(self, graph: GridGraph):
        self.graph = graph
        self.g = {}
        self.h = {}
        self.prev = {}

    def _reset(self):
        self.g = {n: float('inf') for n in self.graph.nodes}
        self.h = {n: 0.0 for n in self.graph.nodes}
        self.prev = {n: None for n in self.graph.nodes}

    def _calc_heuristic(self, goal):
        goal_row, goal_column = map(int, goal.split(','))
        for name in self.graph.nodes:
            row, column = map(int, name.split(','))
            self.h[name] = math.hypot(goal_row - row, goal_column - column)

    def plan(self, start_name, goal):
        if start_name not in self.graph.nodes or goal not in self.graph.nodes:
            return
        self._reset()
        self._calc_heuristic(goal)
        self.g[start_name] = 0.0

        open_heap = [(self.h[start_name], start_name)]
        closed = set()

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                break
            node = self.graph.nodes[cur]
            for neighbor, w in zip(node.neighbors, node.costs):
                neighbor_name = neighbor.name
                newg = self.g[cur] + float(w)
                if newg < self.g[neighbor_name]:
                    self.g[neighbor_name] = newg
                    self.prev[neighbor_name] = cur
                    heapq.heappush(open_heap, (newg + self.h[neighbor_name], neighbor_name))

    def reconstruct(self, start_name, goal):
        if self.g.get(goal, float('inf')) == float('inf'):
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = self.prev[cur]
        path.reverse()
        return path


# RRT* structures 

class RRTStarNode:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent  # index in node list
        self.cost = cost      # accumulated cost from root


# Navigation Node 

class Navigation(Node):
    def __init__(self, node_name='task2_algorithm'):
        super().__init__(node_name)
        self.get_logger().info("[INIT] Task2 Navigation (A* + RRT*) started")

        # Path planning / robot state
        self.path = Path()         # current path being followed (A* or RRT*)
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        # Planner mode: "ASTAR" (default) or "RRTSTAR"
        self.planner_mode = "ASTAR"
        self.rrt_has_plan = False   # we only plan RRT* once per trigger
        self.rrt_path = Path()
        self.backup_steps_remaining = 0  # for backing up before RRT*

        # FSM mode: "FOLLOW" or "BACKUP_RRT"
        self.mode = "FOLLOW"

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.__ttbot_pose_cbk, 10)

        # Map QoS (latched / transient local)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)

        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)

        # Map / robot footprint settings
        self.treat_unknown_as_occupied = True
        self.robot_radius = 0.19
        self.extra_inflation = 0.09

        # Planner / follower parameters
        self.lookahead = 0.30
        self.goal_tol = 0.12
        self.max_lin = 0.12
        self.max_ang = 0.8
        self.rotate_threshold = 0.6

        # Map / graph state
        self.map_loaded = False
        self.map_occ = None
        self.inflated = None
        self.res = 0.05
        self.ox = 0.0
        self.oy = 0.0
        self.W = 0
        self.H = 0
        self.graph = None
        self.astar = None
        self.arrived = False

        # Velocity smoothing
        self.v_last = 0.0
        self.w_last = 0.0
        self.smoothing_alpha = 0.45

        # Laser scan
        self.scan_msg = None

        # RRT* parameters
        self.rrt_max_iters = 2000
        self.rrt_step_size = 0.35
        self.rrt_neighbor_radius = 0.7
        self.rrt_goal_radius = 0.35
        self.rrt_goal_sample_rate = 0.08
        self.rrt_collision_step = 0.03

        # Threshold for RRT* trigger (trashcan very close in front)
        self.rrt_trigger_dist = 0.55

        # Desired clearance from dynamic obstacle for paths
        # Effective forbidden radius around obstacle = robot_radius + rrt_clearance
        self.rrt_clearance = 0.30  # meters extra beyond robot body

        # Flag: whether we are still allowed to use RRT*
        self.rrt_enabled = True

        # RRT* waypoint tracking
        self.rrt_waypoint_idx = 0
        self.rrt_waypoint_tol = 0.10  # distance to consider a waypoint reached

        self.get_logger().info("[READY] Waiting for /map, /amcl_pose, and goals")

    # Callbacks 

    def __goal_pose_cbk(self, data: PoseStamped):
        self.goal_pose = data
        self.arrived = False
        self.mode = "FOLLOW"
        # Reset planner to ASTAR and re-enable RRT* for new goal
        self.planner_mode = "ASTAR"
        self.rrt_has_plan = False
        self.rrt_path = Path()
        self.rrt_enabled = True
        self.get_logger().info(
            f"New goal: ({data.pose.position.x:.3f}, {data.pose.position.y:.3f}) [ASTAR mode]"
        )

    def __ttbot_pose_cbk(self, data: PoseWithCovarianceStamped):
        ps = PoseStamped()
        ps.header = data.header
        ps.pose = data.pose.pose
        self.ttbot_pose = ps

    def map_callback(self, msg: OccupancyGrid):
        self.res = msg.info.resolution
        self.ox = msg.info.origin.position.x
        self.oy = msg.info.origin.position.y
        self.W = msg.info.width
        self.H = msg.info.height

        occ = np.array(msg.data, dtype=np.int16).reshape(self.H, self.W)
        if self.treat_unknown_as_occupied:
            occ[occ < 0] = 100

        self.map_occ = occ
        self.inflated = self._inflate(occ)
        self.graph = self._build_graph_from_inflated(self.inflated)
        self.astar = AStar(self.graph)
        self.map_loaded = True

        free = int(np.count_nonzero(self.inflated == 0))
        occ_cnt = int(np.count_nonzero(self.inflated == 100))
        self.get_logger().info(f"[MAP] Got /map: {self.W}x{self.H}, res={self.res:.3f}, free={free}, occ={occ_cnt}")
        self.get_logger().info(f"[GRAPH] nodes={len(self.graph.nodes)}")

    def scan_callback(self, msg: LaserScan):
        self.scan_msg = msg

    #  Map / Graph helpers 

    def _inflate(self, occ_grid: np.ndarray):
        H, W = occ_grid.shape
        base = np.where(occ_grid >= 50, 100, 0).astype(np.uint8)
        robot_r = float(self.robot_radius)
        pad = float(self.extra_inflation)
        r_cells = int(math.ceil((robot_r + pad) / self.res))
        if r_cells <= 0:
            return base

        yy, xx = np.ogrid[-r_cells:r_cells+1, -r_cells:r_cells+1]
        disk = (xx*xx + yy*yy) <= (r_cells*r_cells)
        dil = base.copy()
        for dy in range(-r_cells, r_cells+1):
            for dx in range(-r_cells, r_cells+1):
                if not disk[dy + r_cells, dx + r_cells]:
                    continue
                src_y0 = max(0, -dy)
                src_y1 = H - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = W - max(0, dx)
                dst_y0 = max(0, dy)
                dst_y1 = H - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = W - max(0, -dx)
                dst = dil[dst_y0:dst_y1, dst_x0:dst_x1]
                src = base[src_y0:src_y1, src_x0:src_x1]
                np.maximum(dst, src, out=dst)
        return dil

    def _build_graph_from_inflated(self, inflated):
        H, W = inflated.shape
        g = GridGraph()
        for r in range(H):
            for c in range(W):
                if inflated[r, c] == 0:
                    g.nodes[f"{r},{c}"] = GraphNode(f"{r},{c}")
        nbrs = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1,  1, math.sqrt(2.0)),
            ( 1, -1, math.sqrt(2.0)), ( 1,  1, math.sqrt(2.0))
        ]
        for r in range(H):
            for c in range(W):
                if inflated[r, c] != 0:
                    continue
                parent = g.nodes.get(f"{r},{c}")
                if parent is None:
                    continue
                neighbors, weights = [], []
                for dr, dc, w in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and inflated[nr, nc] == 0:
                        neighbors.append(g.nodes[f"{nr},{nc}"])
                        weights.append(w)
                if neighbors:
                    parent.add_edges(neighbors, weights)
        return g

    # Grid <-> World 

    def _in_map_world(self, x, y):
        return (self.ox <= x < self.ox + self.W * self.res) and (self.oy <= y < self.oy + self.H * self.res)

    def _nearest_free(self, col, row):
        H, W = self.H, self.W
        if 0 <= row < H and 0 <= col < W and self.inflated[row, col] == 0:
            return col, row
        visited = np.zeros((H, W), dtype=np.uint8)
        q = deque()
        c0 = max(0, min(W-1, col))
        r0 = max(0, min(H-1, row))
        q.append((c0, r0))
        visited[r0, c0] = 1
        while q:
            c, r = q.popleft()
            if self.inflated[r, c] == 0:
                return c, r
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nc, nr = c+dc, r+dr
                if 0 <= nc < W and 0 <= nr < H and not visited[nr, nc]:
                    visited[nr, nc] = 1
                    q.append((nc, nr))
        nc = max(0, min(W-1, col))
        nr = max(0, min(H-1, row))
        return nc, nr

    def world_to_grid(self, x, y):
        col = int((x - self.ox) / self.res)
        row = int((y - self.oy) / self.res)
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        return col, row

    def grid_to_world(self, col, row):
        x = self.ox + (col + 0.5) * self.res
        y = self.oy + (row + 0.5) * self.res
        return x, y

    # Dynamic obstacles into inflated map 

    def mark_dynamic_obstacle_world(self, x, y, radius=None):
        """Mark a circular region around (x,y) as occupied in the inflated map."""
        if self.inflated is None:
            return

        # Default radius = robot body + desired extra clearance
        if radius is None:
            radius = self.robot_radius + self.rrt_clearance  # e.g. 0.19 + 0.25 = 0.44 m

        c_center, r_center = self.world_to_grid(x, y)
        r_cells = int(math.ceil(radius / self.res))
        H, W = self.H, self.W
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                rr = r_center + dr
                cc = c_center + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if dr * dr + dc * dc <= r_cells * r_cells:
                        self.inflated[rr, cc] = 100  # occupied

        # Rebuild A* graph so future A* plans respect this obstacle
        self.graph = self._build_graph_from_inflated(self.inflated)
        self.astar = AStar(self.graph)

    # A* Planner 

    def a_star_path_planner(self, start_pose, end_pose):
        path = Path()
        path.header.frame_id = end_pose.header.frame_id or 'map'
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        sx_w, sy_w = start_pose.pose.position.x, start_pose.pose.position.y
        gx_w, gy_w = end_pose.pose.position.x, end_pose.pose.position.y

        if not self.map_loaded or self.graph is None:
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        if not self._in_map_world(gx_w, gy_w):
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        sc, sr = self.world_to_grid(sx_w, sy_w)
        goal_column, goal_row = self.world_to_grid(gx_w, gy_w)

        sc, sr = self._nearest_free(sc, sr)
        goal_column, goal_row = self._nearest_free(goal_column, goal_row)
        s_name = f"{sr},{sc}"
        g_name = f"{goal_row},{goal_column}"

        if s_name not in self.graph.nodes or g_name not in self.graph.nodes:
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        self.astar.plan(s_name, g_name)
        names = self.astar.reconstruct(s_name, g_name)
        if not names:
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        for nm in names:
            r, c = map(int, nm.split(','))
            wx, wy = self.grid_to_world(c, r)
            ps = PoseStamped()
            ps.header.frame_id = path.header.frame_id
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        tmsg = Float32()
        tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
        self.calc_time_pub.publish(tmsg)

        self.path_pub.publish(path)
        return path

    # RRT* helpers & planner 

    def is_free_world(self, x, y):
        if not self._in_map_world(x, y):
            return False
        c, r = self.world_to_grid(x, y)
        return self.inflated[r, c] == 0

    def collision_free_segment(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return self.is_free_world(x1, y1)
        steps = int(dist / self.rrt_collision_step) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.is_free_world(x, y):
                return False
        return True

    def rrt_star_plan(self, start, goal):
        sx, sy = start
        gx, gy = goal

        if not self.is_free_world(sx, sy) or not self.is_free_world(gx, gy):
            self.get_logger().warn("[RRT*] start or goal in occupied area")
            return [], None

        x_min = self.ox
        x_max = self.ox + self.W * self.res
        y_min = self.oy
        y_max = self.oy + self.H * self.res

        nodes = [RRTStarNode(sx, sy, parent=None, cost=0.0)]
        goal_idx = None
        best_cost = float('inf')
        found_once = False

        for it in range(self.rrt_max_iters):
            # Sample
            if np.random.rand() < self.rrt_goal_sample_rate:
                x_rand, y_rand = gx, gy
            else:
                x_rand = np.random.uniform(x_min, x_max)
                y_rand = np.random.uniform(y_min, y_max)

            if not self.is_free_world(x_rand, y_rand):
                continue

            # Nearest
            dists = [(i, (n.x - x_rand) ** 2 + (n.y - y_rand) ** 2) for i, n in enumerate(nodes)]
            idx_near, _ = min(dists, key=lambda t: t[1])
            n_near = nodes[idx_near]
            theta = math.atan2(y_rand - n_near.y, x_rand - n_near.x)
            dist = math.hypot(x_rand - n_near.x, y_rand - n_near.y)
            step = min(self.rrt_step_size, dist)
            x_new = n_near.x + step * math.cos(theta)
            y_new = n_near.y + step * math.sin(theta)

            if not self.is_free_world(x_new, y_new):
                continue
            if not self.collision_free_segment(n_near.x, n_near.y, x_new, y_new):
                continue

            # New node initial parent = nearest
            new_node = RRTStarNode(x_new, y_new, parent=idx_near,
                                   cost=n_near.cost + step)

            # Best parent among neighbors
            idx_neighbors = []
            for i, ni in enumerate(nodes):
                d = math.hypot(ni.x - x_new, ni.y - y_new)
                if d <= self.rrt_neighbor_radius:
                    idx_neighbors.append((i, d))

            for i_n, d in idx_neighbors:
                if i_n == idx_near:
                    continue
                ni = nodes[i_n]
                if not self.collision_free_segment(ni.x, ni.y, x_new, y_new):
                    continue
                new_cost = ni.cost + d
                if new_cost < new_node.cost:
                    new_node.cost = new_cost
                    new_node.parent = i_n

            nodes.append(new_node)
            new_idx = len(nodes) - 1

            # Rewire neighbors
            for i_n, d in idx_neighbors:
                ni = nodes[i_n]
                if ni is nodes[new_idx]:
                    continue
                if not self.collision_free_segment(new_node.x, new_node.y, ni.x, ni.y):
                    continue
                alt_cost = new_node.cost + d
                if alt_cost < ni.cost:
                    ni.cost = alt_cost
                    ni.parent = new_idx

            # Check connection to goal
            d_goal = math.hypot(x_new - gx, y_new - gy)
            if d_goal <= self.rrt_goal_radius:
                if self.collision_free_segment(x_new, y_new, gx, gy):
                    goal_cost = new_node.cost + d_goal
                    if goal_cost < best_cost:
                        goal_node = RRTStarNode(gx, gy, parent=new_idx, cost=goal_cost)
                        if goal_idx is None:
                            nodes.append(goal_node)
                            goal_idx = len(nodes) - 1
                        else:
                            nodes[goal_idx] = goal_node
                            goal_idx = len(nodes) - 1
                        best_cost = goal_cost
                        found_once = True

            if found_once and it > self.rrt_max_iters * 0.6:
                break

        if goal_idx is None:
            self.get_logger().warn("[RRT*] Failed to find a path")
        else:
            self.get_logger().info(f"[RRT*] Path found with cost {best_cost:.3f}, nodes={len(nodes)}")

        return nodes, goal_idx

    def rrtstar_path_planner(self, start_pose, end_pose):
        """Like a_star_path_planner but using RRT* once."""
        path = Path()
        path.header.frame_id = end_pose.header.frame_id or 'map'
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        sx_w, sy_w = start_pose.pose.position.x, start_pose.pose.position.y
        gx_w, gy_w = end_pose.pose.position.x, end_pose.pose.position.y

        if not self.map_loaded or self.inflated is None:
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        if not self._in_map_world(gx_w, gy_w):
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        sc, sr = self.world_to_grid(sx_w, sy_w)
        gc, gr = self.world_to_grid(gx_w, gy_w)
        sc, sr = self._nearest_free(sc, sr)
        gc, gr = self._nearest_free(gc, gr)

        sx_w, sy_w = self.grid_to_world(sc, sr)
        gx_w, gy_w = self.grid_to_world(gc, gr)

        nodes, goal_idx = self.rrt_star_plan((sx_w, sy_w), (gx_w, gy_w))
        if goal_idx is None:
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(tmsg)
            return path

        # Reconstruct path
        idx = goal_idx
        pts = []
        while idx is not None:
            n = nodes[idx]
            pts.append((n.x, n.y))
            idx = n.parent
        pts.reverse()

        for (wx, wy) in pts:
            ps = PoseStamped()
            ps.header.frame_id = path.header.frame_id
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        tmsg = Float32()
        tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
        self.calc_time_pub.publish(tmsg)

        self.path_pub.publish(path)
        return path

    # Path following 

    def get_path_idx(self, path, vehicle_pose):
        if not path.poses:
            return 0
        x = vehicle_pose.pose.position.x
        y = vehicle_pose.pose.position.y
        for i, ps in enumerate(path.poses):
            dx = ps.pose.position.x - x
            dy = ps.pose.position.y - y
            if math.hypot(dx, dy) >= self.lookahead:
                return i
        return len(path.poses) - 1

    def path_follower_astar(self, vehicle_pose, current_goal_pose):
        """Follower used for A* paths."""
        robot_x = vehicle_pose.pose.position.x
        robot_y = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y
        ryaw = get_yaw_from_quaternion(vehicle_pose.pose.orientation)
        dx = gx - robot_x
        dy = gy - robot_y
        dist = math.hypot(dx, dy)
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = normalize_angle(tgt_yaw - ryaw)

        if abs(yaw_err) > self.rotate_threshold:
            v_cmd = 0.0
            w_cmd = max(-self.max_ang, min(self.max_ang, 1.5 * yaw_err))
        else:
            v_cmd = self.max_lin * max(0.0, math.cos(yaw_err))
            w_cmd = max(-self.max_ang, min(self.max_ang, 1.2 * yaw_err))

        if dist < self.goal_tol:
            v_cmd = 0.0
            w_cmd = 0.0

        self.v_last = (1.0 - self.smoothing_alpha) * self.v_last + self.smoothing_alpha * v_cmd
        self.w_last = (1.0 - self.smoothing_alpha) * self.w_last + self.smoothing_alpha * w_cmd

        return self.v_last, self.w_last

    def path_follower_rrt(self, vehicle_pose, current_goal_pose):
        """
        RRT* follower: similar to A*, but we also return distance
        to the current waypoint so we can step through RRT* path
        waypoint-by-waypoint.
        """
        robot_x = vehicle_pose.pose.position.x
        robot_y = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y
        ryaw = get_yaw_from_quaternion(vehicle_pose.pose.orientation)

        dx = gx - robot_x
        dy = gy - robot_y
        dist = math.hypot(dx, dy)

        tgt_yaw = math.atan2(dy, dx)
        yaw_err = normalize_angle(tgt_yaw - ryaw)

        rotate_threshold = self.rotate_threshold
        max_lin = self.max_lin
        max_ang = self.max_ang

        if abs(yaw_err) > rotate_threshold:
            v_cmd = 0.0
            w_cmd = max(-max_ang, min(max_ang, 1.5 * yaw_err))
        else:
            v_cmd = max_lin * max(0.0, math.cos(yaw_err))
            w_cmd = max(-max_ang, min(max_ang, 1.2 * yaw_err))

        # Close enough to this waypoint → slow down / stop
        if dist < self.rrt_waypoint_tol:
            v_cmd = 0.0
            w_cmd = 0.0

        self.v_last = (1.0 - self.smoothing_alpha) * self.v_last + self.smoothing_alpha * v_cmd
        self.w_last = (1.0 - self.smoothing_alpha) * self.w_last + self.smoothing_alpha * w_cmd

        return self.v_last, self.w_last, dist

    # Scan analysis & obstacle estimation 

    def analyze_scan(self):
        """
        Returns (front_min, left_min, right_min) distances in meters.
        front:   [-40°, +40°]
        left:    [ 20°, +90°]
        right:   [-90°, -20°]
        """
        if self.scan_msg is None:
            return float('inf'), float('inf'), float('inf')

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment

        def angle_to_index(angle):
            return int((angle - angle_min) / angle_inc)

        def sector_min(a_start_deg, a_end_deg):
            a0 = math.radians(a_start_deg)
            a1 = math.radians(a_end_deg)
            i0 = max(0, min(len(ranges) - 1, angle_to_index(a0)))
            i1 = max(0, min(len(ranges) - 1, angle_to_index(a1)))
            i_min = min(i0, i1)
            i_max = max(i0, i1)
            sector = ranges[i_min:i_max+1]
            sector = sector[np.isfinite(sector)]
            if len(sector) == 0:
                return float('inf')
            return float(np.min(sector))

        front_min = sector_min(-40, 40)
        left_min = sector_min(20, 90)
        right_min = sector_min(-90, -20)

        return front_min, left_min, right_min

    def estimate_front_obstacle_world(self, deg_min=-30, deg_max=30):
        """
        Estimate the nearest obstacle in a front cone in WORLD coordinates.
        """
        if self.scan_msg is None:
            return None

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment
        n = len(ranges)
        angles = angle_min + angle_inc * np.arange(n)

        a0 = math.radians(deg_min)
        a1 = math.radians(deg_max)
        mask = (angles >= a0) & (angles <= a1)
        sector_ranges = ranges[mask]
        sector_angles = angles[mask]

        finite_mask = np.isfinite(sector_ranges)
        if not finite_mask.any():
            return None

        idx_rel = np.argmin(sector_ranges[finite_mask])
        r = float(sector_ranges[finite_mask][idx_rel])
        ang = float(sector_angles[finite_mask][idx_rel])

        robot_x = self.ttbot_pose.pose.position.x
        robot_y = self.ttbot_pose.pose.position.y
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)
        theta_world = yaw + ang
        x_obs = robot_x + r * math.cos(theta_world)
        y_obs = robot_y + r * math.sin(theta_world)
        return x_obs, y_obs, r

    # Robot motion 

    def move_ttbot(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd)

    #  Main loop 

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if not self.map_loaded:
                continue
            if self.goal_pose.header.stamp.sec == 0 or self.ttbot_pose.header.stamp.sec == 0:
                continue

            robot_x = self.ttbot_pose.pose.position.x
            robot_y = self.ttbot_pose.pose.position.y
            gx = self.goal_pose.pose.position.x
            gy = self.goal_pose.pose.position.y

            # Goal check
            if math.hypot(gx - robot_x, gy - robot_y) < self.goal_tol:
                if not self.arrived:
                    self.arrived = True
                    self.move_ttbot(0.0, 0.0)
                    self.get_logger().info("[RUN] Goal reached. Stopping.")
                self.mode = "FOLLOW"
                self.planner_mode = "ASTAR"
                self.rrt_has_plan = False
                self.rrt_path = Path()
                continue

            # Analyze scan
            front_min, left_min, right_min = self.analyze_scan()

            # --- RRT* trigger: trashcan very close in front while following A* ---
            if self.mode == "FOLLOW" and self.planner_mode == "ASTAR" and self.rrt_enabled:
                if front_min < self.rrt_trigger_dist:
                    est = self.estimate_front_obstacle_world()
                    if est is not None:
                        x_obs, y_obs, r = est
                        # Mark dynamic obstacle using robot_radius + rrt_clearance
                        self.mark_dynamic_obstacle_world(x_obs, y_obs)
                        self.get_logger().info(
                            f"[RRT-TRIGGER] Obstacle at ({x_obs:.2f},{y_obs:.2f}), d={r:.2f}. "
                            f"Marking and switching to BACKUP_RRT + RRT*."
                        )
                        self.planner_mode = "RRTSTAR"
                        self.rrt_has_plan = False
                        self.rrt_path = Path()
                        self.backup_steps_remaining = 25  # ~2.5s at 0.1s per loop
                        self.mode = "BACKUP_RRT"
                        self.move_ttbot(0.0, 0.0)
                        continue

            # BACKUP FOR RRT* 
            if self.mode == "BACKUP_RRT":
                if self.backup_steps_remaining > 0:
                    self.backup_steps_remaining -= 1
                    # simple straight reverse (no wandering)
                    self.move_ttbot(-0.08, 0.0)
                    continue
                else:
                    # finished backing, now compute RRT* path once
                    if not self.rrt_has_plan:
                        path = self.rrtstar_path_planner(self.ttbot_pose, self.goal_pose)
                        self.rrt_path = path
                        self.rrt_has_plan = bool(path.poses)
                        self.rrt_waypoint_idx = 0

                        if not self.rrt_has_plan:
                            # RRT* FAILED: stop, disable RRT*, and fall back to A*
                            self.get_logger().warn(
                                "[RRT*] No path after backup. Disabling RRT* and falling back to A*."
                            )
                            self.move_ttbot(0.0, 0.0)
                            self.rrt_enabled = False
                            self.planner_mode = "ASTAR"
                            self.mode = "FOLLOW"
                            # Graph already rebuilt in mark_dynamic_obstacle_world
                            continue
                        else:
                            self.get_logger().info(
                                f"[RRT*] Planned path with {len(self.rrt_path.poses)} poses after backup."
                            )

                    # RRT* succeeded → go follow that RRT* path
                    self.mode = "FOLLOW"
                    # fall through to follower using RRT* path

            #  Global plan selection 
            if self.planner_mode == "ASTAR":
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            else:  # RRTSTAR
                if not self.rrt_has_plan or not self.rrt_path.poses:
                    # safety: if plan is missing, try to replan once
                    path = self.rrtstar_path_planner(self.ttbot_pose, self.goal_pose)
                    self.rrt_path = path
                    self.rrt_has_plan = bool(path.poses)
                    self.rrt_waypoint_idx = 0
                else:
                    path = self.rrt_path

            self.path = path

            if not path.poses:
                self.move_ttbot(0.0, 0.0)
                continue

            # FOLLOW behavior only – no local AVOID
            if self.mode == "FOLLOW":
                if self.planner_mode == "ASTAR":
                    # --- A* FOLLOW: follow the A* path only ---
                    idx = self.get_path_idx(path, self.ttbot_pose)
                    current_goal = path.poses[idx]
                    v_cmd, w_cmd = self.path_follower_astar(self.ttbot_pose, current_goal)

                else:
                    # RRT* FOLLOW: strictly follow RRT* waypoints in order 
                    if not self.rrt_has_plan or not path.poses:
                        self.get_logger().warn("[RRT*] FOLLOW mode but no valid path. Stopping.")
                        v_cmd = 0.0
                        w_cmd = 0.0
                    else:
                        if self.rrt_waypoint_idx >= len(path.poses):
                            self.rrt_waypoint_idx = len(path.poses) - 1

                        current_goal = path.poses[self.rrt_waypoint_idx]
                        v_cmd, w_cmd, dist_wp = self.path_follower_rrt(self.ttbot_pose, current_goal)

                        # Advance through the RRT* waypoints strictly in order
                        if dist_wp < self.rrt_waypoint_tol and self.rrt_waypoint_idx < len(path.poses) - 1:
                            self.rrt_waypoint_idx += 1
            else:
                # BACKUP_RRT handled earlier; shouldn't reach here
                v_cmd = 0.0
                w_cmd = 0.0

            self.move_ttbot(v_cmd, w_cmd)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='task2_algorithm')
    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            nav.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
