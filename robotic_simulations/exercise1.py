import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from robotic_simulations.environments.environment1 import Environment as Env1
from robotic_simulations.environments.environment2 import Environment as Env2
from robotic_simulations.robots import Robot, StickRobot
import argparse

class RobotSimulator(Node):
    def __init__(self, dt=0.05, robot_type=1, environment_type=1, obs_num=20):
        super().__init__('robot_simulator')
        self.dt = dt
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(self.dt, self.update_simulation)
        self.shutdown_called = False  # Flag to ensure shutdown is called only once

        # Initialize the environment
        self.environment_type = environment_type
        if self.environment_type == 1:
            self.env = Env1([1 for _ in range(obs_num)])  # Set number of obstacles
            self.env_limit = 20
            # Spawn a robot at random position
            x, y = 0, 0
            while True:
                x = np.random.uniform(-self.env_limit, self.env_limit)
                y = np.random.uniform(-self.env_limit, self.env_limit)
                if not self.env.is_collision(x, y, 2):
                    break
        else:
            self.env = Env2()
            self.env_limit = 5
            x, y = self.env.generate_start_position(1.1)

        self.robot_type = robot_type
        self.robot_length = 1.5
        if self.robot_type == 1:
            self.robot = Robot(x, y)
            self.path = [[self.robot.x, self.robot.y]]
            self.input_hist = [[0, 0]]
        else:
            if self.environment_type == 1:
                theta = np.random.uniform(0, np.pi)
            else:
                aux = np.arccos(min(abs(x - self.env.start_zone[0][0]), self.env.start_zone[0][1] - x) / self.robot_length)
                theta = np.random.uniform(aux + 0.1, np.pi - aux - 0.1)
            self.robot = StickRobot(x, y, theta, self.robot_length)
            self.path = [[self.robot.x, self.robot.y, self.robot.theta]]
            self.input_hist = [[0, 0, 0]]

        self.step_count = 0

        # Draw obstacles as blue markers
        if self.environment_type == 1:
            for i, obstacle in enumerate(self.env.obstacles):
                obstacle_marker = self.create_marker(i + 200, float(obstacle[1]), float(obstacle[2]), 0.0, 0.0, 1.0, scale=float(obstacle[0]), shape=Marker.CYLINDER)
                self.publisher.publish(obstacle_marker)
        else:
            for i, wall in enumerate(self.env.boundaries):
                wall_marker = self.create_wall_marker(i + 300, wall[0], wall[1], 0.0, 0.0, 1.0)
                self.publisher.publish(wall_marker)
            # Add goal marker
            goal_marker = self.create_marker(400, self.env.goal_position, 0, 0.0, 1.0, 0.0, scale=0.5, shape=Marker.CYLINDER)
            self.publisher.publish(goal_marker)

    def update_simulation(self):
        if self.step_count >= 5000:
            if rclpy.ok() and not self.shutdown_called:
                self.shutdown_called = True
                rclpy.shutdown()
            return

        if self.environment_type == 1:
            if self.robot_type == 1:
                self.robot.apf_update(self.dt, self.env, k=1, k_rep=1.8, gamma=2, rho=1.5, eta_0=6, nu=0.8)
                self.path.append([self.robot.x, self.robot.y])
                self.input_hist.append([self.robot.u_x, self.robot.u_y])
            else:
                self.robot.apf_update(self.dt, self.env, k=1, k_rep=2, gamma=2, rho=2, eta_0=6, nu=0.6+1e-4*self.step_count, mul_theta_eta=3, mul_theta_k=4, tail_orientation=-1)
                self.path.append([self.robot.x, self.robot.y, self.robot.theta])
                self.input_hist.append([self.robot.u_x, self.robot.u_y, self.robot.u_theta])
            if np.sqrt(self.path[-1][0]**2 + self.path[-1][1]**2) < 1e-1:
                if rclpy.ok() and not self.shutdown_called:
                    self.shutdown_called = True
                    rclpy.shutdown()
                return
        else:
            if self.robot_type == 1:
                self.robot.apf_update(self.dt, self.env, k=0.6, k_rep=1.5, gamma=2, rho=1, eta_0=0.65, nu=0)
                self.path.append([self.robot.x, self.robot.y])
                self.input_hist.append([self.robot.u_x, self.robot.u_y])
            else:
                self.robot.apf_update(self.dt, self.env, k=0.6, k_rep=0.5, gamma=2, rho=1, eta_0=0.5, nu=0, mul_theta_k=4, mul_theta_eta=2, tail_orientation=0)
                self.path.append([self.robot.x, self.robot.y, self.robot.theta])
                self.input_hist.append([self.robot.u_x, self.robot.u_y, self.robot.u_theta])
            if np.abs(self.path[-1][0] - self.env.goal_position) < 1e-1:
                if rclpy.ok() and not self.shutdown_called:
                    self.shutdown_called = True
                    rclpy.shutdown()
                return
        self.step_count += 1
        if self.step_count % 5 == 0:
            self.publish_markers()

    def publish_markers(self):
        # Publish robot marker
        robot_marker = self.create_marker(99, self.robot.x, self.robot.y, 1.0, 0.65, 0.0, float(self.robot.diameter), Marker.SPHERE)
        self.publisher.publish(robot_marker)

        # Publish velocity arrow
        vel_marker = self.create_arrow_marker(100, float(self.robot.x), float(self.robot.y), float(self.robot.u_x), float(self.robot.u_y))
        self.publisher.publish(vel_marker)

        # Publish path
        path_marker = self.create_path_marker(101, self.path)
        self.publisher.publish(path_marker)

    def create_marker(self, marker_id, x, y, r, g, b, scale=0.5, shape=Marker.SPHERE):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "visualization"
        marker.id = marker_id
        marker.type = shape
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        if self.robot_type == 2 and shape == Marker.SPHERE:
            # Draw the stick robot as a line strip
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.1  # Line width
            start_point = Point()
            start_point.x = -self.robot.length * np.cos(self.robot.theta)
            start_point.y = -self.robot.length * np.sin(self.robot.theta)
            end_point = Point()
            marker.points.append(start_point)
            marker.points.append(end_point)

        return marker

    def create_arrow_marker(self, marker_id, x, y, vel_x, vel_y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "velocity"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        scale_factor = 1.5

        # Define arrow start and end points
        start_point = Point()
        end_point = Point()
        end_point.x = vel_x * scale_factor
        end_point.y = vel_y * scale_factor

        marker.points.append(start_point)
        marker.points.append(end_point)

        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1  # Arrowhead diameter
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green for velocity
        marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    def create_path_marker(self, marker_id, path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        for point in path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            marker.points.append(p)

        return marker

    def create_wall_marker(self, marker_id, start, end, r, g, b):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "walls"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Calculate the midpoint of the wall to position the marker
        midpoint_x = (start[0] + end[0]) / 2.0
        midpoint_y = (start[1] + end[1]) / 2.0

        marker.pose.position.x = float(midpoint_x)
        marker.pose.position.y = float(midpoint_y)
        marker.pose.position.z = 0.5  # Set the height of the wall

        # Calculate the orientation of the wall
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        yaw = np.arctan2(dy, dx)
        quat = self.euler_to_quaternion(0, 0, yaw)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # Set the scale of the wall
        length = np.sqrt(dx**2 + dy**2)
        marker.scale.x = length
        marker.scale.y = 0.1  # Wall thickness
        marker.scale.z = 0.6  # Wall height

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        return marker

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Robot Simulator')
    parser.add_argument('--robot_type', '-r', type=int, default=1, help='Type of robot (1 for Robot, 2 for StickRobot)', choices=[1, 2])
    parser.add_argument('--environment_type', '-e', type=int, default=1, help='Type of environment (1 for Env1, 2 for Env2)', choices=[1, 2])
    parser.add_argument('--obs_num', '-o', type=int, default=4, help='Number of obstacles in the environment')
    parsed_args = parser.parse_args()

    node = RobotSimulator(dt=0.01, robot_type=parsed_args.robot_type, environment_type=parsed_args.environment_type, obs_num=parsed_args.obs_num)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()