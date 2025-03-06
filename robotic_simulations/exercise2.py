import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from robotic_simulations.environments.environment3 import Environment as Env
from robotic_simulations.robots import Robot, StickRobot
import argparse

class RobotSimulator(Node):
    def __init__(self, dt=0.05, robot_type=1, obs_num=20, sigma=5):
        super().__init__('robot_simulator')
        self.dt = dt
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(self.dt, self.update_simulation)
        self.shutdown_called = False  # Flag to ensure shutdown is called only once

        # Initialize the environment
        self.env = Env([1 for _ in range(obs_num)], sigma=sigma)  # Set number of obstacles
        self.env_limit = 20
        # Spawn a robot at random position
        x, y = 0, 0
        while True:
            x = np.random.uniform(-self.env_limit // 2, self.env_limit // 2)
            y = np.random.uniform(-self.env_limit // 2, self.env_limit // 2)
            if not self.env.is_collision(x, y, 2):
                break

        self.robot_type = robot_type
        self.robot_length = 1.5
        if self.robot_type == 1:
            self.robot = Robot(x, y)
            self.path = [[self.robot.x, self.robot.y]]
            self.input_hist = [[0, 0]]
        else:
            theta = np.random.uniform(0, np.pi / 2)
            self.robot = StickRobot(x, y, theta, self.robot_length)
            self.path = [[self.robot.x, self.robot.y, self.robot.theta]]
            self.input_hist = [[0, 0, 0]]

        self.step_count = 0
        self.grad_hist = [[0, 0]]


        # Draw obstacles as blue markers
        for i, obstacle in enumerate(self.env.obstacles):
            obstacle_marker = self.create_marker(i + 200, float(obstacle[1]), float(obstacle[2]), 0.0, 0.0, 1.0, scale=float(obstacle[0]), shape=Marker.CYLINDER)
            self.publisher.publish(obstacle_marker)

    def update_simulation(self):
        if self.step_count >= 1500:
            if rclpy.ok() and not self.shutdown_called:
                self.shutdown_called = True
                rclpy.shutdown()
            return

        if self.robot_type == 1:
            x, y = self.robot.bb_grad_descent(self.env)
            self.robot.apf_update(self.dt, self.env, goal=[x, y], k=0.4, k_rep=1.6, gamma=2, rho=2, eta_0=6, nu=0.8)
            self.path.append([self.robot.x, self.robot.y])
            self.grad_hist.append(self.robot.prev_dh)
            self.input_hist.append([self.robot.u_x, self.robot.u_y])
        else:
            x, y = self.robot.bb_grad_descent(self.env)
            self.robot.apf_update(self.dt, self.env, goal=[x, y], k=0.5, k_rep=1, gamma=2, rho=2, eta_0=3, nu=0.8, mul_theta=2, tail_orientation=-1)
            self.path.append([self.robot.x, self.robot.y, self.robot.theta])
            self.grad_hist.append(self.robot.prev_dh)
            self.input_hist.append([self.robot.u_x, self.robot.u_y, self.robot.u_theta])
            
        if self.robot.prev_dh is not None:
            if np.sqrt(self.robot.prev_dh[0]**2 + self.robot.prev_dh[1]**2) < 1e-3:
                if rclpy.ok() and not self.shutdown_called:
                    self.shutdown_called = True
                    rclpy.shutdown()
                return

        self.publish_markers()
        self.step_count += 1

    def publish_markers(self):
        # Publish robot marker
        robot_marker = self.create_marker(99, self.robot.x, self.robot.y, 1.0, 0.65, 0.0, float(self.robot.diameter), Marker.SPHERE)
        self.publisher.publish(robot_marker)

        # Publish velocity arrow
        vel_marker = self.create_arrow_marker(100, float(self.robot.x), float(self.robot.y), float(self.robot.u_x), float(self.robot.u_y))
        self.publisher.publish(vel_marker)
        
        # Publish gradient arrow
        grad_marker = self.create_arrow_marker(101, float(self.robot.x), float(self.robot.y), float(self.robot.prev_dh[0]), float(self.robot.prev_dh[1]), 0.0, 1.0, 1.0, 15)
        self.publisher.publish(grad_marker)

        # Publish path
        path_marker = self.create_path_marker(102, self.path)
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
            marker.scale.x = 0.15  # Line width
            start_point = Point()
            start_point.x = -self.robot.length * np.cos(self.robot.theta)
            start_point.y = -self.robot.length * np.sin(self.robot.theta)
            end_point = Point()
            marker.points.append(start_point)
            marker.points.append(end_point)

        return marker
        
    def create_arrow_marker(self, marker_id, x, y, vel_x, vel_y, r=0.0, g=1.0, b=0.0, scale=1):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "velocity"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y

        # Define arrow start and end points
        start_point = Point()
        end_point = Point()
        end_point.x = vel_x * scale
        end_point.y = vel_y * scale

        marker.points.append(start_point)
        marker.points.append(end_point)

        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1  # Arrowhead diameter
        marker.color.r = r
        marker.color.g = g 
        marker.color.b = b
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

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Robot Simulator')
    parser.add_argument('--robot_type', '-r', type=int, default=1, help='Type of robot (1 for Robot, 2 for StickRobot)', choices=[1, 2])
    parser.add_argument('--obs_num', '-o', type=int, default=4, help='Number of obstacles in the environment')
    parser.add_argument('--sigma', '-s', type=float, default=5, help='Sigma value for the environment')
    parsed_args = parser.parse_args()

    node = RobotSimulator(dt=0.05, robot_type=parsed_args.robot_type, obs_num=parsed_args.obs_num)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()