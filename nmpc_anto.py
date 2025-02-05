import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
import casadi as ca
import do_mpc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class NMPC_APF(Node):
    def __init__(self):
        super().__init__('nmpc_apf_node')

        # Initialize ROS2 publishers and subscribers
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10) # Publish control commands
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) # Subscribe to robot odometry

        # Vehicle parameters
        self.wheelbase = 0.65 # Distance between front and rear axles
        self.max_steering_angle = np.pi/5 # Maximum steering angle (36 degrees)
        self.max_velocity = 3.0 # Maximum linear velocity

        # Define goal position
        self.goal_x = 50.0 # Goal x position
        self.goal_y = 0.0 # Goal y position
 
        # Define obstacles and safety parameters
        self.obstacles = [(11.0, 0.0), (22.0, 5.0), (16.0, 0.0), (3.0, 1.0)] # Obstacle positions
        self.obstacles = [(5.0, 0.0), (10.0, 5.0), (10.0, -5.0), (15.0, 0.0), (20.0, 5.0), (20.0, -5.0), (25.0, 0.0), (30.0, 5.0), (30.0, -5.0), (35.0, 0.0), (40.0, 5.0), (40.0, -5.0), ] # Obstacle positions

        self.d_safe = 1.5   
        self.d_influence = 2.0

        # State variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # MPC Parameters
        self.N = 5
        self.dt = 0.05
        self.nmpc = self.setup_nmpc()

        # Add visualization setup
        self.setup_visualization()
        self.path_history = {'x': [], 'y': []}
        
        self.timer = self.create_timer(self.dt, self.control_loop)

    def setup_visualization(self):
        """Initialize the matplotlib visualization"""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-5, 60)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Plot goal position
        self.goal_marker = self.ax.plot(self.goal_x, self.goal_y, 'g*', markersize=15, label='Goal')[0]
        
        # Plot obstacles
        self.obstacle_plots = []
        for obs_x, obs_y in self.obstacles:
            # Plot obstacle and its influence radius
            obstacle = Circle((obs_x, obs_y), self.d_safe, color='red', alpha=0.5, label='Obstacle')
            self.ax.add_patch(obstacle)
            self.obstacle_plots.extend([obstacle])
        
        # Initialize robot visualization
        robot_size = 0.5
        self.robot_body = Rectangle((self.x - robot_size/2, self.y - robot_size/2), 
                                  robot_size, robot_size, 
                                  color='blue', alpha=0.7, label='Robot')
        self.ax.add_patch(self.robot_body)
        
        # Initialize path history
        self.path_line, = self.ax.plot([], [], 'b-', alpha=0.5, label='Path')
        
        # Add legend
        self.ax.legend()
        
        plt.title('NMPC-APF Navigation Visualization')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')

    def update_visualization(self):
        """Update the visualization with current state"""
        # Update robot position and orientation
        robot_size = 0.5
        self.robot_body.set_xy((self.x - robot_size/2, self.y - robot_size/2))
        trans = plt.matplotlib.transforms.Affine2D().rotate_around(self.x, self.y, self.yaw) + self.ax.transData
        self.robot_body.set_transform(trans)
        
        # Update path history
        self.path_history['x'].append(self.x)
        self.path_history['y'].append(self.y)
        self.path_line.set_data(self.path_history['x'], self.path_history['y'])
        
        # Draw force vectors if needed (optional)
        f_x, f_y = self.compute_apf_forces()
        force_scale = 0.5  # Scale factor for force visualization
        if hasattr(self, 'force_arrow'):
            self.force_arrow.remove()
        self.force_arrow = self.ax.arrow(self.x, self.y, f_x * force_scale, f_y * force_scale,
                                       head_width=0.2, head_length=0.3, fc='g', ec='g', alpha=0.5)
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setup_nmpc(self):
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Define state variables
        x = model.set_variable('_x', 'x')
        y = model.set_variable('_x', 'y')
        yaw = model.set_variable('_x', 'yaw')

        # Define control variables
        v = model.set_variable('_u', 'v')
        delta = model.set_variable('_u', 'delta')

        #Define model equations
        x_dot = v * ca.cos(yaw)
        y_dot = v * ca.sin(yaw)
        yaw_dot = v * ca.tan(delta) / self.wheelbase

        model.set_rhs('x', x_dot)
        model.set_rhs('y', y_dot)
        model.set_rhs('yaw', yaw_dot)

        model.setup()

        # Create MPC Controller
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.N,
            't_step': self.dt,
            'n_robust': 1,
            'store_full_solution': True,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
        }
        mpc.set_param(**setup_mpc)

        # Reference Trajectory from APF
        f_x, f_y = self.compute_apf_forces()
        ref_yaw = ca.atan2(f_y, f_x)  # Desired APF heading direction

        # Running Cost (lterm) - Encourage following the APF trajectory
        weight_position = 10.0   # Weight for position error
        weight_control = 0.5    # Weight for control effort (smoother changes)
        weight_heading = 5.0    # Encourage alignment with APF direction
        #for obs_x, obs_y in self.obstacles:
        #    d = np.sqrt((self.x - obs_x)**2 + (self.y - obs_y)**2)
        #    if d < self.d_safe and d>0:
        #        weight_heading = 0.0
        #    else:
        #        weight_heading = 0.0

        lterm = (weight_position * ((x - self.goal_x)**2 + (y - self.goal_y)**2) +
                weight_control * (v**2 + delta**2) +
                weight_heading * (yaw - ref_yaw)**2)

        # Terminal Cost (mterm) - Encourage reaching the goal
        mterm = 20.0 * (((x - self.goal_x)**2 + (y - self.goal_y)**2) + (yaw - ref_yaw)**2)
        mpc.set_rterm(v=1, delta=1) # Penalizes rapid velocity and steering changes
        mpc.set_objective(mterm=mterm, lterm=lterm)

        self.get_logger().info(f'MTERM: {mterm}, LTERM: {lterm}')

        # Define constraints
        # Lower bounds on states:
        mpc.bounds['lower', '_x', 'x'] = 0
        mpc.bounds['lower', '_x', 'y'] = -10
        #mpc.bounds['lower', '_x', 'yaw'] = -ca.pi/4
        # Upper bounds on states:
        mpc.bounds['upper', '_x', 'x'] = 100
        mpc.bounds['upper', '_x', 'y'] = 10
        #mpc.bounds['upper', '_x', 'yaw'] = ca.pi/4

        # Lower bounds on control inputs: 
        mpc.bounds['lower', '_u', 'v'] = 0.0
        mpc.bounds['lower', '_u', 'delta'] = -self.max_steering_angle
        # Upper bounds on control inputs:
        mpc.bounds['upper', '_u', 'v'] = self.max_velocity
        mpc.bounds['upper', '_u', 'delta'] = self.max_steering_angle

        mpc.setup()
        return mpc

    def compute_apf_forces(self):
        """Compute the attractive and repulsive forces with enhanced obstacle avoidance"""
        # Attractive Force parameters
        K_att = 1.5
        f_att_x = K_att * (self.goal_x - self.x)
        f_att_y = K_att * (self.goal_y - self.y)

        # Repulsive Forces
        K_rep = 30.0
        f_rep_x, f_rep_y = 0.0, 0.0

        for obs_x, obs_y in self.obstacles:
            d = np.sqrt((self.x - obs_x)**2 + (self.y - obs_y)**2)
            if d < self.d_safe and d > 0:
                rep_factor = K_rep * (1.0/d - 1.0/self.d_safe)**2 * (1.0/d**2)
                f_rep_x += rep_factor * ((self.x - obs_x) / d)
                f_rep_y += rep_factor * ((self.y - obs_y) / d)
        
        f_x = f_rep_x + f_att_x
        f_y = f_rep_y + f_att_y

        self.get_logger().info(f'F_R: f_x={f_x:.2f}, f_y={f_y:.2f}')
        return f_x, f_y

    def control_loop(self):
        """Main control loop integrating nMPC and APF"""

        # Compute APF forces
        f_x, f_y = self.compute_apf_forces()
        
        # Compute desired heading from APF
        desired_yaw = np.arctan2(f_y, f_x)
        
        # Compute heading error
        yaw_error = desired_yaw - self.yaw
        heading_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        # Compute smooth steering angle based on heading error
        steering_angle_apf = np.clip(heading_error, -self.max_steering_angle, self.max_steering_angle)
        
        # Compute velocity considering obstacle influence
        force_magnitude = np.sqrt(f_x**2 + f_y**2)
        turning_factor = 1.0 - 0.8 * abs(steering_angle_apf) / self.max_steering_angle
        velocity_apf = self.max_velocity * turning_factor * min(1.0, force_magnitude)
        
        # Solve the nMPC
        x0 = np.array([self.x, self.y, self.yaw])
        self.nmpc.x0 = x0
        self.nmpc.u0 = np.array([1.0, 0.0])
        self.nmpc.set_initial_guess()
        
        u0 = self.nmpc.make_step(x0)
        v_mpc, delta_mpc = float(u0[0]), float(u0[1])
        
        # Blend APF and nMPC outputs
        alpha = 1.5  # Weighting factor for APF influence
        v_final = (1 - alpha) * v_mpc + alpha * velocity_apf
        delta_final = (1 - alpha) * delta_mpc + alpha * steering_angle_apf
        
        # Apply safety constraints
        v_final = np.clip(v_final, 0.1, self.max_velocity)
        delta_final = np.clip(delta_final, -self.max_steering_angle, self.max_steering_angle)

        # Publish control commands
        cmd = Twist()
        cmd.linear.x = float(v_final)
        cmd.angular.z = float(delta_final)
        self.pub_cmd.publish(cmd)

        self.get_logger().info(f'v={v_final:.2f}, delta={delta_final:.2f}')
        
        # Update visualization
        self.update_visualization()

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = np.arctan2(siny_cosp, cosy_cosp)
    
def main(args=None):
    rclpy.init(args=args)
    node = NMPC_APF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()