import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class NMPCObstacleAvoidance(Node):
    def __init__(self):
        super().__init__("nmpc_obstacle_avoidance")

        self.subscription = self.create_subscription(
            PoseArray, "/cone_positions", self.cone_callback, 10)
        self.trajectory_publisher = self.create_publisher(Path, "/nmpc_trajectory", 10)

        self.obstacles = []
        self.goal = np.array([5.0, 5.0])  # Obiettivo finale
        self.k_att = 1.0  # Coefficiente attrattivo
        self.k_rep = 5.0  # Coefficiente repulsivo
        self.rep_radius = 2.0  # Raggio di influenza degli ostacoli

        self.setup_nmpc()
        
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def setup_nmpc(self):
        self.N = 20  # Orizzonte di predizione
        self.dt = 0.1

        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")

        state = ca.vertcat(x, y, theta)
        control = ca.vertcat(v, omega)

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        self.f = ca.Function("f", [state, control], [rhs])

        self.X = ca.SX.sym("X", 3, self.N+1)
        self.U = ca.SX.sym("U", 2, self.N)

        self.cost = 0
        self.constraints = []

        for k in range(self.N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            print("Tipo di xk:", type(xk))
            print("Tipo di uk:", type(uk))

            x_next = self.X[:, k+1]
            
            self.cost += ca.mtimes(uk.T, uk)
            self.cost += self.potential_field_cost(xk[:2])
            
            xk_next = xk + self.dt * self.f(xk.reshape((-1, 1)), uk.reshape((-1, 1)))
            self.constraints.append(x_next - xk_next)

        self.g = ca.vertcat(*self.constraints)
        self.nlp = {"x": ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1))),
                    "f": self.cost, "g": self.g}
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp)

    def potential_field_cost(self, pos):
        att = self.k_att * ca.norm_2(pos - self.goal)
        rep = 0
        for obs in self.obstacles:
            dist = ca.norm_2(pos - ca.DM(obs))
            if dist < self.rep_radius:
                rep += self.k_rep * (1/dist - 1/self.rep_radius)**2
        return att + rep

    def cone_callback(self, msg):
        self.obstacles = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.compute_nmpc()
        
    def compute_nmpc(self):
        if not self.obstacles:
            self.get_logger().warn("Nessun ostacolo rilevato. NMPC non aggiornato.")
            return

        x0 = np.array([0.0, 0.0, 0.0])
        X_guess = np.zeros((3, self.N+1))
        U_guess = np.zeros((2, self.N))

        # Flatten the initial guess
        x0_flat = np.concatenate([X_guess.flatten(), U_guess.flatten()])

        # Prepare the arguments for the solver
        args = {
            "lbg": np.zeros(self.g.shape[0]),  # Lower bounds for constraints
            "ubg": np.zeros(self.g.shape[0]),  # Upper bounds for constraints
            "x0": x0_flat  # Initial guess
        }

        # Call the solver
        res = self.solver(**args)
        sol = res["x"].full().flatten()
        X_sol = sol[:3*(self.N+1)].reshape((3, self.N+1))

        self.publish_trajectory(X_sol)
        self.update_plot(X_sol)

    def publish_trajectory(self, X_sol):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for k in range(self.N+1):
            pose = PoseStamped()
            pose.pose.position.x = X_sol[0, k]
            pose.pose.position.y = X_sol[1, k]
            path_msg.poses.append(pose)

        self.trajectory_publisher.publish(path_msg)
        self.get_logger().info("Traiettoria NMPC pubblicata!")
        
    def update_plot(self, X_sol):
        """Aggiorna il plot con la nuova traiettoria e gli ostacoli"""
        self.ax.clear()
        
        # Plotta la traiettoria
        self.ax.plot(X_sol[0, :], X_sol[1, :], '-b', label='Traiettoria NMPC')
        
        # Plotta la posizione del robot
        self.ax.plot(0, 0, 'go', label='Veicolo')
        
        # Plotta l'obiettivo finale
        self.ax.plot(self.goal[0], self.goal[1], 'ro', label='Obiettivo')

        # Plotta gli ostacoli con il loro campo di influenza
        for obs in self.obstacles:
            self.ax.plot(obs[0], obs[1], 'rx', label='Ostacolo')
            circle = plt.Circle(obs, self.rep_radius, color='r', fill=False, linestyle='dashed')
            self.ax.add_patch(circle)
        
        self.ax.set_xlim(-3, 7)
        self.ax.set_ylim(-3, 7)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Traiettoria NMPC con Campo di Potenziale")
        self.ax.legend()
        
        plt.draw()
        plt.pause(0.1)

        # Mostra il plot in modo bloccante alla fine dell'esecuzione
        plt.show(block=True)  


def main(args=None):
    rclpy.init(args=args)
    node = NMPCObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()