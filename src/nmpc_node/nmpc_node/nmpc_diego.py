import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
import casadi as ca
import numpy as np

class NMPCObstacleAvoidance(Node):
    def __init__(self):
        super().__init__("nmpc_obstacle_avoidance")

        # Sottoscrizione al topic delle posizioni dei coni
        self.subscription = self.create_subscription(
            PoseArray, "/cone_positions", self.cone_callback, 10)

        # Publisher per la traiettoria calcolata
        self.trajectory_publisher = self.create_publisher(Path, "/nmpc_trajectory", 10)

        # Inizializza le posizioni degli ostacoli
        self.obstacles = []

        # Configura il problema NMPC con CasADi
        self.setup_nmpc()

    def setup_nmpc(self):
        """Inizializza il problema NMPC con CasADi"""
        self.N = 20  # Orizzonte di predizione
        self.dt = 0.1  # Tempo di campionamento

        # Definizione degli stati del robot
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")

        state = ca.vertcat(x, y, theta)
        control = ca.vertcat(v, omega)

        # Modello cinematico del robot
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        self.f = ca.Function("f", [state, control], [rhs])

        # Variabili di ottimizzazione
        self.X = ca.SX.sym("X", 3, self.N+1)
        self.U = ca.SX.sym("U", 2, self.N)

        # Funzione costo e vincoli
        self.cost = 0
        self.constraints = []

        for k in range(self.N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            x_next = self.X[:, k+1]

            # Termini di costo: minimizzazione del controllo
            self.cost += ca.mtimes(uk.T, uk)

            # Vincolo di dinamica del sistema
            xk_next = xk + self.dt * self.f(xk, uk)
            self.constraints.append(x_next - xk_next)

        self.g = ca.vertcat(*self.constraints)

        # Definizione del problema di ottimizzazione
        self.nlp = {"x": ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1))),
                    "f": self.cost, "g": self.g}

        self.solver = ca.nlpsol("solver", "ipopt", self.nlp)

    def cone_callback(self, msg):
        """Riceve le coordinate dei coni e aggiorna gli ostacoli"""
        self.obstacles = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.compute_nmpc()

    def compute_nmpc(self):
        """Esegue l'NMPC calcolando la traiettoria ottimale"""
        if not self.obstacles:
            self.get_logger().warn("Nessun ostacolo rilevato. NMPC non aggiornato.")
            return

        # Condizioni iniziali
        x0 = np.array([0.0, 0.0, 0.0])  # Stato iniziale (x, y, theta)
        X_guess = np.zeros((3, self.N+1))
        U_guess = np.zeros((2, self.N))

        # Definizione dei limiti
        args = {"lbg": np.zeros(self.g.shape[0]), "ubg": np.zeros(self.g.shape[0]),
                "x0": np.concatenate([X_guess.flatten(), U_guess.flatten()])}

        # Risoluzione dell'ottimizzazione NMPC
        res = self.solver(args)
        sol = res["x"].full().flatten()
        X_sol = sol[:3*(self.N+1)].reshape((3, self.N+1))

        # Pubblica la traiettoria su ROS2
        self.publish_trajectory(X_sol)

    def publish_trajectory(self, X_sol):
        """Pubblica la traiettoria calcolata su /nmpc_trajectory"""
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

def main(args=None):
    rclpy.init(args=args)
    node = NMPCObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
