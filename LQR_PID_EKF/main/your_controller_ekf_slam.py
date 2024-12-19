# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM
from scipy.linalg import expm, block_diag
from scipy.signal import StateSpace, lsim, dlsim
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.
        self.cumulative_error = 0
        self.previous_error = 0

    def calc_PID_input(self, current_error, kp,ki,kd, delta_T):
        self.cumulative_error += current_error*delta_T
        differential_error = (current_error-self.previous_error)/delta_T
        self.previous_error = current_error
        pid_input = kp*current_error + kd*differential_error + ki*self.cumulative_error
        return pid_input   
    
    def LQR(self, A, B, Q, R):
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def C2D(self, A, B, deltaT):
    # Create the augmented matrix for zero-order hold
        n = A.shape[0]
        Z = np.zeros((n, B.shape[1]))
        M = np.block([[A, B], [Z.T, np.zeros((B.shape[1], B.shape[1]))]])
        expM = expm(M * deltaT)
        Ad = expM[:n, :n]
        Bd = expM[:n, n:]
        return Ad, Bd
    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # You are free to reuse or refine your code from P3 in the spaces below.

        """ Time Horizon distance(Looking ahead logic)"""
        time_horizon_steps = 150
        closest_dist, closest_index = closestNode(X,Y,trajectory)

        """Desired Values"""
        try:
            X_desired = trajectory[closest_index + time_horizon_steps, 0]
            Y_desired = trajectory[closest_index + time_horizon_steps, 1]
            psi_desired = np.arctan2(Y_desired - Y, X_desired - X)
        except IndexError:
            # Fallback to the last point in the trajectory if out of range
            X_desired = trajectory[-1, 0]
            Y_desired = trajectory[-1, 1]
            psi_desired = np.arctan2(Y_desired - Y, X_desired - X)


        x_velocity=12 #desired velocity

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca / m, -(2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, -(2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        R=50
        Q=np.array([[1,0,0,0], [0,0.1,0,0], [0,0,0.1,0], [0,0,0,0.01]])
        C = np.eye(4)
        D = np.zeros((4,1))
        Ad, Bd = self.C2D(A, B, delT)
        sys_ct = StateSpace(A, B, C, D)
        sys_dt= sys_ct.to_discrete(delT)
        dt_A = sys_dt.A
        dt_B = sys_dt.B
        K = self.LQR(Ad,Bd,Q,R)
        #K = K.gain_matrix

        # Calculate errors
        phi = np.arctan2(trajectory[closest_index, 1] - Y_desired, trajectory[closest_index, 0] - X_desired)
        e1 = np.dot([-np.sin(phi),np.cos(phi)], [trajectory[closest_index, 0] - X, trajectory[closest_index, 1] - Y])
        e2 = wrapToPi(psi - psi_desired)
        e1_dot = ydot * np.cos(psi_desired-psi) - xdot * np.sin(psi_desired-psi)  # Projected rate of change along the trajectory  
        e2_dot = psidot

        # Construct error vector and compute control input
        e = np.array([e1, e1_dot, e2, e2_dot])
        delta = -np.dot(K, e)
        delta = float(delta)
        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        velocity_error= x_velocity - xdot
        F = self.calc_PID_input(velocity_error,200,10,30,delT)
        F = clamp(F,0,15736)

        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
