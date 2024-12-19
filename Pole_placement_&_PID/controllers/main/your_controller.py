# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
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

        # Add additional member variables according to your need here.
        self.cumulative_error = 0
        self.previous_error = 0

    def calc_PID_input(self, current_error, kp,ki,kd, delta_T):
        self.cumulative_error += current_error*delta_T
        differential_error = (current_error-self.previous_error)/delta_T
        self.previous_error = current_error
        pid_input = kp*current_error + kd*differential_error + ki*self.cumulative_error
        return pid_input    
    
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        #delT = 0.032
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        #F = 1000
        #delta = 0

        """ Time Horizon distance(Looking ahead logic)"""
        time_horizon_steps = 20
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





        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.

        """
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca / m, -(2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, -(2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        P = np.array([-10, -5, -1, 0])
        K  = signal.place_poles(A,B,P)
        K = K.gain_matrix

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
        position_error = np.power(np.power(X_desired-X,2) + np.power(Y_desired-Y,2),0.5)
        velocity_error= position_error/delT
        F = self.calc_PID_input(velocity_error,5,0,0,delT)
        F = clamp(F,0,15736)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
