import numpy as np 
import matplotlib.pyplot as plt
from helpers_FD import *

class FD_CAV:
    def __init__(self, X_BOUNDARY = (0, 1), Y_BOUNDARY=(0, 1), T_STEP_SIZE = 0.001):
        self.X_BOUNDARY = X_BOUNDARY
        self.Y_BOUNDARY = Y_BOUNDARY
        self.T_STEP_SIZE = T_STEP_SIZE
        self.DENSITY = 1.0
        self.VISCOSITY = 0.1
        self.HORIZENTOL_VEL = 1.0
        self.x_num_points = 42
        self.y_num_points = 42

    def set_resolution(self, x_num_points, y_num_points):
        self.x_num_points = x_num_points
        self.y_num_points = y_num_points

    def set_time_step(self, value):
        self.T_STEP_SIZE = value

    def get_x_points(self):
        return np.linspace(*self.X_BOUNDARY, self.x_num_points)
    
    def get_y_points(self):
        return np.linspace(*self.Y_BOUNDARY, self.y_num_points)
    
    def set_viscosity(self, viscosity):
        self.VISCOSITY = viscosity
    
    def set_density(self, density):
        self.DENSITY = density
    
    def set_horizentol_vel(self, velocity):
        self.HORIZENTOL_VEL = velocity
    
    def initialize(self):
        u = np.zeros((self.y_num_points, self.x_num_points))
        v = np.zeros((self.y_num_points, self.x_num_points))
        p = np.zeros((self.y_num_points, self.x_num_points))
        u_boundary_condition(u, self.HORIZENTOL_VEL)
        v_boundary_condition(v)
        p_boundary_condition(p)
        return u, v, p

    def simulate(self, t_final=1):
        x, y = self.get_x_points(), self.get_y_points()
        dx = x[1] - x[0]  #x step size
        dy = y[1] - y[0]  #y step size
        dt = self.T_STEP_SIZE
        density = self.DENSITY 
        viscosity = self.VISCOSITY
        c = ( dx**2 * dy**2 * density ) / ( 2 * (dx**2 + dy**2) )
        
        u_prev, v_prev, p_prev = self.initialize()
        epochs = int( t_final / dt )

        for _ in range(epochs):
            #u_next
            u_next = u_prev + dt * (
                - u_prev * cent_diff_1st_der_x(u_prev, dx)
                - v_prev * cent_diff_1st_der_y(u_prev, dy)
                - (1/density) * cent_diff_1st_der_x(p_prev, dx)
                + viscosity * ( cent_diff_2nd_der_x(u_prev, dx) 
                + cent_diff_2nd_der_y(u_prev, dy) )
            )
            u_boundary_condition(u_next, self.HORIZENTOL_VEL)

            #v_next
            v_next = v_prev + dt * (
                - u_prev * cent_diff_1st_der_x(v_prev, dx)
                - v_prev * cent_diff_1st_der_y(v_prev, dy)
                - (1/density) * cent_diff_1st_der_y(p_prev, dy)
                + viscosity * ( cent_diff_2nd_der_x(v_prev, dx) 
                + cent_diff_2nd_der_y(v_prev, dy) )
            )
            v_boundary_condition(v_next)

            #update
            u_prev = u_next
            v_prev = v_next

            #First smooth out the pressure by iterating few times
            for _ in range(50):
                p_next = c * (
                    pressure_diff_x(p_prev, dx) + pressure_diff_y(p_prev, dy)
                    - density * (
                        (1/dt) * (cent_diff_1st_der_x(u_prev, dx) + cent_diff_1st_der_y(v_prev, dy) )
                        - cent_diff_1st_der_x(u_prev, dx)**2
                        - 2 * cent_diff_1st_der_y(u_prev, dy) * cent_diff_1st_der_x(v_prev, dx)
                        - cent_diff_1st_der_y(v_prev, dy)**2
                    )
                )
                p_boundary_condition(p_next)
                p_prev = p_next
        
        return u_next, v_next, p_next

    

      

