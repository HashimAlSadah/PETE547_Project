import numpy as np 

#The central finite difference for second derivative with respect to x
def cent_diff_2nd_der_x(grid, x_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[1:-1, 2:]
        -
        2 * grid[1:-1, 1:-1]
        +
        grid[1:-1, :-2]
    ) / x_step_size**2
    return output

#The central finite difference for second derivative with respect to y
def cent_diff_2nd_der_y(grid, y_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[2:, 1:-1]
        -
        2 * grid[1:-1, 1:-1]
        +
        grid[:-2, 1:-1]
    ) / y_step_size**2
    return output

#The centeral fintie difference for the first derivative with respect to x
def cent_diff_1st_der_x(grid, x_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[1:-1, 2:]
        -
        grid[1:-1, :-2]
    ) / ( 2 * x_step_size )
    return output

#The centeral fintie difference for the first derivative with respect to y
def cent_diff_1st_der_y(grid, y_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[2:, 1:-1]
        -
        grid[:-2, 1:-1]
    ) / (2 * y_step_size)
    return output

#The backwaed difference for the derivative with repsect to x
def backward_diff_der_x(grid, x_step_size):
    output = np.copy(grid)
    output[1:-1, 1:] = (
        grid[1:-1, 1:]
        -
        grid[1:-1, :-1]
    ) / x_step_size
    return output

#The backwaed difference for the derivative with repsect to x
def backward_diff_der_y(grid, y_step_size):
    output = np.copy(grid)
    output[:-1, 1:-1] = (
        grid[1:, 1:-1]
        -
        grid[:-1, 1:-1]
    ) / y_step_size
    return output

def pressure_diff_x(grid, x_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[1:-1, 2:]
        +
        grid[1:-1, :-2]
    ) / x_step_size**2
    return output

def pressure_diff_y(grid, y_step_size):
    output = np.copy(grid)
    output[1:-1, 1:-1] = (
        grid[2:, 1:-1]
        +
        grid[:-2, 1:-1]
    )/ y_step_size**2
    return output

def u_boundary_condition(grid, horizentola_vel):
  grid[:, 0] = 0.0
  grid[:, -1] = 0.0
  grid[0, :] = 0.0
  grid[-1, :] = horizentola_vel

#apply boundary condition to y-component of the velocity
def v_boundary_condition(grid):
  grid[:, 0] = 0.0
  grid[:, -1] = 0.0
  grid[0, :] = 0.0
  grid[-1, :] = 0.0

#Apply pressure boundary condition
def p_boundary_condition(grid):
  grid[:, -1] = grid[:, -2]
  grid[:, 0] = grid[:, 1]
  grid[0, :] = grid[1, :]
  grid[-1, :] = 0.0
