"""
TO-DO
---------------------------------------------------------------------------------------------------------
1) Isotropic
- Phase Grid
- Temperature Grid
- Update Temperature
- Update Phase
- Visualization
-

2) Anisotropic if above works
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

np.random.seed(1)


def init_phase_grid(nx, ny):
    """
    Initialize 2D grid of -1 and 1 values.
    :param nx: number of grid rows
    :param ny: number of grid columns
    :return: (nx, ny) grid
    """
    P = np.zeros((nx, ny))
    return P


def init_temp_grid(nx, ny):
    T = np.zeros((nx, ny))
    return T


def add_seed(P, seed_list):
    for seed in seed_list:
        P[seed[0], seed[1]] = 1


def initialize(nx, ny, seed_list):
    P = init_phase_grid(nx, ny)
    T = init_temp_grid(nx, ny)
    add_seed(P, seed_list)
    return P, T


def random_thermal_noise(size):
    random_noise = 0.5 - np.random.random((size[0], size[1]))
    return random_noise


def m_calc(T):
    ones = np.ones_like(T)
    M = alpha/np.pi*np.arctan(gamma*(ones-T))
    return M


def g_calc(P, M):
    random_noise = random_thermal_noise(P.shape)
    ones = np.ones_like(P)

    G = P*(ones-P)*(P-0.5*ones+M)+a*P*(ones-P)*random_noise
    return G


def theta_calc(P):
    P_padded = np.pad(P, ((1, 1), (1, 1)), mode="reflect")
    p_jMinus = np.roll(P_padded, 1, axis=0)
    p_jPlus = np.roll(P_padded, -1, axis=0)
    p_iMunus = np.roll(P_padded, 1, axis=1)
    p_iPlus = np.roll(P_padded, -1, axis=1)

    Theta = np.arctan((p_jPlus - p_jMinus)/(p_iPlus - p_iMunus))

    return Theta[1:-1, 1:-1]


def epsilon_calc(Theta):
    ones = np.ones_like(Theta)
    epsilon = ones*epsilon_avg + epsilon_avg * delta * np.cos(j * (Theta - ones*theta_0))
    epsilon_prime = -epsilon_avg * delta * j * np.sin(j * (Theta - ones*theta_0))
    epsilon_squared =   (ones*epsilon_avg**2 + 2 * epsilon_avg**2 * delta * np.cos(j * (Theta - ones*theta_0))
                        + epsilon_avg**2 * delta**2 * (np.cos(j * (Theta - ones*theta_0)))**2)

    return epsilon, epsilon_prime, epsilon_squared




def phase_update(P, T):
    M = m_calc(T)
    G = g_calc(P, M)
    stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])/dx**2
    P_new = P + epsilon**2*dt/tau*convolve(P, stencil, mode="reflect") + dt/tau*G
    return P_new


def temp_update(T, P, P_new):
    stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])/dx**2
    T_new = T + dt*convolve(T, stencil, mode="reflect") + k*(P_new - P)
    return T_new


def crystal_solve(P, T):
    for i in range(steps):
        P_new = phase_update(P, T)
        T_new = temp_update(T, P, P_new)
        P = P_new.copy()
        T = T_new.copy()
    print(P)
    return P


def crystal_plot(P):
    plt.imshow(P, cmap="coolwarm")
    plt.colorbar()
    plt.show()
    return 0


def crystal_visualize():
    return 0


def main():
    global alpha
    global gamma
    global epsilon
    global tau
    global a
    global k
    global dt
    global dx
    global delta
    global j
    global epsilon_avg
    global theta_0

    global steps

    alpha = 0.9     # element of interval (0, 1)
    gamma = 10      #
    epsilon = 0.01  # thickness of interface / motion of interface
    epsilon_mean = 0.01
    delta = 0.050
    tau = 0.0003    # time constant
    theta_0 = 0
    a = 0.01        # noise amplitude
    j = 4
    k = 1.4         # dimensionless latent heat
    dt = 0.0002     # timestep
    steps = 3000    # steps
    dx = 0.03       # grid size
    j = 4
    delta = 0.05
    epsilon_avg = 0.01
    theta_0 = 0

    nx = 300
    ny = 300

    seed_border = [[50, 200] for x in range(nx)]

    seed_list_middle = [[150, 150], [150, 151], [151, 151], [151, 150]]


    P, T = initialize(nx, ny, seed_border)
    P_solved = crystal_solve(P, T)
    crystal_plot(P_solved)


if __name__ == '__main__':
    main()

