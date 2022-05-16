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
from numba import jit

np.random.seed(1)


@jit(nopython=True)
def pad(M):
    M_new = np.zeros((nx+2, ny+2))

    M_new[1:-1, 1:-1] = M

    M_new[0, 1:-1] = M[1, :]
    M_new[-1, 1:-1] = M[-2, :]
    M_new[1:-1, 0] = M[:, 1]
    M_new[1:-1, -1] = M[:, -2]

    M_new[0, 0] = M[1, 1]
    M_new[0, -1] = M[1, -2]
    M_new[-1, 0] = M[-2, 1]
    M_new[-1, -1] = M[-2, -2]
    return M_new


@jit(nopython=True)
def roll(A, step, axis=0):
    if axis == 0:
        A_out = np.roll(A, step*(nx+2))
        return A_out
    else:
        A_out = np.roll(A.T, step*(ny+2))
        A_out = A_out.T
        return A_out


#@jit
def init_phase_grid(nx, ny):
    """
    Initialize 2D grid of -1 and 1 values.
    :param nx: number of grid rows
    :param ny: number of grid columns
    :return: (nx, ny) grid
    """
    P = np.zeros((nx, ny))
    return P


#@jit
def init_temp_grid(nx, ny):
    T = np.zeros((nx, ny))
    return T


#@jit
def add_seed(P, seed_list):
    for seed in seed_list:
        P[seed[0], seed[1]] = 1


#@jit
def initialize(nx, ny, seed_list):
    P = init_phase_grid(nx, ny)
    T = init_temp_grid(nx, ny)
    add_seed(P, seed_list)
    return P, T


@jit(nopython=True)
def random_thermal_noise(size):
    random_noise = 0.5 - np.random.random((size[0], size[1]))
    return random_noise


@jit(nopython=True)
def m_calc(T):
    Ones = np.ones_like(T)
    M = alpha/np.pi*np.arctan(gamma*(Ones-T))
    return M


@jit(nopython=True)
def g_calc(P, M):
    random_noise = random_thermal_noise(P.shape)
    Ones = np.ones_like(P)

    G = P*(Ones-P)*(P-0.5*Ones+M)+a*P*(Ones-P)*random_noise
    return G


@jit(nopython=True)
def theta_calc(P):
    P_padded = pad(P)
    P_dx = d_dx(P_padded)
    P_dy = d_dy(P_padded)
    P_iMinus = roll(P_padded, 1, axis=0)[1:-1, 1:-1]
    P_iPlus = roll(P_padded, -1, axis=0)[1:-1, 1:-1]
    P_jMinus = roll(P_padded, 1, axis=1)[1:-1, 1:-1]
    P_jPlus = roll(P_padded, -1, axis=1)[1:-1, 1:-1]

    Theta = np.empty_like(P)
    for m in range(Theta.shape[0]):
        for n in range(Theta.shape[1]):

            if P_dy[m, n] != 0:
                Theta[m, n] = np.arctan(P_dx[m, n]/P_dy[m, n])
            else:
                Theta[m, n] = 0
            """
            if P_dy[m,n] != 0 and P_dx[m,n] != 0:
                Theta[m, n] = np.arccos(P_dy[m, n]/np.sqrt((P_dy[m,n]**2+P_dx[m,n]**2)))
            else:
                Theta[m, n] = np.pi/2
            """
    return Theta


@jit(nopython=True)
def epsilon_calc(Theta):
    Ones = np.ones_like(Theta)
    Epsilon = Ones*epsilon_avg + epsilon_avg * delta * np.cos(j * (Theta - Ones*theta_0))
    Epsilon_prime = -epsilon_avg * delta * j * np.sin(j * (Theta - Ones*theta_0))
    Epsilon_squared = Epsilon**2

    return Epsilon, Epsilon_prime, Epsilon_squared


@jit(nopython=True)
def d_dx(N_padded):
    N_iPlus = roll(N_padded, -1, axis=0)[1:-1, 1:-1]
    N_iMinus = roll(N_padded, 1, axis=0)[1:-1, 1:-1]
    N_x = (N_iPlus - N_iMinus) / 2 / dx

    return N_x


@jit(nopython=True)
def d_dy(N_padded):
    N_jPlus = roll(N_padded, -1, axis=1)[1:-1, 1:-1]
    N_jMinus = roll(N_padded, 1, axis=1)[1:-1, 1:-1]
    N_y = (N_jPlus - N_jMinus) / 2 / dy

    return N_y


@jit(nopython=True)
def d2_dxdy(N_padded):
    N_jPlus = roll(N_padded, -1, axis=1)
    N_iPlus_jPlus = roll(N_jPlus, -1, axis=0)[1:-1, 1:-1]
    N_iMinus_jPlus = roll(N_jPlus, 1, axis=0)[1:-1, 1:-1]
    N_jMinus = roll(N_padded, 1, axis=1)
    N_iPlus_jMinus = roll(N_jMinus, -1, axis=0)[1:-1, 1:-1]
    N_iMinus_jMinus = roll(N_jMinus, 1, axis=0)[1:-1, 1:-1]

    N_dxdy = (N_iPlus_jPlus - N_iMinus_jPlus - N_iPlus_jMinus - N_iMinus_jMinus) / 4 / dx / dy

    return N_dxdy


@jit(nopython=True)
def d2_dxdx(N_padded):
    N_iPlus = roll(N_padded, -1, axis=0)[1:-1, 1:-1]
    N_iMinus = roll(N_padded, 1, axis=0)[1:-1, 1:-1]
    N = N_padded[1:-1, 1:-1]

    N_dxdx = (N_iPlus - 2 * N + N_iMinus) / dx**2

    return N_dxdx


@jit(nopython=True)
def d2_dydy(N_padded):
    N_jPlus = roll(N_padded, -1, axis=1)[1:-1, 1:-1]
    N_jMinus = roll(N_padded, 1, axis=1)[1:-1, 1:-1]
    N = N_padded[1:-1, 1:-1]

    N_dydy = (N_jPlus - 2 * N + N_jMinus) / dy ** 2

    return N_dydy


@jit(nopython=True)
def phase_update(P, T):
    M = m_calc(T)
    G = g_calc(P, M)
    Theta = theta_calc(P)
    Epsilon, Epsilon_prime, Epsilon_squared = epsilon_calc(Theta)

    Epsilon_padded = pad(Epsilon)
    Epsilon_prime_padded = pad(Epsilon_prime)
    Epsilon_squared_padded = pad(Epsilon_squared)
    P_padded = pad(P)

    P_dx = d_dx(P_padded)
    P_dy = d_dy(P_padded)
    P_dxdx = d2_dxdx(P_padded)
    P_dydy = d2_dydy(P_padded)
    P_dxdy = d2_dxdy(P_padded)

    Epsilon_dx = d_dx(Epsilon_padded)
    Epsilon_dy = d_dy(Epsilon_padded)

    Epsilon_prime_dx = d_dx(Epsilon_prime_padded)
    Epsilon_prime_dy = d_dy(Epsilon_prime_padded)

    Epsilon_squared_dx = d_dx(Epsilon_squared_padded)
    Epsilon_squared_dy = d_dy(Epsilon_squared_padded)

    DX = Epsilon*Epsilon_prime_dx*P_dy + Epsilon_prime*Epsilon_dx*P_dy + Epsilon*Epsilon_prime*P_dxdy
    DY = Epsilon*Epsilon_prime_dy*P_dx + Epsilon_prime*Epsilon_dy*P_dx + Epsilon*Epsilon_prime*P_dxdy
    GRAD = Epsilon_squared*P_dxdx + Epsilon_squared_dx*P_dx + Epsilon_squared*P_dydy + Epsilon_squared_dy*P_dy

    P_new = P + dt/tau*(-DX + DY + GRAD + G)
    return P_new


@jit(nopython=True)
def temp_update(T, P, P_new):
    T_padded = pad(T)
    T_iMinus = roll(T_padded, 1, 0)[1:-1, 1:-1]
    T_iPlus = roll(T_padded, -1, 0)[1:-1, 1:-1]
    T_jMinus = roll(T_padded, 1, 1)[1:-1, 1:-1]
    T_jPlus = roll(T_padded, -1, 1)[1:-1, 1:-1]
    T_new = T + dt*(T_iMinus+T_iPlus+T_jMinus+T_jPlus-4*T)/dx**2 + k*(P_new - P)
    return T_new


@jit(nopython=True)
def crystal_solve(P, T):
    for i in range(steps):
        P_new = phase_update(P, T)
        T_new = temp_update(T, P, P_new)
        P = P_new.copy()
        T = T_new.copy()
    return P


def crystal_plot(P):
    plt.imshow(P, cmap="binary")
    plt.colorbar()
    plt.show()
    return 0


def crystal_visualize():
    return 0


#@jit
def main():
    global alpha
    global gamma
    global epsilon
    global tau
    global a
    global k
    global dt
    global dx
    global dy
    global delta
    global j
    global epsilon_avg
    global theta_0

    global steps
    global nx
    global ny

    alpha = 0.9     # element of interval (0, 1)
    gamma = 10      #
    #epsilon = 0.01  # thickness of interface / motion of interface
    tau = 0.0003    # time constant
    theta_0 = 0
    a = 0.01        # noise amplitude
    j = 6
    k = 2.0         # dimensionless latent heat
    dt = 0.0002     # timestep
    steps = 2000    # steps
    dx = 0.03       # grid size
    dy = 0.03
    j = 5
    delta = 0.04
    epsilon_avg = 0.01
    theta_0 = np.pi/2

    nx = 500
    ny = 500

    #seed_border = [[50, 200] for x in range(nx)]

    #seed_middle = [[250, 250], [250, 251], [251, 250], [251, 251]]

    seed_middle = [[149, 149], [151, 151], [149, 151], [151, 149]]

    P, T = initialize(nx, ny, seed_middle)
    P_solved = crystal_solve(P, T)
    crystal_plot(P_solved)


if __name__ == '__main__':
    main()

