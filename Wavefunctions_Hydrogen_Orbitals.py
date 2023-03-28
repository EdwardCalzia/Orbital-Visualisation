import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define the hydrogen wavefunction
def psi(n, l, m, r, theta, phi):
    a0 = 1 # Bohr radius
    R_nl = np.sqrt((2/(n*a0))**3*np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l)**3))*np.exp(-r/(n*a0))*(2*r/(n*a0))**l* np.polyval(np.array([-1, 2*l+1, -l*(l+1)])/2, r/(n*a0))
    Y_lm = np.real(np.sqrt((2*l+1)/(4*np.pi)*np.math.factorial(l-np.abs(m))/np.math.factorial(l+np.abs(m))) * np.exp(1j*m*phi) * np.polynomial.legendre.Legendre(l)(np.cos(theta)))
    return R_nl*Y_lm

# Define the Bohmian trajectory function
def bohmian_trajectory(n, l, m, initial_position, num_steps=5000, step_size=0.01):
    # Set up the initial position and velocity
    x, y, z = initial_position
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    p_r = np.real(psi(n, l, m, r, theta, phi))
    p_theta = np.imag(psi(n, l, m, r, theta, phi))/r
    p_phi = np.imag(psi(n, l, m, r, theta, phi))/(r*np.sin(theta))

    # Simulate the Bohmian trajectory
    positions = []
    for i in range(num_steps):
        positions.append([x, y, z])
        x += step_size * p_r / np.sqrt(1 + (p_theta**2 + p_phi**2)/p_r**2)
        y += step_size * p_theta / np.sqrt(1 + (p_r**2 + p_phi**2)/p_theta**2)
        z += step_size * p_phi / np.sqrt(1 + (p_r**2 + p_theta**2)/p_phi**2)
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        p_r = np.real(psi(n, l, m, r, theta, phi))
        p_theta = np.imag(psi(n, l, m, r, theta, phi))/r
        p_phi = np.imag(psi(n, l, m, r, theta, phi))/(r*np.sin(theta))

    return np.array(positions)

# Define the hydrogen orbital to visualize
n = 2
l = 1
m = 0

# Generate Bohmian trajectories for the orbital
initial_positions = np.random.normal(0, 1, size=(1000, 3))
trajectories = []
for pos in initial_positions:
    traj = bohmian_trajectory(n, l, m, pos)
    trajectories.append(traj)
trajectories = np.array(trajectories)

# Plot the trajectories in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

r_vals = np.linspace(0, 10, 100)
theta_vals = np.linspace(0, np.pi, 100)
phi_vals = np.linspace(0, np.pi, 100)
R, Theta, Phi = np.meshgrid(r_vals, theta_vals, phi_vals)
X = R * np.sin(Theta) * np.cos(Phi) # fixed typo here: use * instead of .
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Theta)
psi_vals = np.abs(psi(n, l, m, R, Theta, Phi))**2
ax.scatter(X, Y, Z, c=psi_vals.flatten(), cmap='coolwarm', alpha=0.1)

for traj in trajectories:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.1)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
