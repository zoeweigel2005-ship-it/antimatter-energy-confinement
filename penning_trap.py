"""
Relativistic Particle-in-Cell Boris Pusher for Antiproton Confinement in a Penning-Malmberg Trap.
"""


from dataclasses import dataclass
import numpy as np

def njit(f):
    return f

C = 299792458.0
Q_E = 1.602176634e-19
M_P = 1.67262192369e-27


@dataclass
class TrapConfig:
    b0: float = 4.0
    v0: float = 200.0
    z0: float = 0.03
    r0: float = 0.015

@njit
def compute_electric_field(r: np.ndarray, v0: float, z0: float, r0: float) -> np.ndarray:
    d_sq = z0**2 + 0.5 * r0**2
    coeff = v0 / d_sq
    # Inversion des signes pour créer un puits de potentiel pour charge négative
    return np.array([-coeff * r[0], -coeff * r[1], 2.0 * coeff * r[2]])

@njit
def boris_step(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    charge: float,
    mass: float,
    b0: float,
    v0: float,
    z0: float,
    r0: float
) -> tuple[np.ndarray, np.ndarray]:
    gamma = 1.0 / np.sqrt(1.0 - np.dot(vel, vel) / C**2)
    u = gamma * vel

    e_field = compute_electric_field(pos, v0, z0, r0)
    b_field = np.array([0.0, 0.0, b0])

    u_minus = u + (charge * e_field / (2.0 * mass)) * dt
    gamma_half = np.sqrt(1.0 + np.dot(u_minus, u_minus) / C**2)
    t_vec = (charge * b_field / (2.0 * mass * gamma_half)) * dt
    s_vec = 2.0 * t_vec / (1.0 + np.dot(t_vec, t_vec))

    u_prime = u_minus + np.cross(u_minus, t_vec)
    u_plus = u_minus + np.cross(u_prime, s_vec)
    u_next = u_plus + (charge * e_field / (2.0 * mass)) * dt

    gamma_next = np.sqrt(1.0 + np.dot(u_next, u_next) / C**2)
    vel_next = u_next / gamma_next
    pos_next = pos + vel_next * dt

    return pos_next, vel_next
