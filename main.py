import numpy as np
import matplotlib.pyplot as plt
from src.penning_trap import TrapConfig, boris_step, M_P, Q_E
from src.annihilation import simulate_ppbar_annihilation_yield


def run_confinement_simulation():
    config = TrapConfig(b0=4.0, v0=200.0, z0=0.03, r0=0.015)
    pos = np.array([0.002, 0.0, 0.005])
    v_thermal = 3.0e4
    vel = np.array([v_thermal, 0.5 * v_thermal, 0.2 * v_thermal])

    dt = 1e-10
    n_steps = 20000

    trajectory = np.zeros((n_steps, 3))
    for step in range(n_steps):
        trajectory[step] = pos
        pos, vel = boris_step(
            pos, vel, dt,
            charge=-Q_E, mass=M_P,
            b0=config.b0, v0=config.v0,
            z0=config.z0, r0=config.r0
        )

    return trajectory


def main():
    print("Simulating relativistic antiproton confinement...")
    traj = run_confinement_simulation()

    print("Computing annihilation channels and energy yield...")
    yield_data = simulate_ppbar_annihilation_yield()

    print("\n--- Summary Results ---")
    print(f"Energy Density: {yield_data['specific_energy_density_j_per_kg']:.2e} J/kg")
    print(f"MHD Usable Energy (Charged Pions): {yield_data['usable_mhd_efficiency'] * 100:.1f}%")
    print(f"Prompt Gamma Radiation Losses: {yield_data['gamma_losses_fraction'] * 100:.1f}%")

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, traj[:, 2] * 1e3, color='crimson', lw=0.6)
    ax1.set_title("Relativistic Antiproton Trajectory in Penning Trap")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_zlabel("z (mm)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, color='indigo', lw=0.6)
    ax2.set_title("Radial Plane Projection (Cyclotron + Magnetron Motion)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
