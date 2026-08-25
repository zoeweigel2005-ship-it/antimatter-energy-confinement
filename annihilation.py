"""
Annihilation reaction dynamics and energy spectrum modeling.
"""

import numpy as np

C = 299792458.0
M_P_MEV = 938.272


def simulate_ppbar_annihilation_yield(num_events: int = 100_000) -> dict:
    total_energy_per_annihilation_mev = 2.0 * M_P_MEV
    n_charged = np.random.poisson(3.0, num_events)
    n_neutral = np.random.poisson(1.5, num_events)
    total_pions = np.where(n_charged + n_neutral == 0, 1, n_charged + n_neutral)

    charged_fraction = n_charged / total_pions
    neutral_fraction = n_neutral / total_pions

    gamma_energy_mev = neutral_fraction * total_energy_per_annihilation_mev
    charged_energy_mev = charged_fraction * total_energy_per_annihilation_mev

    return {
        "mean_gamma_energy_mev": float(np.mean(gamma_energy_mev)),
        "mean_charged_kinetic_mev": float(np.mean(charged_energy_mev)),
        "usable_mhd_efficiency": float(np.mean(charged_fraction)),
        "gamma_losses_fraction": float(np.mean(neutral_fraction)),
        "specific_energy_density_j_per_kg": float(2.0 * C**2)
    }
