import sys
import numpy as np
from src.input_ipie import IpieInput
import h5py
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked
import os


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    option_file = sys.argv[1]
    input_ipie = IpieInput(option_file)

    if input_ipie.generate_chol_hamiltonian:
        input_ipie.gen_hamiltonian()
    input_ipie.gen_wave_function()

    input_ipie.check_energy_state()
    with h5py.File("hamiltonian.h5") as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    num_basis = chol.shape[1]
    system = Generic(nelec=input_ipie.mol_nelec)

    num_chol = chol.shape[0]
    ham = HamGeneric(
        np.array([h1e, h1e]),
        chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    # Build trial wavefunction
    with h5py.File(os.path.join(input_ipie.file_path, input_ipie.trial_name), "r") as fh5:
        coeff = fh5["ci_coeffs"][:]
        occa = fh5["occ_alpha"][:]
        occb = fh5["occ_beta"][:]

    wavefunction = (coeff, occa, occb)
    trial = ParticleHoleNonChunked(
        wavefunction,
        input_ipie.mol_nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    afqmc_msd = AFQMC.build(
        input_ipie.mol_nelec,
        ham,
        trial,
        num_walkers=10,
        num_steps_per_block=25,
        num_blocks=10,
        timestep=0.005,
        stabilize_freq=5,
        seed=96264512,
        pop_control_freq=5,
        verbose=True,
    )

    afqmc_msd.run()
    afqmc_msd.finalise(verbose=True)


if __name__ == "__main__":
    main()
