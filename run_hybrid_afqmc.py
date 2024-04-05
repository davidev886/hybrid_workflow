import sys
import numpy as np
import os
import json
import h5py
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked
from src.input_ipie import IpieInput
from src.s2_estimator import S2Mixed


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    options_file = sys.argv[1]
    with open(options_file) as f:
        options = json.load(f)

    input_ipie = IpieInput(options)

    if input_ipie.generate_chol_hamiltonian:
        input_ipie.gen_hamiltonian()
    input_ipie.gen_wave_function()

    input_ipie.check_energy_state()

    with h5py.File(os.path.join(input_ipie.ipie_input_dir, input_ipie.chol_hamil_file)) as fa:
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
    with h5py.File(os.path.join(input_ipie.ipie_input_dir, input_ipie.trial_name), "r") as fh5:
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

    estimators = {"S2": S2Mixed(ham=afqmc_msd.hamiltonian)}
    afqmc_msd.run(additional_estimators=estimators,
                  estimator_filename=os.path.join(input_ipie.output_dir, "S2_data.dat"))
    afqmc_msd.finalise(verbose=True)

    # We can extract the qmc data as as a pandas data frame like so
    from ipie.analysis.extraction import extract_observable

    # Note the 'energy' estimator is always computed.
    qmc_data = extract_observable(afqmc_msd.estimators.filename, "S2")
    print(qmc_data)


if __name__ == "__main__":
    main()

