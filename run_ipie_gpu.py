"""
Run ipie with gpu support

"""
import os
import sys
import numpy as np
import json
from pyscf import gto
import h5py
# env CUPY_ACCELERATORS = cutensor # for notebook, for .py you can set this in terminal

try:
    import cupy
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

from ipie.config import config
config.update_option("use_gpu", True)

from ipie.trial_wavefunction.particle_hole import ParticleHole  # noqa: E402
from ipie.hamiltonians.generic import Generic as HamGeneric  # noqa: E402
from ipie.qmc.afqmc import AFQMC  # noqa: E402
from ipie.systems.generic import Generic  # noqa: E402
from src.input_ipie import IpieInput  # noqa: E402


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)

    with open(sys.argv[1]) as f:
        options = json.load(f)

    input_ipie = IpieInput(options)
    num_active_orbitals = input_ipie.num_active_orbitals
    num_active_electrons = input_ipie.num_active_electrons
    basis = input_ipie.basis
    atom = input_ipie.atom
    charge = input_ipie.charge
    spin = input_ipie.spin
    label_molecule = input_ipie.label_molecule
    nwalkers = input_ipie.nwalkers
    nsteps = input_ipie.nsteps
    nblocks = input_ipie.nblocks
    seed = input_ipie.seed
    pop_control_freq = input_ipie.pop_control_freq
    use_gpu = input_ipie.use_gpu
    ipie_input_dir = input_ipie.ipie_input_dir
    ham_file = input_ipie.ham_file
    wfn_file = input_ipie.wfn_file
    chol_fname = input_ipie.chol_fname
    timestep = input_ipie.timestep
    stabilize_freq = input_ipie.stabilize_freq

    multiplicity = spin + 1

    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=4
    )
    # nocca, noccb = mol.nelec

    with h5py.File(os.path.join(ipie_input_dir, ham_file), "r") as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    num_basis = chol.shape[1]
    mol_nelec = mol.nelec
    system = Generic(nelec=mol_nelec)

    num_chol = chol.shape[0]
    ham = HamGeneric(
        np.array([h1e, h1e]),
        chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    # Build trial wavefunction
    with h5py.File(os.path.join(ipie_input_dir, wfn_file), "r") as fh5:
        coeff = fh5["ci_coeffs"][:]
        occa = fh5["occ_alpha"][:]
        occb = fh5["occ_beta"][:]

    wavefunction = (coeff, occa, occb)
    trial = ParticleHole(
        wavefunction,
        mol.nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )

    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    afqmc_msd = AFQMC.build(
        mol_nelec,
        ham,
        trial,
        num_walkers=nwalkers,
        num_steps_per_block=nsteps,
        num_blocks=nblocks,
        timestep=timestep,
        stabilize_freq=stabilize_freq,
        seed=seed,
        pop_control_freq=pop_control_freq,
        verbose=True)

    afqmc_msd.run()
    afqmc_msd.finalise(verbose=True)
