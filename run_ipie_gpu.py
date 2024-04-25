import os
import sys
import numpy as np
import json

# env CUPY_ACCELERATORS = cutensor # for notebook, for .py you can set this in terminal

try:
    import cupy
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

from ipie.config import config
config.update_option("use_gpu", True)

from pyscf import gto
import h5py

from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)
    # dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    # dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    # dmrgscf.settings.MPIPREFIX = ''

    with open(sys.argv[1]) as f:
        options = json.load(f)

    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    chkptfile_rohf = options.get("chkptfile_rohf", None)
    chkptfile_cas = options.get("chkptfile_cas", None)
    # ipie_input_dir contains the hamiltonian.h5 and wavefunction.h5 for running ipie
    ipie_input_dir = options.get("ipie_input_dir", "./")
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    dmrg = options.get("dmrg", 0)
    dmrg_states = options.get("dmrg_states", 1000)
    spin = options.get("spin", 1)
    label_molecule = options.get("label_molecule", "FeNTA")
    dmrg_thread = options.get("dmrg_thread", 2)
    threshold_wf = options.get("threshold_wf", 1e-6)
    generate_chol_hamiltonian = bool(options.get("generate_chol_hamiltonian", 1))
    nwalkers = options.get("nwalkers", 25)
    nsteps = options.get("nsteps", 10)
    nblocks = options.get("nblocks", 10)
    hamiltonian_fname = f"ham_{label_molecule}_{basis}_{num_active_electrons}e_{num_active_orbitals}o.pickle"

    os.makedirs(ipie_input_dir, exist_ok=True)
    multiplicity = spin + 1
    charge = 0

    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=4
    )
    nocca, noccb = mol.nelec

    chk_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
    ham_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_ham.h5"
    wfn_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_wfn.h5"

    num_walkers = 24
    nsteps = 25
    nblocks = 4
    timestep = 0.005
    rng_seed = None

    with h5py.File("hamiltonian.h5") as fa:
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
        timestep=0.005,
        stabilize_freq=5,
        seed=96264512,
        pop_control_freq=5,
        verbose=True)

    afqmc_msd.run()
    afqmc_msd.finalise(verbose=True)
