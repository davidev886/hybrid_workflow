import os
import sys
import numpy as np
import json
from pyscf import gto, scf, fci, mcscf, lib
from pyscf.lib import chkfile
import shutil
import h5py
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked
from src.from_pyscf_mod import gen_ipie_input_from_pyscf_chk_mod


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

    try:
        import cupy
        from mpi4py import MPI
    except ImportError:
        sys.exit(0)

    from ipie.config import config

    config.update_option("use_gpu", True)

    from ipie.utils.backend import arraylib as xp

    from ipie.hamiltonians.generic_chunked import GenericRealCholChunked as HamGeneric
    from ipie.qmc.afqmc import AFQMC
    from ipie.systems.generic import Generic
    from ipie.trial_wavefunction.single_det import SingleDet

    gpu_number_per_node = 4
    nmembers = 1
    gpu_id = MPI.COMM_WORLD.rank % gpu_number_per_node
    xp.cuda.Device(gpu_id).use()

    comm = MPI.COMM_WORLD
    num_walkers = 24 // comm.size
    nsteps = 25
    nblocks = 4
    timestep = 0.005
    rng_seed = None

    with h5py.File("hamiltonian.h5") as fa:
        e0 = fa["e0"][()]
        hcore = fa["hcore"][()]

    rank = comm.Get_rank()
    size = comm.Get_size()
    srank = rank % nmembers

    from ipie.utils.mpi import MPIHandler, make_splits_displacements

    handler = MPIHandler(nmembers=nmembers, verbose=True)

    from ipie.utils.pack_numba import pack_cholesky

    num_basis = hcore.shape[-1]
    with h5py.File(f"chol_{srank}.h5") as fa:
        chol_chunk = fa["chol"][()]

    chunked_chols = chol_chunk.shape[-1]
    num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)

    chol_chunk_view = chol_chunk.reshape((num_basis, num_basis, -1))
    cp_shape = (num_basis * (num_basis + 1) // 2, chol_chunk_view.shape[-1])
    chol_packed_chunk = np.zeros(cp_shape, dtype=chol_chunk_view.dtype)
    sym_idx = np.triu_indices(num_basis)
    pack_cholesky(sym_idx[0], sym_idx[1], chol_packed_chunk, chol_chunk_view)
    del chol_chunk_view

    split_size = make_splits_displacements(num_chol, nmembers)[0]
    assert chunked_chols == split_size[srank]

    num_basis = hcore.shape[-1]
    mol_nelec = mol.nelec
    system = Generic(nelec=mol_nelec)

    ham = HamGeneric(
        np.array([hcore, hcore]),
        None,
        chol_chunk,
        chol_packed_chunk,
        e0, handler
    )
    ham.nchol = num_chol
    ham.handler = handler

    # Build trial wavefunction
    with h5py.File(os.path.join(ipie_input_dir, wfn_file), "r") as fh5:
        coeff = fh5["ci_coeffs"][:]
        occa = fh5["occ_alpha"][:]
        occb = fh5["occ_beta"][:]

    wavefunction = (coeff, occa, occb)
    trial = ParticleHoleNonChunked(
        wavefunction,
        mol.nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    # from ipie.walkers.uhf_walkers import UHFWalkers
    #
    # walkers = UHFWalkers(numpy.hstack([phi0a, phi0a]), system.nup, system.ndown, ham.nbasis, num_walkers,
    #                      mpi_handler=handler)
    # walkers.build(trial)

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
        verbose=True,
        mpi_handler=handler)

    afqmc_msd.run()
    afqmc_msd.finalise(verbose=True)
