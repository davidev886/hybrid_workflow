"""
Prepare input for MSD gpu ipie
"""

import os
import sys
import numpy as np
import json
from pyscf import gto, scf
import shutil
import h5py
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk


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
    use_gpu = options.get("use_gpu", 0)
    num_gpus = options.get("num_gpus", 4)
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
    mf = scf.ROHF(mol)
    if chkptfile_rohf and os.path.exists(chkptfile_rohf):
        dm = mf.from_chk(chkptfile_rohf)
        # mf.max_cycle = 0
        mf.kernel(dm)
        # make a copy of the chk file from pyscf and append the info on the MSD trial
        chk_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
        shutil.copy(chkptfile_rohf, os.path.join(ipie_input_dir, chk_fname))
    else:
        chk_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
        print("# saving chkfile to", os.path.join(ipie_input_dir, chk_fname))
        mf.chkfile = os.path.join(ipie_input_dir, chk_fname)
        mf.kernel()

    ham_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_ham.h5"

    gen_ipie_input_from_pyscf_chk(os.path.join(ipie_input_dir, chk_fname),
                                  hamil_file=os.path.join(ipie_input_dir, ham_file),
                                  mcscf=False)

    from ipie.utils.chunk_large_chol import split_cholesky

    chol_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chol.h5"
    print("# splitting cholesky in", os.path.join(ipie_input_dir, chol_fname))
    split_cholesky(os.path.join(ipie_input_dir, ham_file),
                   num_gpus,
                   chol_fname=os.path.join(ipie_input_dir, chol_fname))  # split the cholesky to 4 subfiles

    # test that the splitting is done correctly. will this remove later
    from mpi4py import MPI
    from ipie.utils.mpi import MPIHandler, make_splits_displacements
    from ipie.utils.pack_numba import pack_cholesky

    nmembers = 1
    comm = MPI.COMM_WORLD

    with h5py.File(os.path.join(ipie_input_dir, ham_file)) as fa:
        e0 = fa["e0"][()]
        hcore = fa["hcore"][()]

    rank = comm.Get_rank()
    size = comm.Get_size()
    srank = rank % nmembers

    handler = MPIHandler(nmembers=nmembers, verbose=True)

    num_basis = hcore.shape[-1]
    chol_fname = os.path.splitext(os.path.join(ipie_input_dir, chol_fname))[0]
    with h5py.File(f"{chol_fname}_{srank}.h5", 'r') as fa:
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
