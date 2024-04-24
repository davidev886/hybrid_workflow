"""
Prepare input for MSD gpu ipie
"""

import os
import sys
import numpy as np
import json
from pyscf import gto, scf, fci, mcscf, lib
from pyscf.lib import chkfile
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
    chol_cut = options.get("chol_cut", 1e-5)
    hamiltonian_fname = f"ham_{label_molecule}_{basis}_{num_active_electrons}e_{num_active_orbitals}o.pickle"
    chk_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
    ham_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_ham.h5"
    wfn_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_wfn.h5"
    chol_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chol.h5"

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
    try:
        os.remove(os.path.join(ipie_input_dir, chk_fname))
    except OSError:
        pass

    if chkptfile_rohf and os.path.exists(chkptfile_rohf):
        dm = mf.from_chk(chkptfile_rohf)
        # mf.max_cycle = 0
        mf.kernel(dm)
        # make a copy of the chk file from pyscf and append the info on the MSD trial
        shutil.copy(chkptfile_rohf, os.path.join(ipie_input_dir, chk_fname))
    else:
        print("# saving chkfile to", os.path.join(ipie_input_dir, chk_fname))
        mf.chkfile = os.path.join(ipie_input_dir, chk_fname)
        mf.kernel()

    my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
    nocca_act = (num_active_electrons + spin) // 2
    noccb_act = (num_active_electrons - spin) // 2
    if dmrg in (1, 'true'):
        from pyscf import dmrgscf
        # dir_path = (f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o/"
        #             f"dmrg_M_{dmrg_states}")
        my_casci.fcisolver = dmrgscf.DMRGCI(mol, maxM=dmrg_states, tol=1E-10)
        my_casci.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.threads = dmrg_thread
        my_casci.fcisolver.memory = int(mol.max_memory / 1000)  # mem in GB
        my_casci.fcisolver.conv_tol = 1e-14
    else:
        # dir_path = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o"
        x = (mol.spin / 2 * (mol.spin / 2 + 1))
        print(f"fix spin squared to x={x}")
        my_casci.fix_spin_(ss=x)

    if chkptfile_cas and os.path.exists(chkptfile_cas):
        mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel(mo)
    else:
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()

    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec,
                             num_active_orbitals,
                             (nocca_act, noccb_act),
                             tol=threshold_wf,
                             return_strs=False)
    )
    # append the info on the MSD trial to the chk file from pyscf
    with h5py.File(os.path.join(ipie_input_dir, chk_fname), "r+") as fh5:
        fh5["mcscf/ci_coeffs"] = coeff
        fh5["mcscf/occs_alpha"] = occa
        fh5["mcscf/occs_beta"] = occb

    print('FCI Energy in CAS:', e_tot)

    gen_ipie_input_from_pyscf_chk(os.path.join(ipie_input_dir, chk_fname),
                                  hamil_file=os.path.join(ipie_input_dir, ham_file),
                                  wfn_file=os.path.join(ipie_input_dir, wfn_file),
                                  chol_cut=chol_cut,
                                  mcscf=True)

    from ipie.utils.chunk_large_chol import split_cholesky

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
