"""
Prepare input for MSD gpu ipie
"""

import os
import sys
import numpy as np
import json
from pyscf import fci, mcscf, lib
from pyscf.lib import chkfile
import h5py
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from src.input_ipie import IpieInput


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)
    # dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    # dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    # dmrgscf.settings.MPIPREFIX = ''

    with open(sys.argv[1]) as f:
        options = json.load(f)

    input_ipie = IpieInput(options)

    num_active_orbitals = input_ipie.num_active_orbitals
    num_active_electrons = input_ipie.num_active_electrons
    chkptfile_rohf = input_ipie.chkptfile_rohf
    chkptfile_cas = input_ipie.chkptfile_cas
    ipie_input_dir = input_ipie.ipie_input_dir
    basis = input_ipie.basis
    atom = input_ipie.atom
    dmrg = input_ipie.dmrg
    dmrg_states = input_ipie.dmrg_states
    spin = input_ipie.spin
    label_molecule = input_ipie.label_molecule
    dmrg_thread = input_ipie.dmrg_thread
    threshold_wf = input_ipie.threshold_wf
    file_wavefunction = input_ipie.file_wavefunction
    generate_chol_hamiltonian = input_ipie.generate_chol_hamiltonian
    nwalkers = input_ipie.nwalkers
    nsteps = input_ipie.nsteps
    nblocks = input_ipie.nblocks
    use_gpu = input_ipie.use_gpu
    num_gpus = input_ipie.num_gpus
    chol_cut = input_ipie.chol_cut
    chol_split = input_ipie.chol_split
    chk_fname = input_ipie.chk_fname
    hamiltonian_fname = input_ipie.hamiltonian_fname
    ham_file = input_ipie.ham_file
    wfn_file = input_ipie.wfn_file
    chol_fname = input_ipie.chol_fname
    charge = input_ipie.charge
    multiplicity = spin + 1

    mol = input_ipie.mol
    mf = input_ipie.mf
    my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
    nocca_act = (num_active_electrons + spin) // 2
    noccb_act = (num_active_electrons - spin) // 2
    if dmrg:
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
    print('FCI Energy in CAS:', e_tot)

    if file_wavefunction:
        coeff, occa, occb = input_ipie.gen_wave_function()
    else:
        coeff, occa, occb = zip(
            *fci.addons.large_ci(fcivec,
                                 num_active_orbitals,
                                 (nocca_act, noccb_act),
                                 tol=threshold_wf,
                                 return_strs=False)
        )

    # append the info on the MSD trial to the chk file from pyscf
    print(f"# Append the info on the MSD trial to the chk file {os.path.join(ipie_input_dir, chk_fname)}")
    with h5py.File(os.path.join(ipie_input_dir, chk_fname), "r+") as fh5:
        fh5["mcscf/ci_coeffs"] = coeff
        fh5["mcscf/occs_alpha"] = occa
        fh5["mcscf/occs_beta"] = occb

    print(f"# Writing hamiltonian to {os.path.join(ipie_input_dir, ham_file)}")
    print(f"# Writing wavefunction to {os.path.join(ipie_input_dir, wfn_file)}")
    gen_ipie_input_from_pyscf_chk(pyscf_chkfile=os.path.join(ipie_input_dir, chk_fname),
                                  hamil_file=os.path.join(ipie_input_dir, ham_file),
                                  wfn_file=os.path.join(ipie_input_dir, wfn_file),
                                  chol_cut=chol_cut,
                                  mcscf=True)

    if chol_split:
        from ipie.utils.chunk_large_chol import split_cholesky

        print("# splitting cholesky in", os.path.join(ipie_input_dir, chol_fname))
        split_cholesky(os.path.join(ipie_input_dir, ham_file),
                       num_gpus,
                       chol_fname=os.path.join(ipie_input_dir, chol_fname))  # split the cholesky to 4 subfiles
