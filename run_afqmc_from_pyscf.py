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
from from_pyscf_mod import gen_ipie_input_from_pyscf_chk_mod


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
    else:
        print("# saving chkfile to", os.path.join(ipie_input_dir, chkptfile_rohf))
        mf.chkfile = os.path.join(ipie_input_dir, chkptfile_rohf)
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
    chk_fname = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
    ham_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_ham.h5"
    wfn_file = f"{label_molecule}_s_{spin}_{basis}_{num_active_electrons}e_{num_active_orbitals}o_wfn.h5"

    # make a copy of the chk file from pyscf and append the info on the MSD trial
    shutil.copy(chkptfile_cas, os.path.join(ipie_input_dir, chk_fname))
    with h5py.File(os.path.join(ipie_input_dir, chk_fname), "r+") as fh5:
        fh5["mcscf/ci_coeffs"] = coeff
        fh5["mcscf/occs_alpha"] = occa
        fh5["mcscf/occs_beta"] = occb

    print('FCI Energy in CAS:', e_tot)

    gen_ipie_input_from_pyscf_chk_mod(os.path.join(ipie_input_dir, chk_fname),
                                      hamil_file=os.path.join(ipie_input_dir, ham_file),
                                      wfn_file=os.path.join(ipie_input_dir, wfn_file),
                                      mcscf=True,
                                      gen_ham=generate_chol_hamiltonian)

    with h5py.File(os.path.join(ipie_input_dir, ham_file)) as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    num_basis = chol.shape[1]
    system = Generic(nelec=mol.nelec)

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

    afqmc_msd = AFQMC.build(
        mol.nelec,
        ham,
        trial,
        num_walkers=100,
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
