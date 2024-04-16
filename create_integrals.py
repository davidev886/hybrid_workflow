import os
import sys
import numpy as np
import pickle
import json
from pyscf import gto, scf, fci, ao2mo, mcscf, lib
from pyscf.lib import chkfile
from pyscf.scf.chkfile import dump_scf
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
import shutil
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
from openfermion.linalg import get_sparse_operator
from src.spin_square import of_spin_operator
from openfermion.hamiltonians import s_squared_operator
import h5py
from collections import defaultdict
import pandas as pd



def molecule_data(atom_name):
    # table B1 angstrom
    molecules = {'ozone': [('O', (0.0000000, 0.0000000, 0.0000000)),
                           ('O', (0.0000000, 0.0000000, 1.2717000)),
                           ('O', (1.1383850, 0.0000000, 1.8385340))],
                 'H2': [('H', (0.0000000, 0.0000000, 0.0000000)),
                        ('H', (0.0000000, 0.0000000, 1.2717000))]}

    return molecules[atom_name]


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)
    # dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    # dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    # dmrgscf.settings.MPIPREFIX = ''

    with open(sys.argv[1]) as f:
        options = json.load(f)

    target = options.get("target", "nvidia")
    num_active_orbitals = options.get("num_active_orbitals", 5)
    num_active_electrons = options.get("num_active_electrons", 5)
    chkptfile_rohf = options.get("chkptfile_rohf", None)
    chkptfile_cas = options.get("chkptfile_cas", None)
    ipie_input_dir = options.get("ipie_input_dir", "./")
    basis = options.get("basis", 'cc-pVTZ').lower()
    atom = options.get("atom", 'geo.xyz')
    dmrg = options.get("dmrg", 0)
    dmrg_states = options.get("dmrg_states", 1000)
    spin = options.get("spin", 1)
    label_molecule = options.get("label_molecule", "FeNTA")
    hamiltonian_fname = f"ham_{label_molecule}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.pickle"

    print(hamiltonian_fname)
    os.makedirs(ipie_input_dir, exist_ok=True)
    multiplicity = spin + 1
    charge = 0

    if label_molecule.lower() in ("h2", "hydrogen"):
        atom = molecule_data("H2")

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
        dir_path = f"{label_molecule}_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o/dmrg_M_{dmrg_states}"
        my_casci.fcisolver = dmrgscf.DMRGCI(mol, maxM=dmrg_states, tol=1E-10)
        my_casci.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
        # my_casci.fcisolver.threads = 8
        my_casci.fcisolver.memory = int(mol.max_memory / 1000)  # mem in GB
        my_casci.fcisolver.conv_tol = 1e-14
    else:
        dir_path = f"{label_molecule}_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o"
        x = (mol.spin / 2 * (mol.spin / 2 + 1))
        print(f"x={x}")
        my_casci.fix_spin_(ss=x)

    os.makedirs(dir_path, exist_ok=True)

    if chkptfile_cas and os.path.exists(chkptfile_cas):
        mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel(mo)
        coeff, occa, occb = zip(
            *fci.addons.large_ci(fcivec,
                                 num_active_orbitals,
                                 (nocca_act, noccb_act),
                                 tol=0,
                                 return_strs=False)
        )
        wf_fname = f"{label_molecule}_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o_chk.h5"
        shutil.copy(chkptfile_cas, os.path.join(ipie_input_dir, wf_fname))

        with h5py.File(os.path.join(ipie_input_dir, wf_fname), "r+") as fh5:
            fh5["mcscf/ci_coeffs"] = coeff
            fh5["mcscf/occs_alpha"] = occa
            fh5["mcscf/occs_beta"] = occb
        print(coeff)
    else:
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()

    print('FCI Energy in CAS:', e_tot)

    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    # os.makedirs(dir_path, exist_ok=True)
    np.save(os.path.join(dir_path, "h1.npy"), h1)
    np.save(os.path.join(dir_path, "tbi.npy"), tbi)
    np.save(os.path.join(dir_path, "energy_core.npy"), energy_core)

    mol_ham = generate_hamiltonian(h1, tbi, energy_core.item(), EQ_TOLERANCE=1e-8)
    jw_hamiltonian = jordan_wigner(mol_ham)
    print("number of terms in H", len(jw_hamiltonian.terms))
    filehandler = open(os.path.join(dir_path, hamiltonian_fname), 'wb')
    pickle.dump(jw_hamiltonian, filehandler)
    spin_sz = of_spin_operator("projected", 2*num_active_orbitals)
    spin_s_square = get_sparse_operator(s_squared_operator(num_active_orbitals))
    total_operator = get_sparse_operator(jw_hamiltonian) + 1e-8 * spin_sz

    evals, evecs = np.linalg.eigh(total_operator.toarray())

    eigenvectors = evecs.T

    data_energy = defaultdict(list)
    for j, vec in enumerate(eigenvectors):
        spin_value = vec.conj().T @ spin_s_square @ vec
        spin_proj = vec.conj().T @ spin_sz @ vec
        data_energy["S^2"].append(round(spin_value.real, 2))
        data_energy["Sz"].append(round(spin_proj.real, 3))

        data_energy["Energy"].append(evals[j].real)
        print(f"{spin_value.real:+.4f}, {spin_proj.real:+.4f}, {evals[j].real:+.10f}")
        if j == 0:
            np.savetxt("wf_h2.dat", vec)

    df = pd.DataFrame(data_energy)
    df = df.sort_values(["S^2", "Energy", "Sz"])
    df.to_csv(f"energy_{label_molecule}_s_{spin}_{basis.lower()}_{num_active_electrons}e_{num_active_orbitals}o.csv", index=False)