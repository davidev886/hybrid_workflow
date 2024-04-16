import numpy as np

import os
import sys
import json
from openfermion.linalg import get_sparse_operator
from openfermion.hamiltonians import s_squared_operator
from datetime import datetime
import h5py

from ipie.utils.io import write_hamiltonian, write_wavefunction
from ipie.utils.from_pyscf import (load_from_pyscf_chkfile,
                                   generate_hamiltonian,
                                   copy_LPX_to_LXmn,
                                   generate_wavefunction_from_mo_coeff)

from src.spin_square import of_spin_operator
from src.pyscf_scripts import normal_ordering_swap
import pickle


def convert_state_big_endian(state_little_endian):

    state_big_endian = 0. * state_little_endian

    n_qubits = int(np.log2(state_big_endian.size))
    for j, val in enumerate(state_little_endian):
        little_endian_pos = np.binary_repr(j, n_qubits)
        big_endian_pos = little_endian_pos[::-1]
        int_big_endian_pos = int(big_endian_pos, 2)
        state_big_endian[int_big_endian_pos] = state_little_endian[j]

    return state_big_endian


def get_coeff_wf(final_state_vector, n_elec, ncore_electrons=None, thres=1e-6):
    """
    :param final_state_vector: State vector from a VQE simulation
    :param n_elec: Number of electrons in active space
    :param ncore_electrons: Number of electrons in core space
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of occupied alpha, list of occupied bets
    """
    bin_ind = [np.binary_repr(i, width=int(np.log2(len(final_state_vector)))) for i in
               range(len(final_state_vector))]
    coeff = []
    occas = []
    occbs = []

    for k, i in enumerate(bin_ind):
        alpha_aux = []
        beta_aux = []
        for j in range(len(i) // 2):
            alpha_aux.append(i[2 * j])
            beta_aux.append(i[2 * j + 1])
        alpha_occ = [i for i, x in enumerate(alpha_aux) if x == '1']
        beta_occ = [i for i, x in enumerate(beta_aux) if x == '1']

        if (np.abs(final_state_vector[k]) >= thres) and (len(alpha_occ) == n_elec[0]) and (len(beta_occ) == n_elec[1]):
            coeff.append(final_state_vector[k])
            occas.append(alpha_occ)
            occbs.append(beta_occ)
    # We need it non_normal ordered
    for i in range(len(coeff)):
        coeff[i] = (-1) ** (
            normal_ordering_swap([2 * j for j in occas[i]] + [2 * j + 1 for j in occbs[i]])) * \
                   coeff[i]
    ncore = ncore_electrons #// 2
    core = [i for i in range(ncore)]
    occas = [np.array(core + [o + ncore for o in oa]) for oa in occas]
    occbs = [np.array(core + [o + ncore for o in ob]) for ob in occbs]

    return coeff, occas, occbs


class IpieInput(object):
    def __init__(self, options):

        self.num_active_orbitals = options.get("num_active_orbitals", 5)
        self.num_active_electrons = options.get("num_active_electrons", 5)
        self.basis = options.get("basis", 'cc-pVTZ').lower()
        self.atom = options.get("atom", 'geo.xyz')
        self.dmrg = options.get("dmrg", 0)
        self.dmrg_states = options.get("dmrg_states", 1000)
        self.chkptfile_rohf = options.get("chkptfile_rohf", None)
        self.chkptfile_cas = options.get("chkptfile_cas", None)
        self.spin = options.get("spin", 1)
        self.hamiltonian_fname = options.get("hamiltonian_fname", 1)
        self.optimizer_type = options.get("optimizer_type", "cudaq")
        self.start_layer = options.get("start_layer", 1)
        self.end_layer = options.get("end_layer", 10)
        self.init_params = options.get("init_params", None)
        self.filen_state_vec = options.get("file_wavefunction", None)
        self.output_dir = options.get("output_dir", "")
        self.mcscf = options.get("mcscf", 1)
        self.chol_cut = options.get("chol_cut", 1e-5)
        self.chol_hamil_file = options.get("chol_hamil_file", "hamiltonian.h5")
        self.generate_chol_hamiltonian = options.get("generate_chol_hamiltonian", 0)
        self.ortho_ao = options.get("ortho_ao", 0)
        self.n_qubits = 2 * self.num_active_orbitals
        self.num_frozen_core = options.get("num_frozen_core", 0)
        self.ipie_input_dir = options.get("ipie_input_dir", "./")
        self.check_energy_openfermion = options.get("check_energy_openfermion", 0)
        # self.ncore_electrons = options.get("ncore_electrons", 0)

        pyscf_chkfile = self.chkptfile_rohf
        if self.mcscf:
            self.scf_data = load_from_pyscf_chkfile(pyscf_chkfile, base="mcscf")
        else:
            self.scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
        self.mol = self.scf_data["mol"]
        self.mol_nelec = self.mol.nelec

        self.n_alpha = int((self.num_active_electrons + self.spin) / 2)
        self.n_beta = int((self.num_active_electrons - self.spin) / 2)
        self.ncore_electrons = (sum(self.mol_nelec) - (self.n_alpha + self.n_beta)) // 2
        print("# (nalpha, nbeta)_total =", self.mol_nelec)
        print("# (nalpha, nbeta)_active =", self.n_alpha, self.n_beta)
        print("# ncore_electrons", self.ncore_electrons)
        self.trial_name = ""
        self.ndets = 0
        str_date = datetime.today().strftime('%Y%m%d_%H%M%S')

        if len(self.output_dir) == 0:
            self.output_dir = str_date
        else:
            self.output_dir = self.output_dir + "_" + str_date

        os.makedirs(self.output_dir, exist_ok=True)

        os.makedirs(self.ipie_input_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, sys.argv[1]), 'w') as f:
            json.dump(options, f, ensure_ascii=False, indent=4)

    def gen_wave_function(self):
        """
            doc
        """
        mol = self.mol
        num_active_orbitals = self.num_active_orbitals
        num_active_electrons = self.num_active_electrons
        filen_state_vec = self.filen_state_vec
        file_path = self.ipie_input_dir
        ncore_electrons = self.ncore_electrons
        if self.mcscf:
            spin_s_square = get_sparse_operator(s_squared_operator(num_active_orbitals))
            spin_s_z = of_spin_operator("projected", 2 * num_active_orbitals)

            final_state_vector = np.loadtxt(filen_state_vec, dtype=complex, comments="#")
            final_state_vector = convert_state_big_endian(final_state_vector)

            normalization = np.sqrt(np.dot(final_state_vector.T.conj(), final_state_vector)).real

            final_state_vector /= normalization

            spin_sq_value = final_state_vector.conj().T @ spin_s_square @ final_state_vector
            spin_proj = final_state_vector.conj().T @ spin_s_z @ final_state_vector
            if self.check_energy_openfermion:
                filehandler_ham = open(self.hamiltonian_fname, 'rb')
                jw_hamiltonian = pickle.load(filehandler_ham)

                jw_hamiltonian_sparse = get_sparse_operator(jw_hamiltonian, 2 * self.num_active_orbitals)

                energy = final_state_vector.conj().T @ jw_hamiltonian_sparse @ final_state_vector
                print(f"# Energy of the trial {self.filen_state_vec} is ", energy)

            print("# spin_sq_value", spin_sq_value)
            print("# spin_proj", spin_proj)
            print("# ncore_electrons", ncore_electrons)
            coeff, occas, occbs = get_coeff_wf(final_state_vector,
                                               (self.n_alpha, self.n_beta),
                                               ncore_electrons)

            coeff = np.array(coeff, dtype=complex)
            ixs = np.argsort(np.abs(coeff))[::-1]
            coeff = coeff[ixs]
            occas = np.array(occas)[ixs]
            occbs = np.array(occbs)[ixs]
            self.ndets = np.size(coeff)
            bare_filen_state_vec = os.path.splitext(os.path.basename(filen_state_vec))[0]
            self.trial_name = f'{bare_filen_state_vec}_msd_trial_{len(coeff)}.h5'

            write_wavefunction((coeff, occas, occbs),
                               os.path.join(file_path, self.trial_name))

            n_alpha = len(occas[0])
            n_beta = len(occbs[0])

            with h5py.File(
                    os.path.join(file_path, self.trial_name),
                    'a') as fh5:
                fh5['active_electrons'] = num_active_electrons
                fh5['active_orbitals'] = num_active_orbitals
                fh5['nelec'] = (n_alpha, n_beta)
        else:
            hcore = self.scf_data["hcore"]
            ortho_ao_mat = self.scf_data["X"]
            mo_coeffs = self.scf_data["mo_coeff"]
            mo_occ = self.scf_data["mo_occ"]

            nelec = (self.mol_nelec[0] - self.num_frozen_core, self.mol_nelec[1] - self.num_frozen_core)
            if self.ortho_ao:
                basis_change_matrix = ortho_ao_mat
            else:
                basis_change_matrix = mo_coeffs

            wfn = generate_wavefunction_from_mo_coeff(
                mo_coeffs,
                mo_occ,
                basis_change_matrix,
                nelec,
                ortho_ao=self.ortho_ao,
                num_frozen_core=self.num_frozen_core,
            )
            bare_filen_state_vec = os.path.splitext(os.path.basename(filen_state_vec))[0]
            self.trial_name = f'{bare_filen_state_vec}_sd_trial.h5'

            write_wavefunction(wfn, filename=os.path.join(file_path, self.trial_name))

    def gen_hamiltonian(self,
                        verbose: bool = True,
                        ) -> None:
        """
        adapted function gen_ipie_input_from_pyscf_chk from ipie/utils/from_pyscf.py
        """
        file_path = str(self.ipie_input_dir)
        scf_data = self.scf_data
        mol = self.mol
        chol_cut = self.chol_cut
        ortho_ao = self.ortho_ao
        hamil_file_name = self.chol_hamil_file
        hcore = scf_data["hcore"]
        ortho_ao_mat = scf_data["X"]
        mo_coeffs = scf_data["mo_coeff"]
        mo_occ = scf_data["mo_occ"]
        num_frozen_core = self.num_frozen_core

        if ortho_ao:
            basis_change_matrix = ortho_ao_mat
        else:
            basis_change_matrix = mo_coeffs

            if isinstance(mo_coeffs, list) or len(mo_coeffs.shape) == 3:
                if verbose:
                    print(
                        "# UHF mo coefficients found and ortho-ao == False. Using"
                        " alpha mo coefficients for basis transformation."
                    )
                basis_change_matrix = mo_coeffs[0]
        ham = generate_hamiltonian(
            mol,
            mo_coeffs,
            hcore,
            basis_change_matrix,
            chol_cut=chol_cut,
            num_frozen_core=num_frozen_core,
            verbose=verbose,
        )
        write_hamiltonian(ham.H1[0],
                          copy_LPX_to_LXmn(ham.chol),
                          ham.ecore,
                          filename=os.path.join(file_path, hamil_file_name)
                          )

    def check_energy_state(self):
        filen_state_vec = self.filen_state_vec

        final_state_vector = np.loadtxt(filen_state_vec, dtype=complex)

        final_state_vector = convert_state_big_endian(final_state_vector)
        normalization = np.sqrt(np.dot(final_state_vector.T.conj(), final_state_vector))
        final_state_vector /= normalization

        filehandler = open(self.hamiltonian_fname, 'rb')
        jw_hamiltonian = pickle.load(filehandler)

        jw_hamiltonian_sparse = get_sparse_operator(jw_hamiltonian, 2 * self.num_active_orbitals)

        energy = final_state_vector.conj().T @ jw_hamiltonian_sparse @ final_state_vector
        print(f"# Energy of the trial {self.filen_state_vec} is ", energy)

        return energy


def write_json_input_file():
    """
        write basic input json
    """
    basic = {
        "num_active_orbitals": 5,
        "num_active_electrons": 5,
        "hamiltonian_fname": "ham/FeNTA_s_1_cc-pvtz_5e_5o/ham_FeNTA_cc-pvtz_5e_5o.pickle",
        "spin": 1,
        "chkptfile_cas": "FeNTA_spin_1/basis_cc-pVTZ/CAS_5_5/mcscf.chk",
        "chkptfile_rohf": "FeNTA_spin_1/basis_cc-pVTZ/ROHF/scfref.chk",
        "basis": "cc-pVTZ",
        "atom": "FeNTA_spin_1/geo.xyz",
        "dmrg": 0,
        "dmrg_states": 200,
        "target": "",
        "optimizer_type": "scipy",
        "output_dir": "./files_afqmc",
        "file_wavefunction": "wf_6_+0.7500_+0.4998.dat",
        "ncore_electrons": 0,
        "generate_chol_hamiltonian": 1,
        "chol_cut": 1e-0,
        "dir_integral": "ham/FeNTA_s_1_cc-pvtz_5e_5o",
        "ipie_input_dir": "./ipie_fenta_input",
        "nwalkers": 10,
        "nsteps": 10,
        "nblocks": 10
    }
    with open("input_filename.json", "w") as f:
        f.write(json.dumps(basic, indent=4, separators=(",", ": ")))




