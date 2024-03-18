import numpy as np

import os
import sys
import json
from openfermion.linalg import get_sparse_operator
from openfermion.hamiltonians import s_squared_operator
from datetime import datetime
import h5py

from ipie.utils.io import write_hamiltonian, write_wavefunction
from ipie.utils.from_pyscf import load_from_pyscf_chkfile, generate_hamiltonian, copy_LPX_to_LXmn

from src.spin_square import of_spin_operator
from src.pyscf_scripts import normal_ordering_swap


def get_coeff_wf(final_state_vector, ncore_electrons=None, thres=1e-6):
    """
    :param final_state_vector: State vector from a VQE simulation
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
        if np.abs(final_state_vector[k]) >= thres:
            coeff.append(final_state_vector[k])
            occas.append(alpha_occ)
            occbs.append(beta_occ)

    # We need it non_normal ordered
    for i in range(len(coeff)):
        coeff[i] = (-1) ** (
            normal_ordering_swap([2 * j for j in occas[i]] + [2 * j + 1 for j in occbs[i]])) * \
                   coeff[i]
    ncore = ncore_electrons // 2
    core = [i for i in range(ncore)]
    occas = [np.array(core + [o + ncore for o in oa]) for oa in occas]
    occbs = [np.array(core + [o + ncore for o in ob]) for ob in occbs]

    return coeff, occas, occbs


class IpieInput(object):
    def __init__(self, options_file):
        with open(options_file) as f:
            options = json.load(f)

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
        self.str_date_0 = datetime.today().strftime('%Y%m%d_%H%M%S')
        self.str_date = options.get("data_dir", "")
        os.makedirs(self.str_date, exist_ok=True)

        self.n_qubits = 2 * self.num_active_orbitals

        self.file_path = self.str_date

        self.ncore_electrons = options.get("ncore_electrons", 0)

        self.n_alpha = int((self.num_active_electrons + self.spin) / 2)
        self.n_beta = int((self.num_active_electrons - self.spin) / 2)

    def gen_wave_function(self):
        """
            doc
        """
        num_active_orbitals = self.num_active_orbitals
        num_active_electrons = self.num_active_electrons
        filen_state_vec = self.filen_state_vec
        file_path = self.file_path
        ncore_electrons = self.ncore_electrons
        spin_s_square = get_sparse_operator(s_squared_operator(num_active_orbitals))
        spin_s_z = of_spin_operator("projected", 2 * num_active_orbitals)

        final_state_vector = np.loadtxt(filen_state_vec, dtype=complex)

        normalization = np.sqrt(np.dot(final_state_vector.T.conj(), final_state_vector))
        final_state_vector /= normalization

        spin_sq_value = final_state_vector.conj().T @ spin_s_square @ final_state_vector
        spin_proj = final_state_vector.conj().T @ spin_s_z @ final_state_vector

        print(spin_sq_value)
        print(spin_proj)
        coeff, occas, occbs = get_coeff_wf(final_state_vector, ncore_electrons)
        coeff = np.array(coeff, dtype=complex)
        ixs = np.argsort(np.abs(coeff))[::-1]
        coeff = coeff[ixs]
        occas = np.array(occas)[ixs]
        occbs = np.array(occbs)[ixs]

        write_wavefunction((coeff, occas, occbs),
                           os.path.join(file_path, f'trial_{len(coeff)}.h5'))

        n_alpha = len(occas[0])
        n_beta = len(occbs[0])
        with h5py.File(
                f"{file_path}/trial_{len(coeff)}.h5",
                'a') as fh5:
            fh5['active_electrons'] = num_active_electrons
            fh5['active_orbitals'] = num_active_orbitals
            fh5['nelec'] = (n_alpha, n_beta)

    def gen_hamiltonian(self,
                        hamil_file: str = "hamiltonian.h5",
                        verbose: bool = True,
                        chol_cut: float = 1e-5,
                        ortho_ao: bool = False,
                        mcscf: bool = False,
                        num_frozen_core: int = 0,
                        ) -> None:
        pyscf_chkfile = self.chkptfile_rohf
        if mcscf:
            scf_data = load_from_pyscf_chkfile(pyscf_chkfile, base="mcscf")
        else:
            scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
        mol = scf_data["mol"]
        # print(mol.nelec)

        hcore = scf_data["hcore"]
        ortho_ao_mat = scf_data["X"]
        mo_coeffs = scf_data["mo_coeff"]
        mo_occ = scf_data["mo_occ"]
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
        write_hamiltonian(ham.H1[0], copy_LPX_to_LXmn(ham.chol), ham.ecore, filename=os.path.join(self.file_path, hamil_file))


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)

    #
    filen_state_vec = "state_vec_11.dat"
    input_ipie = IpieInput(sys.argv[1])
    input_ipie.gen_hamiltonian(mcscf=True, chol_cut=1e-1)
    input_ipie.gen_wave_function()


if __name__ == "__main__":
    main()
