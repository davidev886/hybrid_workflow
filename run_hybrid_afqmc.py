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

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked, ParticleHole

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
        self.mol_nelec = None
        self.trial_name = None

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

        final_state_vector = convert_state_big_endian(final_state_vector)

        normalization = np.sqrt(np.dot(final_state_vector.T.conj(), final_state_vector))
        final_state_vector /= normalization

        spin_sq_value = final_state_vector.conj().T @ spin_s_square @ final_state_vector
        spin_proj = final_state_vector.conj().T @ spin_s_z @ final_state_vector

        print(spin_sq_value)
        print(spin_proj)
        print(ncore_electrons)
        coeff, occbs, occas = get_coeff_wf(final_state_vector, ncore_electrons)
        coeff = np.array(coeff, dtype=complex)
        ixs = np.argsort(np.abs(coeff))[::-1]
        coeff = coeff[ixs]
        occas = np.array(occas)[ixs]
        occbs = np.array(occbs)[ixs]
        self.trial_name = f'trial_{len(coeff)}.h5'
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
        """
        adapted function gen_ipie_input_from_pyscf_chk from ipie/utils/from_pyscf.py
        """
        pyscf_chkfile = self.chkptfile_rohf
        if mcscf:
            scf_data = load_from_pyscf_chkfile(pyscf_chkfile, base="mcscf")
        else:
            scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
        mol = scf_data["mol"]
        self.mol_nelec = mol.nelec
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


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)

    input_ipie = IpieInput(sys.argv[1])

    input_ipie.gen_hamiltonian(mcscf=True, chol_cut=1e-5)
    input_ipie.gen_wave_function()

    input_ipie.check_energy_state()
    with h5py.File("hamiltonian.h5") as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    num_basis = chol.shape[1]
    system = Generic(nelec=input_ipie.mol_nelec)

    num_chol = chol.shape[0]
    ham = HamGeneric(
        np.array([h1e, h1e]),
        chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    # Build trial wavefunction
    with h5py.File(os.path.join(input_ipie.file_path, input_ipie.trial_name), "r") as fh5:
        coeff = fh5["ci_coeffs"][:]
        occa = fh5["occ_alpha"][:]
        occb = fh5["occ_beta"][:]

    wavefunction = (coeff, occa, occb)
    trial = ParticleHoleNonChunked(
        wavefunction,
        input_ipie.mol_nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    afqmc_msd = AFQMC.build(
        input_ipie.mol_nelec,
        ham,
        trial,
        num_walkers=10,
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


if __name__ == "__main__":
    main()
