import numpy as np
from src.pyscf_scripts import normal_ordering_swap


def prep_input_ipie(final_state_vector, ncore_electrons=None, thres=1e-6):
    """
    :param final_state_vector: Cirq object representing the state vector from a VQE simulation
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


def main():
    coeff, occas, occbs = prep_input_ipie(res.final_state_vector, ncore_electrons)
    coeff = np.array(coeff, dtype=np.complex)
    ixs = np.argsort(np.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occas = np.array(occas)[ixs]
    occbs = np.array(occbs)[ixs]
    print(pyscf_spin_square_from_coeff(coeff, occas, occbs, 6))

    with h5py.File(os.path.join(file_path, "coeffs.h5"), 'w') as hf:
        hf.create_dataset('coeffs', data=coeff)
        hf.create_dataset('occas', data=occas)
        hf.create_dataset('occbs', data=occbs)


if __name__ == "__main__":
    main()