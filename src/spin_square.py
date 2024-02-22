from pyscf.fci import cistring, spin_square
from scipy.special import binom
import numpy as np
import openfermion as of

from src.pyscf_scripts import normal_ordering_swap


def of_spin_square_from_coeff(coeff, occas, occbs, n_spatial):
    """
    This function calculates the expectation value of S^2 by transforming the list of coefficients
    and orbital lists to an Openfermion object. Very slow as it's creating the 2**n wavefunction.

    :param coeff: list of CI coefficients (as in HSCI or ASCI)
    :param occas: list of lists of orbital occupations (alpha)
    :param occbs: list of lists of orbital occupations (beta)
    :param n_spatial: number of spatial orbitals
    :return: S^2 and Sz expectation value
    """
    n_so = 2*n_spatial
    wf = np.zeros(2 ** n_so, dtype=complex)

    for k in range(len(coeff)):
        binary = np.zeros(n_so, dtype=int)
        ll = list(2 * occas[k]) + list(2 * occbs[k] + 1)
        for x in ll:
            binary[x] = 1

        b = bin(int(''.join(map(str, binary)), 2))
        wf[int(b[2:int(n_so + 2)], 2)] = (-1) ** normal_ordering_swap(ll) * coeff[k]
    return spin_square_expectation_value(wf)


def pyscf_spin_square_from_coeff(coeff, occas, occbs, n_spatial):
    """
    This function calculates the expectation value of the spin square operator S^2
    from the coefficients, list of alpha, list of betas using pyscf machinery.
    :param coeff: list of coefficients
    :param occas: list of list of alphas
    :param occbs: list of list of betas
    :param n_spatial: number of spatial orbitals (full space)
    :return: S^2 expectation value
    """
    neleca = len(occas[0])
    nelecb = len(occbs[0])
    nelecm = max(neleca, nelecb)
    pos = int(binom(n_spatial, nelecm))
    civec = np.zeros(shape=(pos, pos))
    for i in range(len(coeff)):
        string = cistring.make_strings(occas[i], neleca)
        addra = cistring.str2addr(n_spatial, neleca, string[0])
        string = cistring.make_strings(occbs[i], nelecb)
        addrb = cistring.str2addr(n_spatial, nelecb, string[0])
        civec[addra, addrb] = coeff[i]

    return spin_square(civec, n_spatial, (neleca, nelecb))


# Spin square from cirq VQE wavefunction
def spin_square_expectation_value(input_1d_array_var):
    """
    Function to calculate S^2 expectation value from cirq state vector.
    :param input_1d_array_var: 1D array (of size 2**n) representing a cirq wavefunction
    :return: S^2 and Sz expectation value
    """
    # All DEFINITIONS of the operators can be found in Helgaker, Jorgensen, Olsen - page 38/39.
    # We assume the following spin-ordering: alpha, beta, alpha, beta, ... so that spin-orbital
    # index 0 corresponds to spatial orbital 0 and spin alpha, spin-orbital index 1 corresponds
    # to spatial orbital 0 and spin beta, spin-orbital index 0 corresponds to spatial orbital 1
    # and spin alpha, ...

    # we assume that the input vector is a power of 2: I have not included a specific check for
    # this, so be weary when using this function! However, there should always come some error if
    # the vector is not a power of two, because numpy will then complain about the dimensions not
    # being compatible.

    n_so = int(np.log2(input_1d_array_var.shape[0]))  # (even) number of spin orbitals
    assert n_so % 2 == 0, "Number of spin-orbitals should be an even."

    n_orb = n_so // 2  # number of spatial orbitals
    # define S_plus, S_minus and S_z operator to generate
    # S_square = S_x^2+S_y^2+S_z^2 = S_plus*S_minus + S_z(S_z-1).
    S_plus = of.FermionOperator.zero()
    S_z = of.FermionOperator.zero()
    for p in range(n_orb):
        S_plus += of.FermionOperator(((2 * p, 1), (2 * p + 1, 0),), 1)
        S_z += of.FermionOperator(((2 * p, 1), (2 * p, 0),), 1 / 2) + of.FermionOperator(
            ((2 * p + 1, 1), (2 * p + 1, 0),), -1 / 2)
    S_minus = of.utils.hermitian_conjugated(S_plus)  # S_plus and S_minus are related by conjugation
    S_square_fermion_op = S_plus * S_minus + S_z * (S_z - of.FermionOperator.identity())
    S_square_array = of.linalg.get_sparse_operator(S_square_fermion_op, n_so).toarray()
    S_z_array = of.linalg.get_sparse_operator(S_z, n_so).toarray()
    S_square_exp_val = np.matmul(input_1d_array_var.conj().T,
                                 np.matmul(S_square_array, input_1d_array_var))
    S_z_exp_val = np.matmul(input_1d_array_var.conj().T,
                            np.matmul(S_z_array, input_1d_array_var))
    return S_square_exp_val, S_z_exp_val
