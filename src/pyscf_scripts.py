import numpy as np
import copy as cp


from pyscf import mcscf, fci
from pyscf.hci import hci
from pyscf.hci.hci import to_fci
from pyscf.fci.addons import overlap


def get_ground_state_overlap(mf, civec, active_electrons, active_orbitals, method='fci'):
    """
    Calculates the (approximate) ground state and determines the overlap with the trial
    wavefunction.
    :param mf: pyscf MF object
    :param method: method to calculate (approximate) ground state, currently only "fci" and "hsci"
                   available
    """
    # Get ground state (in active space)
    mc = mcscf.CASCI(mf, active_orbitals, active_electrons)
    assert sum(mc.nelecas) == active_electrons, "electrons in active space does not match"

    h1, core_energy = mc.get_h1eff()
    h2 = mc.get_h2eff()
    if method == 'fci':
        cisolver = fci.FCI(mf.mol)
    elif method == 'hsci':
        cisolver = hci.SCI(mf.mol)
        cisolver.select_cutoff = 1e-8

    e, gs_civec = cisolver.kernel(h1, h2, active_orbitals, nelec=mf.mol.nelec, ecore=core_energy)
    fcivec_trial = to_fci(civec, active_orbitals, mc.nelecas)

    return overlap(gs_civec, fcivec_trial, active_orbitals, active_electrons)


def calculate_orbital_lists(mf, active_orbitals, active_electrons):
    """
    :param mf: mean field object from pyscf
    :param active_orbitals: number of active orbitals to consider
    :param active_electrons: number of active electrons to consider
    :returns: two lists: active orbitals, frozen orbitals
    """
    n_core_electrons = sum(mf.mol.nelec) - active_electrons
    n_core_orbitals = n_core_electrons // 2

    active_orbitals = list(range(n_core_orbitals, n_core_orbitals + active_orbitals))
    frozen_orbitals = list(set(list(range(mf.mol.nao))) - set(active_orbitals))
    return active_orbitals, frozen_orbitals


def normal_ordering_swap(l):
    """
    This function normal calculates the phase coefficient (-1)^swaps, where swaps is the number of
    swaps required to normal order a given list of numbers.
    :param l: list of numbers, e.g. orbitals
    :returns: number of required swaps
    """
    count = 0
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]:
                count += 1

    return count


def shrink_hsci_vec(civec, k):
    """
    This function sets the k smallest (in magnitude) values of a hsci vector to zero.
    Add for what this function can be used.
    :param civec: civec from naive pyscf HCI
    :param k: number of (smallest) values to be set to zero
    :returns: shrinked civec
    """
    civec_cp = cp.deepcopy(civec)
    for x in [np.argpartition(np.abs(civec_cp[0]), -k)][0].tolist()[:-k]:
        civec_cp[0][x] = 0

    norm = sum(np.abs(civec_cp[0].tolist()) ** 2)
    civec_cp[0] = civec_cp[0] / np.sqrt(norm)
    return civec_cp


def eri_spin_from_spatial(eri):
    """
    :param eri: eri in chemistry notation and spatial orbitals,
        i.e. eri_{pqrs} a_p^dag(r1) a_q(r1) a_r^dag(r2)a_s(r2)
    :return: h0/eri in spin orbitals. Ordering is (alpha, beta, alpha, beta...)
    """
    n_so = 2 * eri.shape[0]

    # Initialize Hamiltonian coefficients.
    eri_spin = np.zeros((n_so, n_so, n_so, n_so))

    # Loop through integrals.
    for p in range(n_so // 2):
        for q in range(n_so // 2):
            for r in range(n_so // 2):
                for s in range(n_so // 2):
                    # Mixed spin
                    eri_spin[2 * p + 1, 2 * q + 1, 2 * r, 2 * s] = eri[p, q, r, s]  # alpha - beta
                    eri_spin[2 * p, 2 * q, 2 * r + 1, 2 * s + 1] = eri[p, q, r, s]  # beta - alpha

                    # Same spin
                    # alpha - alpha
                    eri_spin[2 * p, 2 * q, 2 * r, 2 * s] = eri[p, q, r, s]
                    # beta - beta
                    eri_spin[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = eri[p, q, r, s]

    return eri_spin
