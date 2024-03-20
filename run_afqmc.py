import os
import mpi4py
import numpy
import sys
import h5py
from ipie.hamiltonians.utils import get_hamiltonian

from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import get_input_value
from ipie.utils.mpi import get_shared_comm

mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"


def run_afqmc_code(ham_input_file, wf_input_file, measurements=None):
    """
    :param ham_input_file: (str) Input file for Hamiltonian
    :param wf_input_file: (str) Input file for the trial wavefunction
    :param measurements: (int) Define number of measurements (in Hadamard test). If None, exact
                               overlaps are used.
    :return: Generates output file for further analysis.
    """

    with h5py.File(f"{wf_input_file}.h5", 'r') as fh5:
        active_electrons = sum(fh5['active_electrons'][()])
        active_orbitals = fh5['active_orbitals'][()]
        nelec = fh5['nelec'][()]
        fh5.close()

    # AFQMC paramteres
    nwalkers = 1000
    nsteps = 10
    nblocks = 10
    seed = 7
    options = {
        "system": {
            "nup": nelec[0],
            "ndown": nelec[1],
        },
        "hamiltonian": {"name": "Generic",
                        "integrals": f"{ham_input_file}.h5"},
        "qmc": {
            "dt": 0.005,
            "nsteps": nsteps,
            "nwalkers": nwalkers,
            "blocks": nblocks,
            "batched": True,
            "rng_seed": seed,
        },
        "trial": {"filename": f"{wf_input_file}.h5",
                  "wicks": True,
                  "optimized": True,
                  "nact": active_orbitals,
                  "use_wicks_helper": False,
                  "ncas": active_electrons,
                  "calculate_variational_energy": True
                  },
        "estimators": {"filename": f"{wf_input_file}_results_measurements_{measurements}.h5"},
    }

    numpy.random.seed(seed)
    sys = Generic(nelec=nelec)
    comm = MPI.COMM_WORLD
    verbose = True
    shared_comm = get_shared_comm(comm, verbose=verbose)

    ham_opts = get_input_value(options, "hamiltonian", default={}, verbose=verbose)
    twf_opts = get_input_value(options, "trial", default={}, verbose=verbose)

    ham = get_hamiltonian(sys, ham_opts, verbose=True, comm=shared_comm)

    trial = get_trial_wavefunction(
        sys, ham, options=twf_opts, comm=comm, scomm=shared_comm, verbose=verbose
    )

    # This sets the number of measurements used. If None, exact overlaps will be used.
    trial.measurements = measurements

    trial.calculate_energy(sys, ham)  # this is to get the energy shift

    afqmc = AFQMC(
        comm=comm,
        system=sys,
        hamiltonian=ham,
        trial=trial,
        options=options,
        verbose=verbose,
    )
    afqmc.run(comm=comm, verbose=True)
    afqmc.finalise(verbose=True)


# Run this by e.g.
# python run_afqmc.py results/hamiltonian_ozone_sto-3g results/wf_ozone_sto-3g_1
if __name__ == "__main__":
    run_afqmc_code(sys.argv[1], sys.argv[2])
