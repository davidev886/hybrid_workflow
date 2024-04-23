from pyscf import gto, scf
import h5py
import os
import numpy as np

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.ROHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
gen_ipie_input_from_pyscf_chk(mf.chkfile,
                              chol_cut=1e-1,
                              verbose=True)

from ipie.utils.chunk_large_chol import split_cholesky
split_cholesky('hamiltonian.h5', 4)  # split the cholesky to 4 subfiles

from mpi4py import MPI
from ipie.utils.mpi import MPIHandler, make_splits_displacements
from ipie.utils.pack_numba import pack_cholesky

nmembers = 1
comm = MPI.COMM_WORLD
num_walkers = 24 // comm.size
nsteps = 25
nblocks = 4
timestep = 0.005
rng_seed = None

with h5py.File('hamiltonian.h5', 'r') as fa:
    e0 = fa["e0"][()]
    hcore = fa["hcore"][()]

rank = comm.Get_rank()
size = comm.Get_size()
srank = rank % nmembers

handler = MPIHandler(nmembers=nmembers, verbose=True)

num_basis = hcore.shape[-1]
with h5py.File(f"chol_{srank}.h5", 'r') as fa:
    chol_chunk = fa["chol"][()]

chunked_chols = chol_chunk.shape[-1]
num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)

chol_chunk_view = chol_chunk.reshape((num_basis, num_basis, -1))
cp_shape = (num_basis * (num_basis + 1) // 2, chol_chunk_view.shape[-1])
chol_packed_chunk = np.zeros(cp_shape, dtype=chol_chunk_view.dtype)
sym_idx = np.triu_indices(num_basis)
pack_cholesky(sym_idx[0], sym_idx[1], chol_packed_chunk, chol_chunk_view)
del chol_chunk_view

split_size = make_splits_displacements(num_chol, nmembers)[0]
assert chunked_chols == split_size[srank]