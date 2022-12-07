from jax import numpy as jnp
from jax.scipy import linalg
import jax
from jax import jit, vmap, grad
from jax.ops import index_update, index
from jax.experimental import optimizers
from jax.config import config

from qutip import basis, tensor, qeye, sigmax, sigmay, sigmaz, Qobj
from qutip.superop_reps import to_kraus
from qutip.qip.operations import rx, ry

from qutip import basis, tensor, qeye, sigmax, sigmay, sigmaz, Qobj, fidelity, operator_to_vector, vector_to_operator, hinton
from qutip.superop_reps import to_kraus, to_chi, sprepost, kraus_to_choi, choi_to_chi, to_chi, _pauli_basis, choi_to_kraus, kraus_to_choi
from qutip.qip.operations import rx, ry, rz
from qutip.tomography import qpt_plot_combined
from qutip.tomography import qpt as qpt_matrix
# from qutip import qpt
import qutip as qt

import numpy as np
import scipy as sp
from scipy import linalg

from itertools import product


import matplotlib.pyplot as plt
from matplotlib import colors

from qpt_utils import tensor_product_list, dag, convert_to_jax, hyperplane_intersection_projection_switch_with_storage, proj_TP, proj_CP_threshold, plot_comparison

from tqdm.auto import tqdm

from qpt_utils import Choi

import haiku as hk
import optax

import cvxpy as cp


def resample_data(data, n=5000):
    shape = data.shape
    new_data = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pvals = data[i, j, :]
            new_data[i, j, :] = np.random.multinomial(n, pvals)/n
    return new_data

def resample_from_choose(data, n=5000):
    data_counts = data*n
    shape = data.shape
    new_data = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = [[k for n in range(int(data_counts[i,j,k]))] for k in range(shape[-1])]
            temp2 = [item for sublist in temp for item in sublist]
            temp2 = np.array(temp2)
            temp_data = np.random.choice(temp2, size=n, replace=True)
            new_data[i,j,:] = np.bincount(temp_data, minlength=8)/n
    return new_data
    

def return_rotations(angle, axis):
    """
    Return the noisy rotation operators measured from gate set tomography
    """
    axis = [axis[0]/np.linalg.norm(axis[0]),
            axis[1]/np.linalg.norm(axis[1]),
            axis[2]/np.linalg.norm(axis[2])]

    sigma_list = np.array([qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()])
    ry_p = np.cos(np.pi*angle[0]/2)*qeye(2).full() - 1j*np.sin(np.pi*angle[0]/2)*(np.sum(np.einsum('i,ijk->ijk', axis[0], sigma_list), axis=0))
    ry_2p = np.cos(np.pi*angle[1]/2)*qeye(2).full() - 1j*np.sin(np.pi*angle[1]/2)*(np.sum(np.einsum('i,ijk->ijk', axis[1], sigma_list), axis=0))
    rx_2m = np.cos(np.pi*angle[2]/2)*qeye(2).full() - 1j*np.sin(np.pi*angle[2]/2)*(np.sum(np.einsum('i,ijk->ijk', axis[2], sigma_list), axis=0))

    ry_p = Qobj(ry_p)
    ry_2p = Qobj(ry_2p)
    rx_2m = Qobj(rx_2m)

    return ry_p, ry_2p, rx_2m

def gen_new_rho0(rho=None, n=5000, thermal=False, T=None, dT=None, f0=None):
    """
    Generate a new input state sampling the pvalues of the thermal initial state from GST
    """
    if thermal:
        h_over_k = 4.799e-11 #s*K
        H = Qobj(np.array([[0, 0],[0, f0]]))
        T_new = np.random.normal(T, dT)
        new_rho0 = -h_over_k*H/T_new
        new_rho0 = new_rho0.expm().unit()
    else:
        pvals = [np.real(rho[0,0]), np.real(rho[1,1])]
        new_vals = np.random.multinomial(n, pvals)/n
        new_rho0 = Qobj(np.diag(new_vals)).unit()
    return new_rho0

def gen_new_probes(rhos_in, rots_noisy):
    """
    Generate the set of new probe states for inputing into the reconstructed process matrix
    """
    g_states = rhos_in
    e_states = [rots_noisy[i][0]*g_states[i]*rots_noisy[i][0].dag() for i in range(len(rots_noisy))]
    p_states = [rots_noisy[i][1]*g_states[i]*rots_noisy[i][1].dag() for i in range(len(rots_noisy))]
    i_states = [rots_noisy[i][2]*g_states[i]*rots_noisy[i][2].dag() for i in range(len(rots_noisy))]
    
    initial_errors = {i:{'g':g_states[i],'e':e_states[i],'p':p_states[i],'im':i_states[i]} for i in range(len(g_states))}


    corrected_probes = []

    for p in product(["g", "e", "p", "im"], repeat=len(g_states)):
        corrected_probes.append(tensor(initial_errors[0][p[0]],
                                    initial_errors[1][p[1]],
                                    initial_errors[2][p[2]]))
    return corrected_probes

def _apply_op(op, rho):
    """
    Applies a Krauss operator to an input state x.
    """
    return op@rho@dag(op)

apply_op = jit(vmap(vmap(_apply_op, in_axes = [0, None]), in_axes=[None, 0]))


@jit
def apply_process(ops, probes):
    """
    Applies the process tensor to the vector of density matrices
    
    Args:
        ops (ndarray): An array of k Krauss operators of shape (k, N, N)        
        
    Returns:
        v_out (ndarray): The list (array) of m density matrices each transformed
                         after applying the process.
    """    
    return jnp.sum(apply_op(ops, probes), axis=1)



def _apply_rotation(state: jnp.array, op: jnp.array)->jnp.array:
    """Applies a rotation to the input state (before measurement).

    Args:
        state (jnp.array) : The state (density matrix)
        op (jnp.array) : The rotation operator

    Returns:
        out (jnp.array): Rotated state
    """
    return op@state@dag(op)

apply_rotation = jit(vmap(vmap(_apply_rotation, in_axes=[None, 0]), in_axes=[0, None]))

def U_CZS(theta: float, phi: float, gamma: float)->jnp.array:
    """

    Args:
        theta (float): [description]
        phi (float): [description]
        gamma (float): [description]

    Returns:
        jnp.array: [description]
    """
    gamma_term = np.exp(1j*gamma)
    U = np.diag([1, 1, 1, 1]) + 0j
    U[1,1] = -gamma_term*np.sin(theta/2)**2 + np.cos(theta/2)**2
    U[2,2] = -gamma_term*np.cos(theta/2)**2 + np.sin(theta/2)**2
    U[3,3] = -gamma_term
    U[1,2] = 0.5*(1 + gamma_term)*np.exp(-1j*phi)*np.sin(theta)
    U[2,1] = 0.5*(1 + gamma_term)*np.exp(1j*phi)*np.sin(theta)
    return U

@jit
def U_CZS_jax(theta: float, phi: float, gamma: float)->np.array:
    """[summary]

    Args:
        theta (float): Gate parameter
        phi (float): Gate parameter
        gamma (float): Gate parameter

    Returns:
        np.array: U_CZS
    """
    gamma_term = jnp.exp(1j*gamma)
    U = jnp.diag(jnp.array([1, 1, 1, 1])) + 0j

    U = index_update(U, index[1, 1], -gamma_term*jnp.sin(theta/2)**2 + jnp.cos(theta/2)**2)
    U = index_update(U, index[2, 2], -gamma_term*jnp.cos(theta/2)**2 + jnp.sin(theta/2)**2)
    U = index_update(U, index[3, 3], -gamma_term)
    U = index_update(U, index[1, 2], 0.5*(1 + gamma_term)*jnp.exp(-1j*phi)*jnp.sin(theta))
    U = index_update(U, index[2, 1], 0.5*(1 + gamma_term)*jnp.exp(1j*phi)*jnp.sin(theta))

    return U

theta, phi, gamma = np.random.uniform(-np.pi, np.pi, size=3)
np.testing.assert_array_almost_equal(U_CZS(theta, phi, gamma), U_CZS_jax(theta, phi, gamma))



def U_CCZS(theta, phi, gamma):
    zero = qt.basis(2, 0).proj()
    one = qt.basis(2, 1).proj()

    UCZS = qt.Qobj(U_CZS(theta, phi, gamma))
    
    UCCZS = qt.tensor([zero, qt.qeye(4)]) + qt.tensor([one, UCZS])
    return UCCZS


@jit
def U_CCZS_jax(theta: float, phi: float, gamma: float)->jnp.array:
    """Jax version of the gate

    Args:
        theta (float): [description]
        phi (float): [description]
        gamma (float): [description]

    Returns:
        jnp.array: [description]
    """
    zero = jnp.array([[1., 0.],
                     [0., 0.]])

    one = jnp.array([[0., 0.],
                     [0., 1.]])
    const = jnp.kron(zero, jnp.eye(4))
    UCZS = U_CZS_jax(theta, phi, gamma)

    UCCZS = const + jnp.kron(one, UCZS)

    return UCCZS


@jit
def vec(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorize, or "vec", a matrix by column stacking.
    For example the 2 by 2 matrix A::
        A = [[a, b]
             [c, d]]
    becomes::
      |A>> := vec(A) = (a, c, b, d)^T
    where `|A>>` denotes the vec'ed version of A and :math:`^T` denotes transpose.
    :param matrix: A N (rows) by M (columns) numpy array.
    :return: Returns a column vector with  N by M rows.
    """
    return matrix.T.reshape((-1, 1))# matrix.T.reshape((-1, 1))


def gen_Amat(probes, rotations, POVMs):
    apply_rot_op = jit(vmap(vmap(lambda rot, povm: dag(rot)@povm@rot, in_axes = [0, None]), in_axes=[None, 0]))
    ops_measured = apply_rot_op(rotations, POVMs)
    Amat = []
    measurement_idx = (0, 4, 2, 6, 1, 5, 3, 7)

    for pidx in range(len(probes)):
        for ridx in range(len(rotations)):
            for i in range(len(measurement_idx)):
                operator = ops_measured[i, ridx]
                Amat += [vec(jnp.kron(probes[pidx], operator.T)).T[0]
                ]

    A = jnp.array(Amat)
    A_inv = jnp.linalg.inv((A.T@A))
    Aprod = A_inv@A.T
    return Aprod, A


def estimate_process(data, Aprod, rho_true=None):
    
    b = data.reshape((-1, 1))
    est = Aprod@b
    est = est.reshape((rho_true.shape))
    return np.array(est)

def run_reconstruction(rho, true_process, n_qubit=3):
    k = n_qubit # num qubits
    depo_rtol=1e-16
    depo_tol=1e-16

    true_process = true_process
    group = None

    options_proj={
            "maxiter": 300,
            "HIP_to_alt_switch": "first",
            "missing_w": 3,
            "min_part": 0.1,
            "HIP_steps": 10,
            "alt_steps": 4,
            "alt_to_HIP_switch": "cos",
            "min_cos": 0.99,
            "max_mem_w": 30,
            "genarg_alt": (1, 3, 20),
            "genarg_HIP": (5,),
        }
    all_dists=False
    first_CP_threshold_least_ev=True,
    all_dists=False,
    dist_L2=True,
    with_evs=False,
    keep_key_channels=False,
    keep_main_eigenvectors=0,
    save_intermediate=True
    proj = "HIPswitch"

    first_CP_threshold_least_ev=True,

    # CP estimator: first projection on CP matrices
    rhoCP, LS_least_ev = proj_CP_threshold(rho, full_output=True, thres_least_ev=first_CP_threshold_least_ev)
    ls_rel = -LS_least_ev * depo_rtol
    least_ev_x_dim2_tol = np.maximum(ls_rel, depo_tol)
    projection_with_storage = (hyperplane_intersection_projection_switch_with_storage)
    rho_list, least_ev_pls, _ = projection_with_storage(
                rhoCP,
                true_process,
                **options_proj,
                least_ev_x_dim2_tol=least_ev_x_dim2_tol,
                all_dists=all_dists,
                with_evs=with_evs,
                dist_L2=dist_L2,
                save_intermediate=save_intermediate,
        )
    return rho_list, least_ev_pls


