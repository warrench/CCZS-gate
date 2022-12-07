from jax import numpy as jnp
from jax import jit, vmap

from qutip import basis, tensor, qeye, sigmax, sigmay, sigmaz, Qobj
from qutip.superop_reps import to_kraus
from qutip.qip.operations import rx, ry

import numpy as np
import scipy as sp
from itertools import product


import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
from matplotlib import colors



def Choi(Kraus):
    """
    Takes the rank-r Kraus reprensentation of a channel
    and returns the Choi matrix of the channel.

    Input: (r, d, d)-array.
    Output $(d^2, d^2)$-array.
    """
    r, d, d = Kraus.shape
    vecKraus = Kraus.reshape(r, d ** 2)
    return np.einsum("ij, il -> jl", vecKraus, vecKraus.conj())
    # return np.einsum("ij, il -> jl", vecKraus / d, vecKraus.conj())


def convert_to_jax(arr: list)->jnp.array:
    """Converts QuTiP arrays to Jax arrays.

    Args:
        arr (list): A list of qutip objects to convert.

    Returns:
        jnp.array: Jax version of input arrays
    """
    return jnp.array([op.full() for op in arr])


@jit
def dag(op: jnp.array)->jnp.array:
    """Dagger operation on an operator

    Args:
        op (jnp.array): Any operator to take the dagger operation on.

    Returns:
        jnp.array: Conjuage transpose of the operator.
    """
    return jnp.conjugate(jnp.transpose(op))



@jit
def apply_chi(state: jnp.array,
              xi: float,
              pm: jnp.array,
              pn: jnp.array)->jnp.array:
    """Applies a transformation for one chi matrix element xi_mn, with the Pauli matrices pm, pn

    Args:
        state (jnp.array): Density matrix of the input state.
        xi (float): Chi matrix element the [m, n].
        pm, pn (jnp.array): Pauli matrix basis element (e.g., tensor product of ["X", "I", "Z"]).

    Returns:
        jnp.array: Density matrix of the output state.
    """
    return xi*pm@state@dag(pn)


def tensor_product_list(arr: list, repeat: int)->list:
    """Create a list with all tensor products of elements in arr.

    Uses the itertools.product function to construct all possible permutations.

    Args:
        arr (list): A list of qutip.Qobj representing states or operators.
        repeat (int): The number of elements to permute. 

    Returns:
        prod (list): Tensor products for all combination of elements in arr.
    """
    prod = []
    for p in product(arr, repeat=repeat):
        prod.append(tensor(*p))
    return prod


measurement_labels = ['000', '100', '010', '110', '001', '101', '011', '111']

ground = basis(2, 0)
excited = basis(2, 1)
plus = (ground + excited).unit()
imag = (ground + 1j*excited).unit()

# list(product(['0', '1', '+', "i"], repeat=3))
probes = tensor_product_list([ground*ground.dag(), excited*excited.dag(), plus*plus.dag(), imag*imag.dag()], repeat=3) 

# list(product(['X', 'Y', 'Z', "I"], repeat=3))
pauli_basis = tensor_product_list([sigmax(), sigmay(), sigmaz(), qeye(2)], repeat=3) 
chi_shape = (len(pauli_basis), len(pauli_basis)) # shape of the chi matrix

# X -> ry(-pi/2) single qubit rotation about y, Y -> Rotx(pi/2), Z -> Identity
# list(product(['X', 'Y', 'Z'], repeat=3))

X, Y, Z = ry(-np.pi/2), rx(np.pi/2), qeye(2)
rotations = tensor_product_list([X, Y, Z], repeat=3)

random_chi_matrix = np.random.uniform(size=chi_shape)


def apply_chi(state: Qobj, pauli_basis: list, chi: np.array)->Qobj:
    """Applies the chi matrix to a tensored n-qubit state using the Pauli basis.

    Args:
        state (Qobj): Input state as a tensor basis of single qubit states using QuTiP.
        pauli_basis (list): A list of all tensor products composed by the Pauli matrices and Identity.
        chi (np.array): A matrix of shape (N, N) where N is the length of the Pauli basis list.

    Returns:
        Qobj: The state after applying the chi matrix
    """
    for m,n in np.ndindex(chi.shape):
        state += chi[m, n]*pauli_basis[m]*state*pauli_basis[n]
    return state


def apply_kraus(state: Qobj, kraus_list: list)->Qobj:
    """Applies a list of Kraus operators to a tensored n-qubit state.

    Args:
        state (Qobj): Input state as a tensor basis of single qubit states using QuTiP.
        kraus_list (list): A list of all Kraus operators.

    Returns:
        Qobj: The state after applying the Kraus operators.
    """
    state_new = 0
    for op in kraus_list:
        state_new += op*state*op.dag()
    return state_new


def apply_rotation(state: Qobj, rot: Qobj)->Qobj:
    """Applies a rotation to the input state (before measurement)

    Args:
        state (Qobj): A tensor product of n qubit states
        rot (Qobj): A tensor product of n qubit rotation operations.

    Returns:
        out (Qobj): Rotated state as a tensor product of n qubit states
    """
    return rot*state


def apply_chi(state: Qobj, pauli_basis: list, chi: np.array)->Qobj:
    """Applies the chi matrix to a tensored n-qubit state using the Pauli basis.

    Args:
        state (Qobj): Input state as a tensor basis of single qubit states using QuTiP.
        pauli_basis (list): A list of all tensor products composed by the Pauli matrices and Identity.
        chi (np.array): A matrix of shape (N, N) where N is the length of the Pauli basis list.

    Returns:
        Qobj: The state after applying the chi matrix
    """
    for m,n in np.ndindex(chi.shape):
        state += chi[m, n]*pauli_basis[m]*state*pauli_basis[n]
    return state


def apply_kraus(state: Qobj, kraus_list: list)->Qobj:
    """Applies a list of Kraus operators to a tensored n-qubit state.

    Args:
        state (Qobj): Input state as a tensor basis of single qubit states using QuTiP.
        kraus_list (list): A list of all Kraus operators.

    Returns:
        Qobj: The state after applying the Kraus operators.
    """
    state_new = 0
    for op in kraus_list:
        state_new += op*state*op.dag()
    return state_new

def hyperplane_intersection_projection_switch_with_storage(
    rho,
    true_Choi,
    maxiter=100,
    free_trace=True,
    least_ev_x_dim2_tol=1e-2,
    all_dists=False,
    dist_L2=True,
    with_evs=False,
    save_intermediate=False,
    HIP_to_alt_switch="first",
    alt_to_HIP_switch="counter",
    min_cos=0.99,
    alt_steps=4,
    missing_w=1,
    min_part=0.3,
    HIP_steps=10,
    max_mem_w=30,
    verbose=False,
    **kwargs,
):
    """ Switches between alternate projections and HIP, with the following rules:
    * starts in alternate projections.
    * stays in alternate depending on alt_to_HIP_switch:
        ** if 'counter': uses an iterator (alt_steps) of the iteration number to determine the 
        number of consecutive steps before switching. If alt_steps
        is a number, yields this number. If a list cycles on the list.
        ** if 'cos':  switching when two
        successive steps are sufficiently colinear, namely if the cosinus of
        the vectors is at least min_cos.
    * stays in HIP depending on HIP_to_alt_switch:
        ** if 'first': stops HIP when the first active hyperplane
        of the sequence gets discarded. (ex: enter at iteration 7, then leaves when 
        the hyperplane of iteration 7 is not in w_act anymore).
        ** if 'missing', stops when a total of missing_w (default 1) hyperplanes are 
        deemed unnecessary. (ie w_act has lost missing_w member).
        ** if 'part': ends the loop if the length coeff_first * w_first is less than min_part 
        times the step size, ie the length of \sum coeffs_i w_i. This includes the case when
        the first hyperplane is deemed unnecessary, like in 'first'.
        ** if 'counter': uses an iterator (HIP_steps) of the iteration number to determine the 
        number of consecutive steps before switching. Iterator in input iter_choice. If 
        HIP_steps is a number, yields this number. If a list cycles on the list.
    """

    # loops = group.loops
    dim2 = len(rho)
    comp_time = 0
    # x_sq, xiwi = -1, 1 # For the first entry in the loop. Yields the impossible -1.
    sel = "alternate"  # Selector for the step; 'alternate' or 'HIP'.
    if alt_to_HIP_switch == "cos":
        w_norm_ancien = np.zeros(
            (dim2, dim2)
        )  # Not normalized to ensure at least two steps are takenp.
    elif alt_to_HIP_switch == "counter":
        past_al = 0  # number of steps already made in 'alternate' mode.
        alt_step_gen = step_generator(alt_steps)
        current_alt_step = next(alt_step_gen)
    else:
        raise ValueError('Unknown alt_to_HIP_switch. Must be "cos" or "counter".')

    if HIP_to_alt_switch == "counter":
        HIP_step_gen = step_generator(HIP_steps)
        past_HIP = 0
    elif HIP_to_alt_switch == "part":
        pass
    elif HIP_to_alt_switch == "first":
        pass
    elif HIP_to_alt_switch == "missing":
        missed = 0
    else:
        raise ValueError(
            'Unknown HIP_to_alt_switch. Must be "first", "missing", "part" or "counter".'
        )

    dims = (dim2, dim2)

    active = np.array([])
    nb_actives = 0
    XW = np.zeros((0, 0))
    w_act = np.zeros([0, dim2, dim2])
    target = np.array([])
    coeffs = np.array([])

    # rho is on CP, we first project on TP. Outside the loop because we also end on TP.
    rho = proj_TP(rho)

    if save_intermediate:
        rhoTP = [np.expand_dims(rho, 0)]

    least_ev_list = []

    for m in range(maxiter):
        rho_after_CP, least_ev = proj_CP_threshold(rho, free_trace, full_output=True)

        if save_intermediate:
            rhoTP.append(np.expand_dims(rho, 0))
        least_ev_list.append(least_ev)

        # Breaks here because the (- least_ev) might increase on the next rho
        if (sel == "alternate") or (m >= (maxiter - 2)):  # Ensures last ones are AP.
            if verbose:
                print("Alternate projections mode")
            # On TP and intersection with hyperplane
            if alt_to_HIP_switch == "cos":
                w_new = proj_TP(rho_after_CP) - rho
                norm_w = sp.linalg.norm(w_new)
                change = np.vdot(w_new / norm_w, w_norm_ancien).real > min_cos
                w_norm_ancien = w_new / norm_w

                # If change with alt_steps, the current projection is transformed into
                # the first HIP step.
                if change:
                    active = np.array([m])
                    nb_actives = 1
                    XW = np.array([[norm_w ** 2]])
                    w_act = np.array([w_new])
                    coeffs = np.array([sp.linalg.norm(rho - rho_after_CP) ** 2 / norm_w ** 2])
                    target = np.array([0.0])
                    rho += coeffs[0] * w_new
                else:
                    rho += w_new

            elif alt_to_HIP_switch == "counter":
                rho = proj_TP(rho_after_CP)
                past_al += 1
                change = past_al >= current_alt_step

                if change:
                    active = np.array([])
                    nb_actives = 0
                    XW = np.zeros((0, 0))
                    w_act = np.zeros([0, dim2, dim2])
                    target = np.array([])
                    coeffs = np.array([])

            if change:
                if HIP_to_alt_switch == "missing":
                    missed = 0
                elif HIP_to_alt_switch == "counter":
                    past_HIP = 0
                    current_HIP_step = next(HIP_step_gen)
                sel = "HIP"

        elif sel == "HIP":  # No other possibility
            if verbose:
                print(f"HIP mode. Active hyperplanes: {1 + nb_actives}")

            sq_norm_x_i = sp.linalg.norm(rho - rho_after_CP) ** 2
            w_i = proj_TP(rho_after_CP) - rho
            xiwi = sp.linalg.norm(w_i) ** 2

            XW = np.column_stack([XW, np.zeros(nb_actives)])
            XW = np.row_stack([XW, np.zeros(nb_actives + 1)])
            new_xw = np.einsum(
                "ij, kij -> k", w_i.conj(), w_act
            ).real  # Notice that the scalar product are all real
            # since the matrices are self-adjoint.
            XW[-1, :-1] = new_xw
            XW[:-1, -1] = new_xw
            XW[-1, -1] = xiwi
            target = np.r_[target, sq_norm_x_i]

            active = np.concatenate((active, [m]))
            w_act = np.concatenate([w_act, [w_i]])

            subset, coeffs = step2(XW, target)

            if HIP_to_alt_switch == "missing":
                missed += len(active) - len(
                    subset
                )  # Don't move this after the update to active !!!

            XW = XW[np.ix_(subset, subset)]
            active = active[subset]
            nb_actives = len(active)
            w_act = w_act[subset]
            target = np.zeros((nb_actives,))
            rho += np.einsum("k, kij -> ij", coeffs, w_act)

            if HIP_to_alt_switch in ["first", "part"]:
                if (
                    subset[0] != 0
                ) or nb_actives > max_mem_w:  # max_mem_w limits memory usage
                    change = True
                elif HIP_to_alt_switch == "part":
                    step_size = np.sqrt(np.einsum("i, ij, j", coeffs, XW, coeffs))
                    w_first_contrib = coeffs[0] * np.sqrt(XW[0, 0])
                    change = min_part * step_size >= w_first_contrib
                else:
                    change = False
            elif HIP_to_alt_switch in ["counter", "missing"]:

                # Limits memory usage
                if nb_actives > max_mem_w:
                    nb_actives -= 1
                    active = active[1:]
                    w_act = w_act[1:]
                    target = target[1:]
                    XW = XW[1:, 1:]
                    if HIP_to_alt_switch == "missing":
                        missed += 1
                # End max_mem_w case

                if HIP_to_alt_switch == "missing":
                    change = missed >= missing_w
                elif HIP_to_alt_switch == "counter":
                    past_HIP += 1
                    change = past_HIP >= current_HIP_step

            if change:
                if alt_to_HIP_switch == "cos":
                    w_norm_ancien = np.zeros(
                        (dim2, dim2)
                    )  # Ensures two alternate steps. Also possible to
                    # use w_norm_ancien = w_i / sqrt(xiwi)
                elif alt_to_HIP_switch == "counter":
                    past_al = 0
                    current_alt_step = next(alt_step_gen)
                sel = "alternate"

        else:
            raise ValueError('How did I get there? Typo on "HIP" or "alternate"?')

    return rhoTP, least_ev_list, m


def ensure_trace(eigvals):
    """
    Assumes sum of eigvals is at least one.

    Finds the value l so that $\sum (\lambda_i - l)_+ = 1$
    and set the eigenvalues $\lambda_i$ to $(\lambda_i - l)_+$.
    """
    trace = eigvals.sum()
    while trace > 1:
        indices_positifs = eigvals.nonzero()
        l = len(indices_positifs[0]) # Number of (still) nonzero eigenvalues
        eigvals[indices_positifs] += (1 - trace) / l  
        eigvals = eigvals.clip(0)
        trace = eigvals.sum() 
    return eigvals


def proj_CP_threshold(rho,  free_trace=True, full_output=False, thres_least_ev=False):
    """
    If thres_least_ev=False and free_trace=False, then projects rho on CP
    trace_one operators.
    
    More generally, changes the eigenvalues without changing the eigenvectors:
    * if free_trace=True and thres_least_ev=False, then projects on CP operators,
    with no trace condition.
    * if thres_least_ev=True, free_trace is ignored. Then we bound from below all 
    eigenvalues by their original value plus the least eigenvalue (which is negative).
    Then all the lower eigenvalues take the lower bound (or zero if it is negative),
    all the higher eigenvalues are unchanged, and there is one eigenvalue in the middle
    that gets a value between its lower bound and its original value, to ensure the
    trace is one.
    """
    eigvals, eigvecs = sp.linalg.eigh(rho) # Assumes hermitian; sorted from lambda_min to lambda_max
    
    least_ev = eigvals[0]
    
    if thres_least_ev:
        threshold = - least_ev # > 0
        evlow = (eigvals - threshold).clip(0)
        toadd = eigvals - evlow
        missing = 1 - evlow.sum()
        if missing < 0: # On this rare event, revert to usual projection
            eigvals = eigvals.clip(0)
            eigvals = ensure_trace(eigvals)
        else:
            inv_cum_toadd =  toadd[::-1].cumsum()[::-1]
            last_more_missing = np.where(inv_cum_toadd >= missing)[0][-1]
            eigvals[:last_more_missing] = evlow[:last_more_missing]
            eigvals[last_more_missing] = eigvals[last_more_missing] + missing - inv_cum_toadd[last_more_missing]    
    else:
        eigvals = eigvals.clip(0)
        if not free_trace:
            eigvals = ensure_trace(eigvals)
        #    
    indices_positifs = eigvals.nonzero()[0]    
    rho_hat_TLS = (eigvecs[:,indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:,indices_positifs].T.conj()
    
    if full_output==2:
        return rho_hat_TLS, least_ev, len(indices_positifs)
    elif full_output:
        return rho_hat_TLS, least_ev
    else:
        return rho_hat_TLS


def proj_TP(rho):
    """
    Projects the Choi matrix rho of a channel on trace-preserving channels.
    """
    d = np.sqrt(len(rho)).astype(int)
    partial_mixed = np.eye(d) / d
    
    # np.trace on the axes corresponding to the system
    correction = np.einsum('de, fg -> dfeg',partial_mixed, (partial_mixed - np.trace(rho.reshape(4 * [d]), axis1=0, axis2=2)))
    return rho + correction.reshape(d**2, d**2)

    
def step2(XW, target):
    """
    Finds a (big) subset of hyperplanes, including the last one, such that
    the projection of the current point on the intersection of the corresponding
    half-spaces is the projection on the intersection of hyperplanes.

    Input: XW is the matrix of the scalar products between the different 
    non-normalized normal directions projected on the subspace TP, written w_i
    in the main functions.
    target is the intercept of the hyperplanes with respect to the starting point,
    on the scale given by w_i.

    Outputs which hyperplanes are kept in subset, and the coefficients on their
    respective w_i in coeffs.
    """
    nb_active = XW.shape[0]
    subset = np.array([nb_active - 1])
    coeffs = [target[-1] / XW[-1, -1]] # Always positive
    for i in range(nb_active - 2, -1, -1):
        test = (XW[i, subset].dot(coeffs) < target[i])
        # The condition to project on the intersection of the hyperplanes is that 
        # all the coefficients are non-negative. This is equivalent to belonging
        # to the normal cone to the facet.
        if test:
            subset = np.r_[i, subset]
            coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset]) 
            # Adding a new hyperplane might generate negative coefficients.
            # We remove the corresponding hyperplanes, except if it is the last 
            # hyperplane, in which case we do not add the hyperplane.
            if coeffs[-1] < 0: 
                subset = subset[1:]
                coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset]) 
            elif not np.all(coeffs >= 0):
                subset = subset[np.where(coeffs >= 0)]
                coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset])
    
    return subset, coeffs



def plot_comparison(a, b, title=""):
    """Plot a comparision between Choi matrices

    Args:
        a ([type]): [description]
        b ([type]): [description]
    """
    plt.clf()
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (7.5, 7.5))

    cmap = "viridis"

    norm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(a)), vcenter=0, vmax=np.max(np.abs(a)))

    ax[0, 0].matshow(a.real, cmap=cmap, norm=norm)
    im = ax[0, 1].matshow(b.real, cmap=cmap, norm=norm)


    ax[1, 0].matshow(a.imag, cmap=cmap, norm=norm)
    im = ax[1, 1].matshow(b.imag, cmap=cmap, norm = norm)
    plt.colorbar(im, ax=[axis for axis in ax.ravel()], fraction=0.046, pad=0.04)

    ax[0, 0].set_title("Choi", fontsize=16)
    ax[0, 1].set_title("Choi (reconstructed)", fontsize=16)

    ax[0, 0].set_ylabel(r"Re[$\Phi$]", fontsize=16)
    # ax[0, 1].set_xlabel("Re")

    ax[1, 0].set_ylabel(r"Im[$\Phi$]", fontsize=16)
    # ax[1, 1].set_xlabel("Im")

    plt.suptitle(title, y=0.95, fontsize=20)
    plt.show()