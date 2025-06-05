import numpy as np
import numpy.typing as npt
from copy import deepcopy

# from python.utils import *
from python.Decomposition import *
from python.Contract import *
from python.Zippers import MPS_MPS_overlap
from python.utils import get_entropy
from python.Backend import Backend


def site_canonical_MPS(
    MPS: list[npt.NDArray[np.complex128]],
    loc: int = 0, Dcut: int | None = None,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray[np.complex128]]:
    
    """
    Site canonical form of MPS
    
       2|         2|  3|
    0 -- -- 1, 0 -- --- -- 1
    """
    
    len_MPS = len(MPS)
    assert loc < len_MPS, f"{loc=} >= {len(MPS)=}"
    
    check_mps(MPS, bk)
    
    copy: list[npt.NDArray[np.complex128]] = []
    for it, mps in enumerate(MPS):
        copy.append(bk.to_device(deepcopy(mps)))
    
    prod_left = bk.identity(copy[0].shape[0], dtype=bk.complex)
    prod_right = bk.identity(copy[-1].shape[1], dtype=bk.complex)
    
    for it in range(loc):
        matrix = Contract("ia,ajk->ijk", prod_left, copy[it], bk=bk)
        temp = bk.transpose(matrix, (0,2,1))
        temp = bk.reshape(temp, (-1, matrix.shape[1]))
        
        U, S, Vh = SVD(temp, Nkeep=Dcut, Skeep=1.e-8, bk=bk)
        
        copy[it] = bk.transpose(bk.reshape(U, (matrix.shape[0], matrix.shape[2], -1)), (0,2,1))
        prod_left = bk.matmul(bk.diag(S), Vh)
        
        assert left_isometry(copy[it], bk) < 1.e-6, f"copy[{it}] not left isometry\nleft_isometry(copy[{it}])={left_isometry(copy[it], bk)}\nU.conj().T@U={bk.matmul(bk.conj(U).T, U)}\nleft_prod_MPS(copy[{it}], bk)={left_prod_MPS(copy[it], bk)}"
    
    for it in range(loc+1, len_MPS)[::-1]:
        matrix = Contract("iak,aj->ijk", copy[it], prod_right, bk=bk)
        temp = bk.reshape(matrix, (matrix.shape[0], -1))
        
        U, S, Vh = SVD(temp, Nkeep=Dcut, Skeep=1.e-8, bk=bk)
        
        copy[it] = bk.reshape(Vh, (-1, matrix.shape[1], matrix.shape[2]))
        prod_right = bk.matmul(U, bk.diag(S))
        
        assert right_isometry(copy[it], bk) < 1.e-6, f"copy[{it}] not right isometry\nright_isometry(copy[{it}])={right_isometry(copy[it], bk)}\ncopy[{it}]={copy[it]}"
    
    copy[loc] = Contract("ia,abk,bj->ijk", prod_left, copy[loc], prod_right, bk=bk)

    return copy


def site_to_bond_canonical_MPS(
    orthogonality_center: npt.NDArray,
    isometry: npt.NDArray,
    dirc: str = "left",
    tol: float = 1e-6,
    bk: Backend = Backend('auto')
):
    
    ortho_shape0, ortho_shape1, ortho_shape2 = orthogonality_center.shape
    iso_shape0, iso_shape1, iso_shape2 = isometry.shape
    
    orthogonality_center = bk.to_device(orthogonality_center)
    isometry = bk.to_device(isometry)
    
    if dirc == "left":
        
        assert orthogonality_center.shape[0] == isometry.shape[1]
        assert left_isometry(isometry, bk) < tol, print(f"{left_isometry(isometry, bk)=}")    
        
        tensor = bk.reshape(orthogonality_center, (ortho_shape0, ortho_shape1 * ortho_shape2))
        U, Sigma, Vh = SVD(tensor, bk=bk)
        
        left_isometry_mps = Contract("iak,aj->ijk", isometry, U, bk=bk)
        right_isometry_mps = bk.reshape(Vh, (-1, ortho_shape1, ortho_shape2))
        
        assert left_isometry(left_isometry_mps, bk) < tol, print(f"{left_isometry(left_isometry_mps, bk)=}")
        assert right_isometry(right_isometry_mps, bk) < tol, print(f"{right_isometry(right_isometry_mps, bk)=}")
        assert left_isometry_mps.shape[1] == right_isometry_mps.shape[0]
    
    elif dirc == "right":
        
        assert orthogonality_center.shape[1] == isometry.shape[0]
        assert right_isometry(isometry, bk) < tol, print(f"{right_isometry(isometry, bk)=}")
                
        tensor = bk.reshape(bk.transpose(orthogonality_center, (0, 2, 1)), (ortho_shape0 * ortho_shape2, ortho_shape1))
        U, Sigma, Vh = SVD(tensor, bk=bk)
        
        left_isometry_mps = bk.transpose(bk.reshape(U, (ortho_shape0, ortho_shape2, -1)), (0, 2, 1))
        right_isometry_mps = Contract("ia,ajk->ijk", Vh, isometry, bk=bk)
        
        assert left_isometry(left_isometry_mps, bk) < tol, print(f"{left_isometry(left_isometry_mps, bk)=}")
        assert right_isometry(right_isometry_mps, bk) < tol, print(f"{right_isometry(right_isometry_mps, bk)=}")
        assert left_isometry_mps.shape[1] == right_isometry_mps.shape[0]
    
    # Convert back to numpy arrays if needed
    # if bk.lib == "torch":
    #     left_isometry_mps = bk.to_cpu(left_isometry_mps)
    #     right_isometry_mps = bk.to_cpu(right_isometry_mps)
    #     Sigma = bk.to_cpu(Sigma)
        
    return left_isometry_mps, right_isometry_mps, Sigma


def Gamma_Lambda_MPS(
    MPS: list[npt.NDArray],
    Dcut: int | None = None,
    verbose: bool = False,
    bk: Backend = Backend('auto')
):
    """
        2           2
        |           |
    0 -- -- 1   0 -- -- 1
    """
    check_mps(MPS, bk)
    len_MPS = len(MPS)
    
    norm = MPS_MPS_overlap(MPS, MPS)
    
    assert np.abs(norm - 1) < 1.e-6, f"{norm=} != 1, no canonical form exists"
    
    Gammas: list[npt.NDArray] = []
    Lambdas: list[list[npt.NDArray]] = [[bk.array([1]) for _ in range(2)] for _ in range(len_MPS)]
    
    right_can = site_canonical_MPS(MPS, loc=0)
    if verbose:
        print(f"Canonical form finished")
    
    left_prod = bk.identity(right_can[0].shape[0])
    for it in range(len_MPS):
        if verbose:
            print(f"{it}, {left_prod.shape=}, {right_can[it].shape=}")
        tensor = bk.tensordot(left_prod, right_can[it], axes=[(1), (0)])
        matrix = bk.transpose(tensor, (0,2,1)).reshape(
            -1, tensor.shape[1])

        U, S, Vh = SVD(matrix, Skeep=1.e-8, Nkeep=Dcut, bk=bk)
        
        if it < len_MPS-1:
            Lambdas[it][1] = S
            Lambdas[it+1][0] = S
        
        left_prod = bk.diag(S) @ Vh
        
        left_can = bk.transpose(
            U.reshape(
                tensor.shape[0], tensor.shape[2], -1), (0,2,1))
        
        assert left_isometry(left_can) < 1.e-6, f"left canonical error, {left_isometry(left_can)=}"
        
        Gammas.append(bk.tensordot(
            bk.diag(1/Lambdas[it][0]), left_can, axes=[(1), (0)]))
        
    return Gammas, Lambdas        


def dist_from_vidal_mps(
    Gammas: list[npt.NDArray],
    Lambdas: list[list[npt.NDArray]],
    return_list: bool = False,
    bk: Backend = Backend('auto')
) -> list | float:
    
    distance = 0
    length = len(Gammas)
    
    dists = [0.0 for _ in range(length)]
    
    for it, gamma in enumerate(Gammas):
        approx = left_gauge(gamma, Lambdas[it][0], bk)
        left = scale_inv_id(approx, bk)
        
        distance += left
        dists[it] += left / 2
        
        approx = right_gauge(gamma, Lambdas[it][1], bk)
        right = scale_inv_id(approx, bk)
        
        distance += right
        dists[it] += right / 2
    
    if return_list:
        return dists
    else:
        return distance / 2 / length


def get_Neumann_entropy(
    MPS: list[npt.NDArray],
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    """
    Get Neumann entropy of MPS
    
    return np.array([entropy at 1, entropy at site 2, ..., entropy at site L-1])
    
    """
    
    Neumann_entropy = []
    
    for it in range(len(MPS)-1):
        site_canonical = site_canonical_MPS(MPS, loc=it, bk=bk)
        _, _, sigma = site_to_bond_canonical_MPS(
            orthogonality_center=site_canonical[it],
            isometry=site_canonical[it+1],
            dirc = "right",
            bk = bk,
        )
        
        Neumann_entropy.append(get_entropy(sigma, bk = bk,))

    Neumann_entropy = bk.array(Neumann_entropy)
    
    return Neumann_entropy


def get_Neumann_entropy_from_left_isometry(
    left_isometry_MPS: list[npt.NDArray],
    bk: Backend = Backend("auto")
):
    
    length = len(left_isometry_MPS)
    
    Neumann_entropy = bk.zeros(length-1)
    
    right_Lambda = bk.identity(1)
    additional_left_idometry = bk.identity(1)
    
    for it in range(1, length):
        
        loc = length - it
        
        left_isometry = deepcopy(left_isometry_MPS[loc])
        
        tensor = Contract(
            "iak,ab,bj->ijk", left_isometry,
            additional_left_idometry, right_Lambda
        )
        
        matrix = tensor.reshape(tensor.shape[0], -1)
        
        U, S, Vh = SVD(matrix, full_SVD=True, bk=bk)
        
        Neumann_entropy[loc - 1] = get_entropy(S, bk)
        
        right_Lambda = bk.diag(S)
        additional_left_idometry = U
    
    return Neumann_entropy


def left_gauge(gamma, Lambda, bk):
    
    assert len(gamma.shape) == 3
    assert len(Lambda.shape) == 1
    
    return Contract(
            "bid,cjd,ba,ac->ij", gamma, gamma.conj(),
            bk.diag(Lambda), bk.diag(Lambda)
        )


def right_gauge(gamma, Lambda, bk):
    
    assert len(gamma.shape) == 3
    assert len(Lambda.shape) == 1
    
    return Contract(
            "iad,jcd,ab,bc->ij", gamma, gamma.conj(), 
            bk.diag(Lambda), bk.diag(Lambda)
        )


def scale_inv_id(approx, bk):
    
    id = bk.identity(approx.shape[0])
    
    tensor = approx/np.trace(approx)-id/np.trace(id)
    _, S, _ = SVD(tensor, bk=bk)
    
    return np.sum(S)


def check_mps(MPS: list[npt.NDArray], bk: Backend = Backend('auto')):
    
    len_MPS = len(MPS)
    
    for it, mps in enumerate(MPS):
        assert len(mps.shape) == 3, f"len(MPS[{it}].shape)={len(mps.shape)} != 3"
        assert mps.shape[0] == MPS[(it-1)%len_MPS].shape[1], f"MPS[{it}].shape[0]={mps.shape[0]} != {MPS[(it-1)%len_MPS].shape[1]=}"
        assert mps.shape[1] == MPS[(it+1)%len_MPS].shape[0], f"MPS[{it}].shape[1]={mps.shape[1]} != {MPS[(it+1)%len_MPS].shape[0]=}"


def find_site_loc(
    site_canonical: list[npt.NDArray], tol: float = 1e-8,
    bk: Backend = Backend('auto')
) -> int:
    
    for i, site in enumerate(site_canonical):        
        if left_isometry(site, bk) > tol and right_isometry(site, bk) > tol and check_site_canonical(site_canonical, i, bk) < tol:
            return i
    
    return None


def check_site_canonical(
    site_canonical: list[npt.NDArray], loc: int,
    bk: Backend = Backend('auto')
) -> float:
    
    non_isometry_max = 0.0
    
    for i, site in enumerate(site_canonical):        
        if i < loc:
            val = left_isometry(site, bk)
            if val > non_isometry_max:
                non_isometry_max = val
        
        elif i > loc:
            val = right_isometry(site, bk)
            if val > non_isometry_max:
                non_isometry_max = val
    
    return non_isometry_max


def get_only_isometry(
    site_canonical: list[npt.NDArray], loc1: int | None = None, loc2: int | None = None,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray]:
    
    if loc1 == None:
        loc1 = find_site_loc(site_canonical, bk=bk)
    
    only_isometry = []
    
    for i, site in enumerate(site_canonical):        
        if i == loc1:
            only_isometry.append(None)
        elif i == loc2:
            only_isometry.append(None)
        else:
            only_isometry.append(site)
    
    return deepcopy(only_isometry)


def move_site_left(
    site_canonical: list[npt.NDArray], loc: int | None = None,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray]:
    
    if loc == None:
        loc = find_site_loc(site_canonical, bk=bk)
    
    assert loc < len(site_canonical), f"{loc=} >= {len(site_canonical)=}"
    assert loc > 0, f"{loc=} <= 0"
    
    temp = []    
    for site in site_canonical:
        temp.append(bk.to_device(deepcopy(site)))
    
    current = temp[loc]
    left = temp[loc-1]
    
    matrix = bk.reshape(current, (current.shape[0], current.shape[1] * current.shape[2]))
    U, S, Vh = SVD(matrix, bk=bk)
    
    temp[loc] = bk.reshape(Vh, (-1, current.shape[1], current.shape[2]))
    temp[loc-1] = Contract("iak,ab,bj->ijk", left, U, bk.diag(S), bk=bk)
    
    assert right_isometry(temp[loc], bk) < 1.e-6, f"Move left error, {right_isometry(temp[loc], bk)=}"
    
    return temp
    
    
def move_site_right(
    site_canonical: list[npt.NDArray], loc: int | None = None,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray]:
    
    if loc == None:
        loc = find_site_loc(site_canonical, bk=bk)
    
    assert loc < len(site_canonical), f"{loc=} >= {len(site_canonical)=}"
    assert loc < len(site_canonical)-1, f"{loc=} >= {len(site_canonical)-1=}"
    
    temp = []    
    for site in site_canonical:
        temp.append(bk.to_device(deepcopy(site)))
        
    current = temp[loc]
    right = temp[loc+1]
    
    matrix = bk.reshape(bk.transpose(current, (0, 2, 1)), (current.shape[0]*current.shape[2], current.shape[1]))
    U, S, Vh = SVD(matrix, bk=bk)
    
    temp[loc] = bk.transpose(bk.reshape(U, (current.shape[0], current.shape[2], -1)), (0, 2, 1))
    temp[loc+1] = Contract("ia,ab,bjk->ijk", bk.diag(S), Vh, right, bk=bk)
    
    assert left_isometry(temp[loc], bk) < 1.e-6, f"Move right error, {left_isometry(temp[loc], bk)=}"

    return temp


def contract_MPS(MPS: list[npt.NDArray], bk: Backend = Backend('auto')) -> npt.NDArray:
    
    MPS = [bk.to_device(tensor) for tensor in MPS]
    absolute = MPS[0]
    
    for it in range(1, len(MPS)):
        absolute = bk.tensordot(absolute, MPS[it], axes=[(1), (0)])
        ord_absolute = len(absolute.shape)
        lst = [i for i in range(ord_absolute)]
        absolute = bk.transpose(absolute, rearrange_list_by_values(lst, [ord_absolute-2], [1]))
    
    absolute = bk.tensordot(absolute, bk.identity(absolute.shape[0]), axes=[(0,1), (0,1)])

    assert len(absolute.shape) == len(MPS)

    return bk.to_cpu(absolute) if bk.lib == "torch" else absolute


def left_isometry(single_MPS: npt.NDArray, bk: Backend = Backend('auto')) -> float:
    
    prod_val = left_prod_MPS(single_MPS, bk)
    ident_val_shape = single_MPS.shape[1] # As per current code

    # Determine dtype for converting prod_val to backend's array type
    # Use prod_val's own dtype if it's a valid numerical type, otherwise default to bk.complex
    prod_val_bk = bk.array(prod_val, dtype=bk.complex)

    # Create identity matrix of the same backend type and matching dtype as prod_val_bk
    ident_val_bk = bk.identity(ident_val_shape, dtype=prod_val_bk.dtype)

    numerator = bk.norm(prod_val_bk - ident_val_bk)
    
    # Denominator: norm of a consistently typed identity matrix
    denominator_ident = bk.identity(ident_val_shape, dtype=prod_val_bk.dtype)
    denominator = bk.norm(denominator_ident)
    
    if denominator == 0:
        # Fallback if norm of identity is zero (e.g., for a 0-dimensional identity)
        return bk.norm(prod_val_bk - ident_val_bk) 
    return numerator / denominator


def right_isometry(single_MPS: npt.NDArray, bk: Backend = Backend('auto')) -> float:
    
    prod_val = right_prod_MPS(single_MPS, bk)
    ident_val_shape = single_MPS.shape[0] # As per current code

    # Determine dtype for converting prod_val to backend's array type
    prod_val_bk = bk.array(prod_val, dtype=bk.complex)

    # Create identity matrix of the same backend type and matching dtype as prod_val_bk
    ident_val_bk = bk.identity(ident_val_shape, dtype=prod_val_bk.dtype)

    numerator = bk.norm(prod_val_bk - ident_val_bk)

    # Denominator: norm of a consistently typed identity matrix
    denominator_ident = bk.identity(ident_val_shape, dtype=prod_val_bk.dtype)
    denominator = bk.norm(denominator_ident)
    
    if denominator == 0:
        return bk.norm(prod_val_bk - ident_val_bk)
    return numerator / denominator


def left_prod_MPS(single_MPS: npt.NDArray, bk: Backend = Backend('auto')) -> npt.NDArray:
    
    return Contract("aib,ajb->ij", bk.conj(single_MPS), single_MPS, bk=bk)


def right_prod_MPS(single_MPS: npt.NDArray, bk: Backend = Backend('auto')) -> npt.NDArray:
    
    return Contract("iab,jab->ij", single_MPS, bk.conj(single_MPS), bk=bk)


def tensor_to_mps(tensor: npt.NDArray, bk: Backend = Backend('auto')) -> list[npt.NDArray]:
    
    """
    0 -- -- 1
        |
        2
    """
    
    tensor = bk.to_device(tensor)
    phys_leg_dims = np.array(tensor.shape)
    remaining_tensor = deepcopy(bk.reshape(tensor, (phys_leg_dims[0], -1)))
    
    MPS = []
    left_leg_dim = 1
    
    for it, phys_leg_dim in enumerate(phys_leg_dims[:-1]):
        U, S, Vh = SVD(remaining_tensor, Skeep=1.e-8, bk=bk)
        
        MPS.append(
            bk.transpose(bk.reshape(U, (left_leg_dim, phys_leg_dim, U.shape[1])), (0, 2, 1))
        )
        
        left_leg_dim = MPS[it].shape[1]
        after_tensor = bk.matmul(bk.diag(S), Vh)
        try:
            remaining_tensor = bk.reshape(after_tensor, (left_leg_dim * phys_leg_dims[it+1], -1))
        except Exception as e:
            # Minimal error reporting, or consider logging instead of extensive prints
            print(f"Error during tensor_to_mps at site {it}: {e}")
            raise # Re-raise the exception after printing minimal info
    
    MPS.append(
        bk.transpose(bk.reshape(remaining_tensor, (left_leg_dim, phys_leg_dims[-1], 1)), (0, 2, 1))
    )
    
    assert MPS[0].shape[0] == 1
    assert MPS[-1].shape[1] == 1
    
    return MPS

