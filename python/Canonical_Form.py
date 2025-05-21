import numpy as numpy
import numpy.typing as npt
from copy import deepcopy

# from python.utils import *
from python.Decomposition import *
from python.Contract import *
from python.Zippers import MPS_MPS_overlap


def site_canonical_MPS(
    MPS: list[npt.NDArray[np.complex128]],
    loc: int = 0, Dcut: int | None = None
) -> list[npt.NDArray[np.complex128]]:
    
    """
    Site canonical form of MPS
    
       2|         2|  3|
    0 -- -- 1, 0 -- --- -- 1
    """
    
    len_MPS = len(MPS)
    assert loc < len_MPS, f"{loc=} >= {len(MPS)=}"
    
    check_mps(MPS)
    
    copy: list[npt.NDArray[np.complex128]] = []
    for it, mps in enumerate(MPS):
        copy.append(deepcopy(mps))
    
    prod_left = np.identity(copy[0].shape[0], dtype=np.complex128)
    prod_right = np.identity(copy[-1].shape[1], dtype=np.complex128)
    
    for it in range(loc):
        matrix = Contract("ia,ajk->ijk", prod_left, copy[it])
        temp = matrix.transpose(0,2,1).reshape(-1, matrix.shape[1])
        
        U, S, Vh = SVD(temp, Nkeep=Dcut, Skeep=1.e-8)
        
        copy[it] = U.reshape(matrix.shape[0], matrix.shape[2], -1).transpose(0,2,1)
        prod_left = np.diag(S) @ Vh
        
        assert left_isometry(copy[it]) < 1.e-6, f"copy[{it}] not left isometry\nleft_isometry(copy[{it}])={left_isometry(copy[it])}\nU.conj().T@U={U.conj().T@U}\nleft_prod_MPS(copy[{it}])={left_prod_MPS(copy[it])}"
    
    for it in range(loc+1, len_MPS)[::-1]:
        matrix = Contract("iak,aj->ijk", copy[it], prod_right)
        temp = matrix.reshape(matrix.shape[0], -1)
        
        U, S, Vh = SVD(temp, Nkeep=Dcut, Skeep=1.e-8)
        
        copy[it] = Vh.reshape(-1, matrix.shape[1], matrix.shape[2])
        prod_right = U @ np.diag(S)
        
        assert right_isometry(copy[it]) < 1.e-6, f"copy[{it}] not right isometry\nright_isometry(copy[{it}])={right_isometry(copy[it])}\ncopy[{it}]={copy[it]}"
    
    copy[loc] = Contract("ia,abk,bj->ijk", prod_left, copy[loc], prod_right)
    
    return copy


def site_to_bond_canonical_MPS(
    orthogonality_center: npt.NDArray,
    isometry: npt.NDArray,
    dirc: str = "left",
    tol: float = 1e-6,
):
    
    ortho_shape0, ortho_shape1, ortho_shape2 = orthogonality_center.shape
    iso_shape0, iso_shape1, iso_shape2 = isometry.shape
    
    if dirc == "left":
        
        assert orthogonality_center.shape[0] == isometry.shape[1]
        assert left_isometry(isometry) < tol, print(f"{left_isometry(isometry)=}")    
        
        tensor = orthogonality_center.reshape(ortho_shape0, ortho_shape1 * ortho_shape2)
        U, Sigma, Vh = SVD(tensor)
        
        left_isometry_mps = Contract("iak,aj->ijk", isometry, U)
        right_isometry_mps = Vh.reshape(-1, ortho_shape1, ortho_shape2)
        
        assert left_isometry(left_isometry_mps) < tol, print(f"{left_isometry(left_isometry_mps)=}")
        assert right_isometry(right_isometry_mps) < tol, print(f"{right_isometry(right_isometry_mps)=}")
        assert left_isometry_mps.shape[1] == right_isometry_mps.shape[0]
    
    elif dirc == "right":
        
        assert orthogonality_center.shape[1] == isometry.shape[0]
        assert right_isometry(isometry) < tol, print(f"{right_isometry(isometry)=}")
                
        tensor = orthogonality_center.transpose(0, 2, 1).reshape(ortho_shape0 * ortho_shape2, ortho_shape1)
        U, Sigma, Vh = SVD(tensor)
        
        left_isometry_mps = U.reshape(ortho_shape0, ortho_shape2, -1).transpose(0, 2, 1)
        right_isometry_mps = Contract("ia,ajk->ijk", Vh, isometry)
        
        assert left_isometry(left_isometry_mps) < tol, print(f"{left_isometry(left_isometry_mps)=}")
        assert right_isometry(right_isometry_mps) < tol, print(f"{right_isometry(right_isometry_mps)=}")
        assert left_isometry_mps.shape[1] == right_isometry_mps.shape[0]
        
    return left_isometry_mps, right_isometry_mps, Sigma


def Gamma_Lambda_MPS(
    MPS: list[npt.NDArray],
    Dcut: int | None = None,
    verbose: bool = False,
):
    """
        2           2
        |           |
    0 -- -- 1   0 -- -- 1
    """
    check_mps(MPS)
    len_MPS = len(MPS)
    
    norm = MPS_MPS_overlap(MPS, MPS)
    
    assert np.abs(norm - 1) < 1.e-6, f"{norm=} != 1, no canonical form exists"
    
    Gammas: list[npt.NDArray] = []
    Lambdas: list[list[npt.NDArray]] = [[np.array([1]) for _ in range(2)] for _ in range(len_MPS)]
    
    right_can = site_canonical_MPS(MPS, loc=0)
    if verbose:
        print(f"Canonical form finished")
    
    # right_can = site_canonical_MPS(MPS, loc=0, Dcut=Dcut)
    
    # assert right_isometry(right_can[0]) < 1.e-6, f"right canonical error, {right_isometry(right_can[0])=}"
    
    left_prod = np.identity(right_can[0].shape[0])
    for it in range(len_MPS):
        if verbose:
            print(f"{it}, {left_prod.shape=}, {right_can[it].shape=}")
        tensor = Tensordot(left_prod, right_can[it], axes=[(1), (0)])
        matrix = tensor.transpose(0,2,1).reshape(
            -1, tensor.shape[1])

        U, S, Vh = SVD(matrix, Skeep=1.e-8, Nkeep=Dcut)
        # U, S, Vh = SVD(matrix, Skeep=1.e-8)
        
        if it < len_MPS-1:
            Lambdas[it][1] = S
            Lambdas[it+1][0] = S
        # if it == len_MPS-1:
        #     assert len(S.shape) == 1, f"{S=} != [1.0]"
        #     assert np.allclose(S, np.identity(1)), f"{S=} != [1.0]"
        
        left_prod = np.diag(S) @ Vh
        
        left_can = U.reshape(
            tensor.shape[0], tensor.shape[2], -1).transpose(0,2,1)
        
        assert left_isometry(left_can) < 1.e-6, f"left canonical error, {left_isometry(left_can)=}"
        
        Gammas.append(Tensordot(
            np.diag(1/Lambdas[it][0]), left_can, axes=[(1), (0)]))
        
    return Gammas, Lambdas        


def dist_from_vidal_mps(
    Gammas: list[npt.NDArray],
    Lambdas: list[list[npt.NDArray]],
    return_list: bool = False
) -> list | float:
    
    distance = 0
    length = len(Gammas)
    
    dists = [0.0 for _ in range(length)]
    
    for it, gamma in enumerate(Gammas):
        # print(f"\n{it}", end=" ")
        
        approx = left_gauge(gamma, Lambdas[it][0])
        left = scale_inv_id(approx)
        
        # print(f"left: {round_sig(left)}", end=" ")
        # print(f"\n{round_sig(approx)}")
        distance += left
        dists[it] += left / 2
        
        approx = right_gauge(gamma, Lambdas[it][1])
        right = scale_inv_id(approx)
        
        # print(f"right: {round_sig(right)}", end=" ")
        # print(f"\n{round_sig(approx)}")
        distance += right
        dists[it] += right / 2
    
    if return_list:
        return dists
    else:
        return distance / 2 / length


def left_gauge(gamma, Lambda):
    
    assert len(gamma.shape) == 3
    assert len(Lambda.shape) == 1
    
    return Contract(
            "bid,cjd,ba,ac->ij", gamma, gamma.conj(),
            np.diag(Lambda), np.diag(Lambda)
        )


def right_gauge(gamma, Lambda):
    
    assert len(gamma.shape) == 3
    assert len(Lambda.shape) == 1
    
    return Contract(
            "iad,jcd,ab,bc->ij", gamma, gamma.conj(), 
            np.diag(Lambda), np.diag(Lambda)
        )


def scale_inv_id(approx):
    
    id = np.identity(approx.shape[0])
    
    tensor = approx/np.trace(approx)-id/np.trace(id)
    _, S, _ = SVD(tensor)
    
    return np.sum(S)


def check_mps(MPS: list[npt.NDArray]):
    
    len_MPS = len(MPS)
    
    for it, mps in enumerate(MPS):
        assert len(mps.shape) == 3, f"len(MPS[{it}].shape)={len(mps.shape)} != 3"
        assert mps.shape[0] == MPS[(it-1)%len_MPS].shape[1], f"MPS[{it}].shape[0]={mps.shape[0]} != {MPS[(it-1)%len_MPS].shape[1]=}"
        assert mps.shape[1] == MPS[(it+1)%len_MPS].shape[0], f"MPS[{it}].shape[1]={mps.shape[1]} != {MPS[(it+1)%len_MPS].shape[0]=}"


def find_site_loc(
    site_canonical: list[npt.NDArray], tol: float = 1e-8
) -> int:
    
    for i, site in enumerate(site_canonical):        
        if left_isometry(site) > tol and right_isometry(site) > tol and check_site_canonical(site_canonical, i) < tol:
            return i
    
    return None


def check_site_canonical(
    site_canonical: list[npt.NDArray], loc: int
) -> float:
    
    non_isometry_max = 0.0
    
    for i, site in enumerate(site_canonical):        
        if i < loc:
            val = left_isometry(site)
            if val > non_isometry_max:
                non_isometry_max = val
                # print(f"Not site canonical\nsite {i} left isometry: {left_isometry(site)=}")
                # return False
        
        elif i > loc:
            val = right_isometry(site)
            if val > non_isometry_max:
                non_isometry_max = val
                # print(f"Not site canonical\nsite {i} right isometry: {right_isometry(site)=}")
                # return False
    
    return non_isometry_max


def get_only_isometry(
    site_canonical: list[npt.NDArray], loc1: int | None = None, loc2: int | None = None,
) -> list[npt.NDArray]:
    
    if loc1 == None:
        loc1 = find_site_loc(site_canonical)
        # print(f"{loc=}")
    
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
    site_canonical: list[npt.NDArray], loc: int | None = None
) -> list[npt.NDArray]:
    
    if loc == None:
        loc = find_site_loc(site_canonical)
        # print(f"{loc=}")
    
    assert loc < len(site_canonical), f"{loc=} >= {len(site_canonical)=}"
    assert loc > 0, f"{loc=} <= 0"
    
    temp = []    
    for site in site_canonical:
        temp.append(deepcopy(site))
    
    current = temp[loc]
    left = temp[loc-1]
    
    matrix = current.reshape(
        current.shape[0], current.shape[1] * current.shape[2])
    U, S, Vh = SVD(matrix)
    
    temp[loc] = Vh.reshape(
        -1, current.shape[1], current.shape[2])
    temp[loc-1] = Contract(
        "iak,ab,bj->ijk", left, U, np.diag(S)
    )
    
    assert right_isometry(temp[loc]) < 1.e-6, f"Move left error, {right_isometry(temp[loc])=}"
    
    return temp
    
    
def move_site_right(
    site_canonical: list[npt.NDArray], loc: int | None = None
) -> list[npt.NDArray]:
    
    if loc == None:
        loc = find_site_loc(site_canonical)
        # print(f"{loc=}")
    
    assert loc < len(site_canonical), f"{loc=} >= {len(site_canonical)=}"
    assert loc < len(site_canonical)-1, f"{loc=} >= {len(site_canonical)-1=}"
    
    temp = []    
    for site in site_canonical:
        temp.append(deepcopy(site))
        
    current = temp[loc]
    right = temp[loc+1]
    
    matrix = current.transpose(0, 2, 1).reshape(
        current.shape[0]*current.shape[2], current.shape[1])
    U, S, Vh = SVD(matrix)
    
    # print(f"{matrix.shape=} {U.shape=} {S.shape=} {Vh.shape=}")
    temp[loc] = U.reshape(
        current.shape[0], current.shape[2], -1
        ).transpose(0, 2, 1)
    temp[loc+1] = Contract(
        "ia,ab,bjk->ijk", np.diag(S), Vh, right
    )
    
    assert left_isometry(temp[loc]) < 1.e-6, f"Move right error, {left_isometry(temp[loc])=}"
    
    return temp


def contract_MPS(MPS: list[npt.NDArray]) -> npt.NDArray:
    
    absolute = MPS[0]
    
    for it in range(1, len(MPS)):
        absolute = Tensordot(absolute, MPS[it], axes=[(1), (0)])
        ord_absolute = len(absolute.shape)
        lst = [i for i in range(ord_absolute)]
        absolute = absolute.transpose(
            rearrange_list_by_values(lst, [ord_absolute-2], [1])
        )
    
    absolute = Tensordot(absolute, np.identity(absolute.shape[0]), axes=[(0,1), (0,1)])

    assert len(absolute.shape) == len(MPS)

    return absolute


def left_isometry(single_MPS: npt.NDArray) -> float:
    
    return np.linalg.norm(left_prod_MPS(single_MPS)-np.identity(single_MPS.shape[1]))/np.linalg.norm(np.identity(single_MPS.shape[1]))


def right_isometry(single_MPS: npt.NDArray) -> float:
    
    return np.linalg.norm(right_prod_MPS(single_MPS)-np.identity(single_MPS.shape[0]))/np.linalg.norm(np.identity(single_MPS.shape[0]))


def left_prod_MPS(single_MPS: npt.NDArray) -> npt.NDArray:
    
    return Contract("aib,ajb->ij", single_MPS.conj(), single_MPS)


def right_prod_MPS(single_MPS: npt.NDArray) -> npt.NDArray:
    
    return Contract("iab,jab->ij", single_MPS, single_MPS.conj())


def tensor_to_mps(tensor: npt.NDArray) -> list[npt.NDArray]:
    
    """
    0 -- -- 1
        |
        2
    """
    
    phys_leg_dims = np.array(tensor.shape)
    remaining_tensor = deepcopy(tensor.reshape(phys_leg_dims[0], -1))
    
    MPS = []
    left_leg_dim = 1
    
    for it, phys_leg_dim in enumerate(phys_leg_dims[:-1]):
        U, S, Vh = SVD(remaining_tensor, Skeep=1.e-8)
        
        MPS.append(
            U.reshape(left_leg_dim, phys_leg_dim, U.shape[1]).transpose(0, 2, 1)
        )
        
        left_leg_dim = MPS[it].shape[1]
        after_tensor = np.diag(S) @ Vh
        try:
            remaining_tensor = after_tensor.reshape(
                left_leg_dim * phys_leg_dims[it+1], -1
            )
        except Exception as e:
            print(e)
            print(f"{it=}\n{tensor.shape=}\n{left_leg_dim=}\n{phys_leg_dims=}")
            print(f"{remaining_tensor.shape=}")
            print(f"{after_tensor.shape=}")
            print(f"{U.shape=}")
            print(f"{S.shape=}")
            print(f"{Vh.shape=}")
            for it, mps in enumerate(MPS):
                print(f"{it}, {mps.shape=}")
    
    MPS.append(
        remaining_tensor.reshape(left_leg_dim, phys_leg_dims[-1], 1).transpose(0, 2, 1)
    )
    
    assert MPS[0].shape[0] == 1
    assert MPS[-1].shape[1] == 1
    
    return MPS

