import numpy as numpy
import numpy.typing as npt
from copy import deepcopy
import itertools
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
# import warnings
# warnings.filterwarnings('error')

# from python.utils import *
from python.Decomposition import *
from python.Contract import *
from python.overlap import *
from python.Canonical_Form import check_mps, dist_from_vidal_mps


def BP_for_MPS(
    MPS: list[npt.NDArray],
    Dcut: int | None = None,
    max_iter: int = 100,
    tol: float = 1.e-8,
    continuous: bool = True,
    return_iter: bool = False,
    normalized: bool = True,
) -> tuple[list[npt.NDArray], list[list[npt.NDArray]]] | tuple[list[npt.NDArray], list[list[npt.NDArray]], int]:
    
    check_mps(MPS)
    len_MPS = len(MPS)
    
    """
    Message[i,j]: Information flow from i to j
    """
    Gammas: list[npt.NDArray] = []
    Lambdas: list[list[npt.NDArray]] = [[0 for _ in range(2)] for _ in range(len_MPS)]
    Gamma_decoms: list[list[npt.NDArray]] = [[0 for _ in range(2)] for _ in range(len_MPS)]
    Messages: list[list[npt.NDArray]] = [[] for _ in range(len_MPS)]

    """
    Get nearest neighborhood
    """
    nn_points = [[] for _ in range(len_MPS)]
    for it in range(len_MPS):
        nn_points[it].append((it-1)%len_MPS)
        nn_points[it].append((it+1)%len_MPS)
    nn_points: npt.NDArray[np.int64] = np.array(nn_points)
    
    rng = np.random.default_rng()
    
    """
    Initialize message tensors
    """
    for loc in range(len_MPS):
        for dirc in range(2):
            tensor = MPS[loc]

            if not continuous and loc == 0 and dirc == 0:
                Messages[loc].append(np.identity(1))
                continue
            
            if not continuous and loc == len_MPS-1 and dirc == 1:
                Messages[loc].append(np.identity(1))
                continue
                        
            if dirc == 0:
                Messages[loc].append(Contract(
                    "iab,jab->ij", tensor, tensor.conj()))

            elif dirc == 1:
                Messages[loc].append(Contract(
                    "aib,ajb->ij", tensor, tensor.conj()))
    
    # print(f"{Messages[0][0]=}")
    # print(f"{Messages[-1][1]=}")
    
    """
    Update Message[i,j]: Information flow from point i to j
    """
    for it in range(max_iter):
        loc_list = list(itertools.product(range(len_MPS), range(1)))
        
        if not continuous:
            loc_list = [item for item in loc_list if item != (0, 0)]
            # loc_list = [item for item in loc_list if item != (len_MPS-1, 1)]
        
        rng.shuffle(loc_list)
        # print(f"{loc_list=}")
        
        new_messages: list[list[npt.NDArray]] = [[] for _ in range(len_MPS)]
        for i in range(len_MPS):
            new_messages[i].append(deepcopy(Messages[i][0]))
            new_messages[i].append(deepcopy(Messages[i][1]))
        
        for locs in loc_list:
            loc, dirc = locs
            
            new = update_message_MPS(
                loc, dirc, MPS, nn_points, new_messages
            )
            
            nn_loc = nn_points[loc][dirc]
            nn_dirc = (dirc+1)%2
            
            assert loc == nn_points[nn_loc][nn_dirc], f"{loc=} != nn_points[{nn_loc}][{nn_dirc}]={nn_points[nn_loc][nn_dirc]}"
            
            new_nn = update_message_MPS(
                nn_loc, nn_dirc, MPS, nn_points, new_messages)
            
            norm = Contract("ij,ij", new, new_nn)
            new = new / np.sqrt(norm)
            new_nn = new_nn / np.sqrt(norm)
            new_norm = Contract("ij,ij",new, new_nn)

            assert np.abs(new_norm-1) < 1.e-8, f"abs(norm-1){round_sig(np.abs(new_norm-1))} > 0"
            
            new_messages[loc][dirc] = new
            new_messages[nn_loc][nn_dirc] = new_nn
        
        error = 0
        for locs in loc_list:
            loc, dirc = locs
            
            new = new_messages[loc][dirc]
            ideal = update_message_MPS(
                loc, dirc, MPS, nn_points, new_messages
            )
            
            try:
                if normalized:
                    error += np.linalg.norm(new-ideal)/np.linalg.norm(ideal)
                else:
                    error += np.linalg.norm(new/np.trace(new)-ideal/np.trace(ideal))

            except Exception as e:
                print(f"{e}")
                print_traceback(e)
                print(f"{np.linalg.norm(nn_ideal)=}\n{ideal=}")
                error += np.linalg.norm(nn_new-ideal)
            
            nn_loc = nn_points[loc][dirc]
            nn_dirc = (dirc+1)%2
            
            nn_new = new_messages[nn_loc][nn_dirc]
            nn_ideal = update_message_MPS(
                nn_loc, nn_dirc, MPS, nn_points, new_messages)
            
            try:
                if normalized:
                    error += np.linalg.norm(nn_new-nn_ideal)/np.linalg.norm(nn_ideal)
                else:
                    error += np.linalg.norm(nn_new/np.trace(nn_new)-nn_ideal/np.trace(nn_ideal))

            except Exception as e:
                print(f"{e}")
                print_traceback(e)
                print(f"{np.linalg.norm(nn_ideal)=}\n{nn_ideal=}")
                error += np.linalg.norm(nn_new-nn_ideal)
        
        for i in range(len_MPS):
            Messages[i][0] = deepcopy(new_messages[i][0])
            Messages[i][1] = deepcopy(new_messages[i][1])
        
        # print(f"iter={it} {error=}")
        
        if error < tol:
            break
    
    # print(f"{Messages[0][0]=}")
    # print(f"{Messages[-1][1]=}")
    
    """
    Get root and inverse root of Message tensors
    """
    root_Messages, inv_root_Messages = get_root_and_inv(Messages)
    
    for loc in range(len_MPS):
        for dirc in range(2):
            nn_loc = nn_points[loc][dirc]
            nn_dirc = (dirc+1) % 2
            
            if type(Lambdas[loc][dirc]) == int:
                assert type(Lambdas[nn_loc][nn_dirc]) == int, f"Lambdas structure error"
                assert type(Gamma_decoms[loc][dirc]) == int, f"Gamma decomposition structure error"
                assert type(Gamma_decoms[nn_loc][nn_dirc]) == int, f"Gamma decomposition structure error"
                
                matrix = Contract(
                    "ia,ja->ij", root_Messages[loc][dirc], root_Messages[nn_loc][nn_dirc])
                
                U, Lambda, Vh = SVD(matrix, Nkeep=Dcut)
                
                Lambdas[loc][dirc] = Lambda
                Lambdas[nn_loc][nn_dirc] = Lambda
                Gamma_decoms[loc][dirc] = inv_root_Messages[loc][dirc] @ U
                Gamma_decoms[nn_loc][nn_dirc] = inv_root_Messages[nn_loc][nn_dirc] @ Vh.T
    
    for loc in range(len_MPS):
        Gammas.append(Contract(
            "abk,ai,bj->ijk", MPS[loc], *Gamma_decoms[loc]))

    if return_iter:
        return Gammas, Lambdas, it
    else:
        return Gammas, Lambdas


def update_message_MPS(
    loc: int, dirc: int, MPS: list[npt.NDArray], nn_points: npt.NDArray, messages: list[list[npt.NDArray]]
) -> npt.NDArray:
    
    tensor = MPS[loc]
            
    rest_dircs = [(dirc+1)%2]
    incoming_messages = []
    
    for rest_dirc in rest_dircs:
        incoming_loc = nn_points[loc][rest_dirc]
        incoming_dirc = (rest_dirc+1)%2
        incoming_messages.append(
            messages[incoming_loc][incoming_dirc]
        )
    
    if dirc == 0:
        return Contract(
            "iax,jbx,ab->ij", tensor, tensor.conj(), *incoming_messages)
    else:
        return Contract(
            "aix,bjx,ab->ij", tensor, tensor.conj(), *incoming_messages)


def get_root_and_inv(
    Messages: list[list[npt.NDArray]]
) -> tuple[list[list[npt.NDArray]], list[list[npt.NDArray]]]:
    
    len_MPS = len(Messages)
    
    root_Messages = [[] for _ in range(len_MPS)]
    inv_root_Messages = [[] for _ in range(len_MPS)]
    
    for loc, messages in enumerate(Messages):
        for it, message in enumerate(messages):
            eigvals, eigvecs = EIGH(message, Skeep=1.e-16)
            
            root = eigvecs @ np.diag(np.sqrt(eigvals))
            inverse_root = np.diag(np.sqrt(1/eigvals)) @ eigvecs.conj().T
            # try:
            #     print(f"pseudoinverse")
            #     inverse_root = sp.linalg.pinv(root)
            # except Exception as e:
            #     print(f"cond num={round_sig(np.linalg.cond(root))}")
            #     print(f"{e}")
            #     inverse_root = np.diag(np.sqrt(1/eigvals)) @ eigvecs.conj().T
            
            # eval = np.linalg.norm(root @ root.conj().T - message)/np.linalg.norm(message)
            # assert eval < 1.e-4, f"Messages[{loc}][{it}], {eval=}, {eigvals=}\n{root @ root.conj().T - message=}"
            
            # eval = np.linalg.norm(root @ inverse_root-np.identity(message.shape[0]))/np.linalg.norm(np.identity(message.shape[0]))
            # assert eval < 1.e-4, f"Messages[{loc}][{it}], cond num={round_sig(np.linalg.cond(root))}, {eval=}, {eigvals=}\n{root @ inverse_root=}"
            # assert np.allclose(
            #     inverse_root @ root, np.identity(message.shape[0])), f"{np.linalg.norm(root @ inverse_root-np.identity(message.shape[0]))/np.linalg.norm(root @ inverse_root)}"
            
            """
            The index of both root and inverse root is loc -> nn_points[loc][dirc]
            """
            root_Messages[loc].append(root.T)
            inv_root_Messages[loc].append(inverse_root.T)
    
    return root_Messages, inv_root_Messages


def sym_gauge_MPS(
    Gammas: list[npt.NDArray],
    Lambdas: list[list[npt.NDArray]],
) -> list[npt.NDArray]:

    sym_gauge_mps: list[npt.NDArray] = []    
    
    for it, tensors in enumerate(zip(Gammas, Lambdas)):
        
        gamma, Lambda = tensors
        
        sym_gauge_mps.append(Contract(
            "ai,abk,bj->ijk", np.diag(np.sqrt(Lambda[0])), gamma, np.diag(np.sqrt(Lambda[1]))))
    
    return sym_gauge_mps

