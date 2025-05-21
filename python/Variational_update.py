import numpy as numpy
import numpy.typing as npt
from copy import deepcopy

# from python.utils import *
from python.Decomposition import *
from python.Contract import *
from python.Canonical_Form import *
from python.Zippers import *
from python.Projectors import get_projector


def Variational_contraction(
    MPS: list[npt.NDArray],
    MPO: list[npt.NDArray],
    Dcut: int | None = None,
    norm_Keep: float = 1.e-1,
    delta: float = 0.1,
    start_loc: int | None = None,
    phy_leg_const: bool = True,
    max_repeat: int = 10,
    min_repeat: int = 2,
    tol: float = 1.e-8,
    verbose: bool = False,
    educated_guess: bool = False,
    mode: str = "two_site",
) -> list[npt.NDArray]:
    
    """
    MPS order
    
    0 -- o -- 1
         |
         2
    
    MPO order

         3
         |
    0 -- o -- 1
         |
         2
    """
    
    assert len(MPS) == len(MPO), f"{len(MPS)=} != {len(MPO)=}"
    assert mode == "single_site" or "two_site" or "CBE", f"{mode=}"
    
    for i in range(len(MPS)):
        assert MPS[i].shape[2] == MPO[i].shape[3], f"MPS[{i}].shape[2]={MPS[i].shape[2]} != MPO[{i}].shape[3]={MPO[i].shape[3]}"
    
    last = len(MPS)-1
    
    if start_loc == None:
        # start_loc: int = int((last-1)/2)
        start_loc: int = 0
    
    if verbose:
        print(f"Initialization", end=" ")
        now = time.perf_counter()
    
    if phy_leg_const:
    # * If there is no change in leg dimension, direct initialization is possible
        new_MPS: list[npt.NDArray] = site_canonical_MPS(MPS, Dcut=Dcut, loc=start_loc)
    
    else:
    # * If the leg dimension changes, we have to take a guess during initialization
        new_MPS = initial_guess(MPS, MPO, Dcut, educated=educated_guess)
        new_MPS = site_canonical_MPS(new_MPS, Dcut=Dcut, loc=start_loc)

    if verbose:
        print(f"{round_sig(time.perf_counter()-now)}s")

    loc_list = []
    dirc_list = []
        
    loc_list.extend(list(range(start_loc, last)))
    dirc_list.extend(["right" for _ in range(start_loc, last)])
    loc_list.extend(list(range(1, last+1)[::-1]))
    dirc_list.extend(["left" for _ in range(1, last+1)])
    loc_list.extend(list(range(0, start_loc)))
    dirc_list.extend(["right" for _ in range(0, start_loc)])
        
    """
    Contraction from left and right
    left: [identity, col_0, col_1, ..., col_last]
    right: [identity, col_last, col_last-1, ..., col_0]
    """
    
    contract_list_left: list[npt.NDArray] = [np.array([1.0]).reshape(1, 1, 1) for _ in range(len(MPS)+1)]
    contract_list_right: list[npt.NDArray] = [np.array([1.0]).reshape(1, 1, 1) for _ in range(len(MPS)+1)]
    
    if verbose:
        print(f"Contracted list preparation", end=" ")
        now = time.perf_counter()
    
    for it in range(1, start_loc+1):
        contract_list_left[it] = Contract(
            "abc,aix,bjyx,cky->ijk", contract_list_left[it-1], MPS[it-1], MPO[it-1], new_MPS[it-1].conj())
        
    for it in range(1, len(MPS)-start_loc):
        contract_list_right[it] = Contract(
            "iax,jbyx,kcy,abc->ijk", MPS[len(MPS)-it], MPO[len(MPS)-it], new_MPS[len(MPS)-it].conj(), contract_list_right[it-1])

    if verbose:
        print(f"{round_sig(time.perf_counter()-now)}s")
        
    for repeat in range(max_repeat):
        if verbose:
            print(f"Sweep={repeat+1}")
            now = time.perf_counter()

        origin = []
        for mps in new_MPS:
            origin.append(deepcopy(mps))

        for loc, dirc in zip(loc_list, dirc_list):
            if verbose:
                print(f"({loc}, {dirc})", end=" ")
            # new_MPS = single_site(
            #     MPS=MPS, MPO=MPO, new_MPS=new_MPS, loc=loc, dirc=dirc
            # )
            
            if mode == "single_site":
                new_MPS, contract_list_left, contract_list_right = single_site_via_storing(
                    MPS=MPS, MPO=MPO, new_MPS=new_MPS, loc=loc, dirc=dirc,
                    contract_list_left=contract_list_left, contract_list_right=contract_list_right
                )
            
            elif mode == "two_site":
                new_MPS, contract_list_left, contract_list_right = two_site_via_storing(
                    MPS=MPS, MPO=MPO, new_MPS=new_MPS, loc=loc, dirc=dirc, Dcut=Dcut, norm_Keep=norm_Keep,
                    contract_list_left=contract_list_left, contract_list_right=contract_list_right, verbose=verbose
                )

            elif mode == "CBE":
                new_MPS, contract_list_left, contract_list_right = CBE_DMRG(
                    MPS=MPS, MPO=MPO, new_MPS=new_MPS, loc=loc, dirc=dirc, Dcut=Dcut, delta = delta, norm_Keep=norm_Keep,
                    contract_list_left=contract_list_left, contract_list_right=contract_list_right
                )
            
            else:
                sys.exit(f"{mode=} is not 'single_site', 'two_site' or 'CBE'")
        
        if verbose:
            print(f"site update: {round_sig(time.perf_counter()-now)}s,", end=" ")
            now = time.perf_counter()
        
        new = np.abs(MPS_MPS_overlap(new_MPS, origin))
        old = np.abs(MPS_MPS_overlap(origin, origin))

        if old > 0:
            deviation = new / old - 1
            
            if verbose:
                print(f"deviation = {round_sig(deviation)}, {round_sig(time.perf_counter()-now)}s")
            
            if deviation < tol and repeat >= min_repeat:
                break
        else:
            if verbose:
                print(f"deviation = {round_sig(np.abs(MPS_MPS_overlap(origin, origin)))}")
            
            if repeat >= min_repeat:
                break
        
    return new_MPS


def initial_guess(
    MPS: list[npt.NDArray], MPO: list[npt.NDArray], Dcut: int, educated: bool = False,
):
    
    # * If the leg dimension changes, we have to take a guess during initialization
    new_MPS: list[npt.NDArray] = []
    projectors: list[list] = [[0 for _ in range(2)] for _ in range(len(MPS))]
    
    norm = MPS_MPS_overlap(MPS, MPS, conj=True)
    
    if educated == False:
        # * Random guess
        for it, tensors in enumerate(zip(MPS, MPO)):
            mps, mpo = tensors
            new_MPS.append(np.random.rand(mps.shape[0], mps.shape[1], mpo.shape[2])+1j*np.random.rand(mps.shape[0], mps.shape[1], mpo.shape[2]))
    else:
        # * Educated guess
        for it, tensors in enumerate(zip(MPS, MPO)):
            mps, mpo = tensors
                    
            if type(projectors[it][0]) == int:
                nn_it = (it-1)%len(MPS)
                assert type(projectors[nn_it][1]) == int
                mps_left, mpo_left = MPS[nn_it], MPO[nn_it]
                
                size = mps.shape[0] * mpo.shape[0] * mps_left.shape[1] * mpo_left.shape[1]
                
                if size > 2**16:
                    print(f"projector {size=} ")
                
                P1, P2, _, _, _, _ = get_projector(
                        mps.transpose(0, 2, 1),
                        mpo.transpose(0, 3, 1, 2),
                        mps_left.transpose(1, 2, 0),
                        mpo_left.transpose(1, 3, 0, 2),
                        Dcut=Dcut, get_loss=False
                    )

                projectors[it][0] = P1
                projectors[nn_it][1] = P2
            
            if type(projectors[it][1]) == int:
                nn_it = (it+1)%len(MPS)
                assert type(projectors[nn_it][0]) == int
                mps_right, mpo_right = MPS[nn_it], MPO[nn_it]
                
                size = mps.shape[0]
                
                P1, P2, _, _, _, _ = get_projector(
                        mps.transpose(1, 2, 0),
                        mpo.transpose(1, 3, 0, 2),
                        mps_right.transpose(0, 2, 1),
                        mpo_right.transpose(0, 3, 1, 2),
                        Dcut=Dcut, get_loss=False
                    )

                projectors[it][1] = P1
                projectors[nn_it][0] = P2
                
        for it, tensors in enumerate(zip(MPS, MPO)):
            mps, mpo = tensors
            
            new_MPS.append(Contract(
                "aci,abe,cdke,bdj->ijk", projectors[it][0], mps, mpo, projectors[it][1]))
    
    new_norm = MPS_MPS_overlap(new_MPS, new_MPS, conj=True)
    
    # * Random initialization up to norm
    for it, new_tensor in enumerate(new_MPS):
        
        # if new_norm == 0 or norm == 0 or len(new_MPS) == 0:
        #     print(f"{new_tensor=}")
        #     print(f"{new_norm=}")
        #     print(f"{norm=}")
        #     print(f"{MPS=}")
        #     print(f"{len(new_MPS)=}")
            
        #     break
        
        new_MPS[it] = new_tensor / (new_norm/norm)**(1/2/len(new_MPS))
    
    return new_MPS


def single_site(
    MPS: list[npt.NDArray], MPO: list[npt.NDArray], new_MPS: list[npt.NDArray],
    loc: int, dirc: str
) -> list[npt.NDArray]:
    
    """
    Updated the site-canonical MPS and moves the site toward to the input dirc
    """
    
    # * Get only isometric part from new_MPS
    site_canonical = get_only_isometry(new_MPS, loc)
    
    # * Updates the site in new_MPS to site_canonical
    site_canonical[loc] = MPS_MPO_MPS_overlap(
        MPS, MPO, site_canonical
    )
    
    if dirc == "right":
        right = move_site_right(site_canonical, loc)
        # assert check_site_canonical(right, loc+1), f"Not site canonical"
        
        return right

    elif dirc == "left":
        left = move_site_left(site_canonical, loc)
        # assert check_site_canonical(left, loc-1), f"Not site canonical"
        
        return left
    
    else:
        raise ValueError(f"{dirc=} != 'right' or 'left'")


def single_site_via_storing(
    MPS: list[npt.NDArray], MPO: list[npt.NDArray], new_MPS: list[npt.NDArray], loc: int, dirc: str, 
    contract_list_left: list[npt.NDArray], contract_list_right: list[npt.NDArray]
) -> tuple[list[npt.NDArray], list[npt.NDArray], list[npt.NDArray]]:
    
    # * Get only isometric part from new_MPS
    site_canonical = get_only_isometry(new_MPS, loc)
    
    # * Updates the site in new_MPS to site_canonical
    
    # site_canonical[loc] = Contract(
    #     "abi,ace,bdke,cdj->ijk", contract_list_left[loc], MPS[loc], MPO[loc], contract_list_right[len(MPS)-loc-1]
    # )
    site_canonical[loc] = get_update_mps(left=contract_list_left[loc], mps=MPS[loc], mpo=MPO[loc], right=contract_list_right[len(MPS)-loc-1])

    if dirc == "right":
        #* Update the left contraction
        site_canonical = move_site_right(site_canonical, loc)
        # assert check_site_canonical(site_canonical, loc+1), f"Not site canonical"

        contract_list_left[loc+1] = Contract(
            "abc,aix,bjyx,cky->ijk", contract_list_left[loc], MPS[loc], MPO[loc], site_canonical[loc].conj())
        contract_list_left[loc+1]

    elif dirc == "left":
        #* Update the right contraction
        site_canonical = move_site_left(site_canonical, loc)
        # assert check_site_canonical(site_canonical, loc-1), f"Not site canonical"
    
        contract_list_right[len(MPS)-loc] = Contract(
            "iax,jbyx,kcy,abc->ijk", MPS[loc], MPO[loc], site_canonical[loc].conj(), contract_list_right[len(MPS)-loc-1])
    else:
        raise ValueError(f"{dirc=} != 'right' or 'left'")
    
    return site_canonical, contract_list_left, contract_list_right


def get_update_mps(
    left: npt.NDArray, mps: npt.NDArray, mpo: npt.NDArray, right: npt.NDArray
):
    left_first = left.shape[2] * right.shape[0]
    right_first = left.shape[0] * right.shape[2]
    
    if left_first < right_first:
        tensor1 = Contract("abi,ace->bice", left, mps)
        tensor2 = Contract("bice,bdke->icdk", tensor1, mpo)
        new_mps = Contract("icdk,cdj->ijk", tensor2, right)
    else:
        tensor1 = Contract("cdj,ace->djae", right, mps)
        tensor2 = Contract("djae,bdke->jabk", tensor1, mpo)
        new_mps = Contract("jabk,abi->ijk", tensor2, left)
    
    return new_mps


def two_site_via_storing(
    MPS: list[npt.NDArray], MPO: list[npt.NDArray], new_MPS: list[npt.NDArray], Dcut: int, norm_Keep: float, loc: int, dirc: str, 
    contract_list_left: list[npt.NDArray], contract_list_right: list[npt.NDArray], verbose: bool = False,
) -> tuple[list[npt.NDArray], list[npt.NDArray], list[npt.NDArray]]:
    
    if dirc == "right":
                
        # * Get only isometric part from new_MPS
        site_canonical = get_only_isometry(new_MPS, loc, loc+1)
        
        if verbose:
            print(f"right 1st con:", end=" ")
        now = time.perf_counter()
        
        # * Updates the site in new_MPS to site_canonical
        update = Contract(
            "abi,acg,bdkg,ceh,dflh,efj->ijkl", contract_list_left[loc], MPS[loc], MPO[loc], MPS[loc+1], MPO[loc+1], contract_list_right[len(MPS)-loc-2]
        )
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s,", end=" ")
        # time1 = time.perf_counter()-now
        # now = time.perf_counter()
        
        # left1 = Contract("abi,acg->cgbi", contract_list_left[loc], MPS[loc])
        # left = Contract("cgbi,bdkg->cdik", left1, MPO[loc])
        # right1 = Contract("efj,ceh->chfj", contract_list_right[len(MPS)-loc-2], MPS[loc+1])
        # right = Contract("chfj,dflh->cdjl", right1, MPO[loc+1])
        # update2 = Contract("cdik,cdjl->ijkl", left, right)
        
        # time2 = time.perf_counter()-now
        
        # assert np.allclose(update, update2), f"Contraction error"
        
        # print(f"time1={round_sig(time1)} time2={round_sig(time2)}")
        
        if verbose:
            print(f"right SVD:", end=" ")
        now = time.perf_counter()
        
        matrix = update.transpose(0, 2, 1, 3).reshape(update.shape[0]*update.shape[2], update.shape[1]*update.shape[3])
        U, S, Vh = SVD(matrix, Skeep=1.e-8, Nkeep=Dcut, norm_Keep=norm_Keep,)
        # U, S, Vh = SVD(matrix)
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s,", end=" ")
        
        site_canonical[loc] = U.reshape(update.shape[0], update.shape[2], -1).transpose(0,2,1)
        site_canonical[loc+1] = np.diag(S) @ Vh
        site_canonical[loc+1] = site_canonical[loc+1].reshape(-1, update.shape[1], update.shape[3])

        #* Update the left contraction
        # right = move_site_right(site_canonical, loc)
        # assert check_site_canonical(site_canonical, loc+1), f"Not site canonical"

        if verbose:
            print(f"right 2nd con:", end=" ")
        now = time.perf_counter()
        
        contract_list_left[loc+1] = Contract(
        "abc,aix,bjyx,cky->ijk", contract_list_left[loc], MPS[loc], MPO[loc], site_canonical[loc].conj())
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s")
        
        return site_canonical, contract_list_left, contract_list_right

    elif dirc == "left":
        # * Get only isometric part from new_MPS
        site_canonical = get_only_isometry(new_MPS, loc, loc-1)
        
        if verbose:
            print(f"left 1st con:", end=" ")
            now = time.perf_counter()
        
        # * Updates the site in new_MPS to site_canonical
        update = Contract(
            "abi,acg,bdkg,ceh,dflh,efj->ijkl", contract_list_left[loc-1], MPS[loc-1], MPO[loc-1], MPS[loc], MPO[loc], contract_list_right[len(MPS)-loc-1]
        )
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s,", end=" ")
            print(f"left SVD:", end=" ")
            now = time.perf_counter()
        
        matrix = update.transpose(0, 2, 1, 3).reshape(update.shape[0]*update.shape[2], update.shape[1]*update.shape[3])
        U, S, Vh = SVD(matrix, Skeep=1.e-8, Nkeep=Dcut, norm_Keep=norm_Keep,)
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s,", end=" ")
        
        site_canonical[loc-1] = U @ np.diag(S)
        site_canonical[loc] = Vh.reshape(-1, update.shape[1], update.shape[3])
        site_canonical[loc-1] = site_canonical[loc-1].reshape(update.shape[0], update.shape[2], -1).transpose(0,2,1)

        # * Update the left contraction
        # right = move_site_right(site_canonical, loc)
        # assert check_site_canonical(site_canonical, loc-1), f"Not site canonical"

        if verbose:
            print(f"left 2nd con:", end=" ")
            now = time.perf_counter()

        contract_list_right[len(MPS)-loc] = Contract(
            "iax,jbyx,kcy,abc->ijk", MPS[loc], MPO[loc], site_canonical[loc].conj(), contract_list_right[len(MPS)-loc-1])
        
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s")
        
        return site_canonical, contract_list_left, contract_list_right
    
    else:
        raise ValueError(f"{dirc=} != 'right' or 'left'")


def CBE_DMRG(
    MPS: list[npt.NDArray],
    MPO: list[npt.NDArray],
    new_MPS: list[npt.NDArray],
    Dcut: int,
    norm_Keep: float,
    loc: int,
    dirc: str, 
    contract_list_left: list[npt.NDArray],
    contract_list_right: list[npt.NDArray],
    delta: float = 0.1, 
    verbose: bool = False,
):

    if dirc == "right":
        loc_right = loc + 1
                
        # * Get only isometric part from new_MPS
        site_canonical = get_only_isometry(new_MPS, loc, loc+1)
        
        if verbose:
            print(f"left 1st con:", end=" ")
            now = time.perf_counter()
        
        original_left_isometry, original_right_isometry, Sigma = site_to_bond_canonical_MPS(
            orthogonality_center = new_MPS[loc],
            isometry = new_MPS[loc_right],
            dirc = dirc
        )
        
        left_D, _, left_d = original_left_isometry.shape
        _, right_D, right_d = original_right_isometry.shape
        
        left_identity = np.identity(left_D * left_d).reshape(left_D, left_d, left_D, left_d)
        right_identity = np.identity(right_D * right_d).reshape(right_D, right_d, right_D, right_d)
        
        left_discarded = left_identity - Contract(
            "iaj,kal->ijkl", original_left_isometry.conj(), original_left_isometry
        )
        
        left_discarded_contracted = Contract(
            "aid,bjed,celk,abc->ijkl",
            MPS[loc], MPO[loc], left_discarded, contract_list_left[loc]
        )
        
        right_discarded = right_identity - Contract(
            "aij,akl->ijkl", original_right_isometry.conj(), original_right_isometry
        )
        
        right_discarded_contracted = Contract(
            "iad,jbed,celk,abc->ijkl",
            MPS[loc_right], MPO[loc_right], right_discarded, contract_list_right[len(MPS)-1-loc_right]
        )
        
        temp = left_discarded_contracted.reshape(left_discarded_contracted.shape[0], -1)
        
        # _, S1, Vh1 = SVD(temp)
        R1, _ = RQ(temp)
        
        # right_discarded_contracted_plus_info = Contract(
        #     "ix,xy,yjkl->ljlk",
        #     np.diag(S1), Vh1, right_discarded_contracted
        # )
        
        right_discarded_contracted_plus_info = Contract(
            "yi,yjkl->ijlk",
            R1, right_discarded_contracted
        )
        
        shape0, shape1, shape2, shape3 = right_discarded_contracted_plus_info.shape
        
        assert shape2 == right_D, f"{shape2=} != {right_D=}"
        assert shape3 == right_d, f"{shape3=} != {right_d=}"
        
        right_discarded_contracted_plus_info = right_discarded_contracted_plus_info.reshape(shape0, shape1*shape2*shape3)
        
        new_shape = math.ceil(Dcut/shape1)
        
        _, S2, Vh2 = SVD(right_discarded_contracted_plus_info, Nkeep=new_shape)
        
        new_shape, = S2.shape
        
        right_info = np.diag(S2) @ Vh2
        right_info = right_info.reshape(new_shape, shape1, shape2, shape3)
        right_info = right_info.reshape(new_shape * shape1, shape2 * shape3)
        
        # _, _, Vh3 = SVD(right_info)
        _, Q3 = RQ(right_info)
        
        # preselection = Vh3.reshape(new_shape * shape1, right_D, right_d)
        preselection = Q3.reshape(-1, right_D, right_d)
        
        left_projection_contracted = Contract(
            "abji,acf,bdgf,keg,cde->ijk", left_discarded_contracted, MPS[loc_right], MPO[loc_right], preselection.conj(), contract_list_right[len(MPS)-1-loc_right]
        )
        left_projection_contracted = left_projection_contracted.reshape(-1, left_projection_contracted.shape[2])
        
        _, _, Vh4 = SVD(left_projection_contracted, Nkeep=int(Dcut*delta))
        
        right_projection = Contract("ia,ajk->ijk", Vh4, preselection)
        
        CBE_right_bond = np.concatenate((original_right_isometry, right_projection), axis=0)

        contract_list_right[len(MPS)-1-loc] = Contract(
            "iax,jbyx,kcy,abc->ijk", MPS[loc_right], MPO[loc_right], CBE_right_bond.conj(), contract_list_right[len(MPS)-1-loc_right]
        )
        
        update = get_update_mps(left=contract_list_left[loc], mps=MPS[loc], mpo=MPO[loc], right=contract_list_right[len(MPS)-1-loc])
        
        site_canonical[loc_right] = CBE_right_bond
        site_canonical[loc] = update
        
        site_canonical = move_site_right(site_canonical, loc)
        assert check_site_canonical(site_canonical, loc_right) < 1.e-8, f"Not site canonical, {loc=} {check_site_canonical(site_canonical, loc_right)=}"
    
        if verbose:
            print(f"left 2nd con:", end=" ")
            now = time.perf_counter()
            
        contract_list_left[loc_right] = Contract(
            "abc,aix,bjyx,cky->ijk", contract_list_left[loc], MPS[loc], MPO[loc], site_canonical[loc].conj()
        )
                
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s")
        
        return site_canonical, contract_list_left, contract_list_right

    elif dirc == "left":
        
        loc_left = loc - 1
        
        # * Get only isometric part from new_MPS
        site_canonical = get_only_isometry(new_MPS, loc, loc-1)
        
        if verbose:
            print(f"left 1st con:", end=" ")
            now = time.perf_counter()
        
        original_left_isometry, original_right_isometry, Sigma = site_to_bond_canonical_MPS(
            orthogonality_center = new_MPS[loc],
            isometry = new_MPS[loc_left],
            dirc = dirc
        )
        
        left_D, _, left_d = original_left_isometry.shape
        _, right_D, right_d = original_right_isometry.shape
        
        left_identity = np.identity(left_D * left_d).reshape(left_D, left_d, left_D, left_d)
        right_identity = np.identity(right_D * right_d).reshape(right_D, right_d, right_D, right_d)
        
        left_discarded = left_identity - Contract(
            "iaj,kal->ijkl", original_left_isometry.conj(), original_left_isometry
        )
        
        left_discarded_contracted = Contract(
            "aid,bjed,celk,abc->ijkl",
            MPS[loc_left], MPO[loc_left], left_discarded, contract_list_left[loc_left]
        )
        
        right_discarded = right_identity - Contract(
            "aij,akl->ijkl", original_right_isometry.conj(), original_right_isometry
        )
        
        right_discarded_contracted = Contract(
            "iad,jbed,celk,abc->ijkl",
            MPS[loc], MPO[loc], right_discarded, contract_list_right[len(MPS)-1-loc]
        )
        
        temp = right_discarded_contracted.reshape(right_discarded_contracted.shape[0], -1)
        
        # U1, S1, _ = SVD(temp)
        # left_discarded_contracted_plus_info = Contract(
        #     "xjkl,xy,yi->lkji",
        #     left_discarded_contracted, U1, np.diag(S1)
        # )
        
        R1, _ = RQ(temp)
        left_discarded_contracted_plus_info = Contract(
            "xjkl,xi->lkji",
            left_discarded_contracted, R1
        )
        
        shape0, shape1, shape2, shape3 = left_discarded_contracted_plus_info.shape
        
        assert shape0 == left_D, f"{shape0=} != {left_D=}"
        assert shape1 == left_d, f"{shape1=} != {left_d=}"
        
        left_discarded_contracted_plus_info = left_discarded_contracted_plus_info.reshape(shape0*shape1*shape2, shape3)
        
        new_shape = math.ceil(Dcut/shape2)
        
        U2, S2, _ = SVD(left_discarded_contracted_plus_info, Nkeep=new_shape)
        
        new_shape, = S2.shape
        
        left_info = U2 @ np.diag(S2)
        left_info = left_info.reshape(shape0, shape1, shape2, new_shape)
        left_info = left_info.reshape(shape0 * shape1, shape2 * new_shape)
        
        # U3, _, _ = SVD(left_info)
        # preselection = U3.reshape(left_D, left_d, shape2 * new_shape)
        
        Q3, _ = QR(left_info)
        preselection = Q3.reshape(left_D, left_d, -1).transpose(0, 2, 1)
        
        right_projection_contracted = Contract(
            "bci,bxd,cyjd,iaj,xyzw->azw", contract_list_left[loc_left], MPS[loc_left], MPO[loc_left], preselection.conj(), right_discarded_contracted
        )
        right_projection_contracted = right_projection_contracted.reshape(right_projection_contracted.shape[0], -1)
        
        U4, _, _ = SVD(right_projection_contracted, Nkeep=int(Dcut*delta))
        
        left_projection = Contract("iak,aj->ijk", preselection, U4)
        
        CBE_left_bond = np.concatenate((original_left_isometry, left_projection), axis=1)

        contract_list_left[loc] = Contract(
            "abc,aix,bjyx,cky->ijk", contract_list_left[loc_left], MPS[loc_left], MPO[loc], CBE_left_bond.conj()
        )
        
        update = get_update_mps(left=contract_list_left[loc], mps=MPS[loc], mpo=MPO[loc], right=contract_list_right[len(MPS)-1-loc])
        
        site_canonical[loc_left] = CBE_left_bond
        site_canonical[loc] = update
        
        site_canonical = move_site_left(site_canonical, loc)
        assert check_site_canonical(site_canonical, loc_left) < 1.e-8, f"Not site canonical, {loc=} {check_site_canonical(site_canonical, loc_left)=}"
    
        if verbose:
            print(f"left 2nd con:", end=" ")
            now = time.perf_counter()
            
        contract_list_right[len(MPS)-1-loc_left] = Contract(
            "iax,jbyx,kcy,abc->ijk", MPS[loc], MPO[loc], site_canonical[loc].conj(), contract_list_right[len(MPS)-1-loc]
        )
                
        if verbose:
            print(f"{round_sig(time.perf_counter()-now)}s")
        
        return site_canonical, contract_list_left, contract_list_right
    
    else:
        raise ValueError(f"{dirc=} != 'right' or 'left'")

