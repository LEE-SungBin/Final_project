import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.stats import gmean
import sys
from pathlib import Path
# * Self only available after python 3.11
# from typing import Self
from typing import Any
from dataclasses import dataclass, field, asdict
import hashlib
import pickle
import pandas as pd
import argparse
import time
import itertools
from datetime import datetime
import opt_einsum as oe
import matplotlib.pyplot as plt
from copy import deepcopy

from python.Contract import *
from python.Decomposition import *


def get_projector(
    Au, Ad, Aunn, Adnn,
    Dcut: int = 0, 
    Skeep: float = 1e-2,
    mode: str = "truncate",
    get_loss: bool = True,
) -> tuple[npt.NDArray, npt.NDArray, float] | tuple[npt.NDArray, npt.NDArray, float, float, npt.NDArray, npt.NDArray]:

    """
    Au-- 0 --Aunn
    |         |
    1         1
    |         |
    Ad-- 0 --Adnn
    """

    assert mode == "truncate" or "full", f"{mode=} != 'truncate' or 'full'"

    Au = Au.reshape(Au.shape[0], Au.shape[1], -1)
    Ad = Ad.reshape(Ad.shape[0], Ad.shape[1], -1)
    Aunn = Aunn.reshape(Aunn.shape[0], Aunn.shape[1], -1)
    Adnn = Adnn.reshape(Adnn.shape[0], Adnn.shape[1], -1)

    assert Au.shape[0] == Aunn.shape[0], f"{Au.shape[0]=} != {Aunn.shape[0]=}"
    assert Ad.shape[0] == Adnn.shape[0], f"{Ad.shape[0]=} != {Adnn.shape[0]=}"
    assert Au.shape[1] == Ad.shape[1], f"{Au.shape[1]=} != {Ad.shape[1]=}"
    assert Aunn.shape[1] == Adnn.shape[1], f"{Aunn.shape[1]=} != {Adnn.shape[1]=}"

    # print("HOTRG started")

    Ausize, Adsize = Au.shape[0], Ad.shape[0]
    
    if Ausize * Adsize > 2**10:
        print(f"{Ausize=} {Adsize=}")

    cutoff = Dcut
    # cutoff = 1
    
    # if Ausize * Adsize <= cutoff or mode == "full" or (Dcut == 0 and Skeep == 0):
    if Ausize * Adsize <= cutoff or mode == "full":
        # print(f"full")
        
        identity = np.identity(Ausize*Adsize)
        # print(f"loss=0.0")

        P1 = identity.reshape(Ausize, Adsize, Ausize*Adsize)
        P2 = identity.reshape(
            Ausize*Adsize, Ausize, Adsize).transpose(1, 2, 0)

        # original = Contract(
        #     "axi,bxj,ayl,byk->ijkl", Au, Ad, Aunn, Adnn
        # )
        # new = Contract(
        #     "axi,bxj,cyl,dyk,abz,cdz->ijkl", Au, Ad, Aunn, Adnn, P1, P2
        # )

        # real_loss = np.linalg.norm(original-new)/np.linalg.norm(original)
        real_loss = 0

        return P1, P2, 1.0, 1.0, np.array([1.0]), np.array([1.0])

    else:
        ABABDagger = Contract(
            "iax,jay,kbx,lby->ijkl", Au, Ad, Au.conj(), Ad.conj()
        ).reshape(Ausize * Adsize, Ausize * Adsize)

        CDCDDagger = Contract(
            "iax,jay,kbx,lby->ijkl", Aunn, Adnn, Aunn.conj(), Adnn.conj()
        ).reshape(Ausize * Adsize, Ausize * Adsize)

        # eigval1, eigvec1 = EIGH(ABABDagger, Nkeep=int(np.sqrt(Ausize*Adsize)))
        eigval1, eigvec1 = EIGH(ABABDagger)
        R1 = eigvec1 @ np.diag(np.sqrt(eigval1))

        # eigval2, eigvec2 = EIGH(CDCDDagger, Nkeep=int(np.sqrt(Ausize*Adsize)))
        eigval2, eigvec2 = EIGH(CDCDDagger)
        R2 = eigvec2 @ np.diag(np.sqrt(eigval2))

        temp = R1.T @ R2

        before_U, before_sigma, before_Vh = SVD(temp, Skeep=1e-8)
        # before_U = before_U[:, before_sigma > 1.e-15]
        # before_Vh = before_Vh[before_sigma > 1.e-15, :]
        # before_sigma = before_sigma[before_sigma > 1.e-15]
        
        ori_norm = np.linalg.norm(before_sigma)
        # print(f"{before_sigma=}")
        after_sigma_computing = deepcopy(before_sigma)
        after_sigma = deepcopy(before_sigma)
        
        after_U = deepcopy(before_U)
        after_Vh = deepcopy(before_Vh)
        
        if Dcut > 0:
            after_U = deepcopy(before_U[:, :Dcut])
            after_Vh = deepcopy(before_Vh[:Dcut, :])
            after_sigma = deepcopy(before_sigma[:Dcut])
        
        if Skeep > 0.0:
            squares = before_sigma ** 2
            # Cumulative sum from the end, then reverse
            # cumsum_from_end = np.sqrt(np.cumsum(squares[::-1])[::-1])
            cumsum_from_end = np.cumsum(squares[::-1])[::-1]
            
            # print(f"{sigma=}")
            # print(f"{cumsum_from_end=}")
            
            # Find indices where cumsum exceeds threshold
            mask = cumsum_from_end > Skeep
            # mask = cumsum_from_end > Skeep * ori_norm**2
            # If no value exceeds threshold, return empty array
            if not np.any(mask):
                pass
            
            else:
                last_idx = np.where(mask)[0][-1]
                # Return elements up to and including that index
                after_U = deepcopy(before_U[:, :last_idx + 1])
                after_Vh = deepcopy(before_Vh[:last_idx + 1, :])
                after_sigma = deepcopy(before_sigma[:last_idx + 1])
                
                after_sigma_computing = deepcopy(before_sigma)
                after_sigma_computing[last_idx+1:] = 0.0
            
            # U = U[:, sigma > Skeep]
            # Vh = Vh[sigma > Skeep, :]
            # sigma = sigma[sigma > Skeep]
        
        # new_temp = U @ np.diag(sigma) @ Vh
        # print(f"{np.linalg.norm(temp) - np.linalg.norm(new_temp)=}")
        
        new_norm = np.linalg.norm(after_sigma)
        
        # print(f"{ori_norm**2 - new_norm**2=}")
        
        # print(f"{after_sigma=}")

        P1 = R2 @ before_Vh.conj().T @ np.diag(1.0 / before_sigma) @ np.diag(np.sqrt(after_sigma_computing))
        P2 = np.diag(np.sqrt(after_sigma_computing)) @ np.diag(1.0 / before_sigma) @ before_U.conj().T @ R1.T
        
        # P1 = R2 @ after_Vh.conj().T @ np.diag(np.sqrt(1.0 / after_sigma))
        # P2 = np.diag(np.sqrt(1.0 / after_sigma)) @ after_U.conj().T @ R1.T

        normalization = np.sqrt(abs(P2).sum()/abs(P1).sum())

        if normalization > 0:
            P1 *= normalization
            P2 /= normalization

        P1 = P1.reshape(Ausize, Adsize, -1)
        P2 = P2.reshape(-1, Ausize, Adsize).transpose(1, 2, 0)

        # Projectors = Contract(
        #     "ija,kla->ijkl", P1, P2
        # ).reshape(Ausize * Adsize, Ausize * Adsize)
        
        # print(f"{np.linalg.norm(Projectors-np.identity(Ausize * Adsize))=}")
        # print(f"{Projectors=}")

        if get_loss:
            # ABABDagger = ABABDagger.reshape(Ausize, Adsize, Ausize, Adsize)
            # CDCDDagger = CDCDDagger.reshape(Ausize, Adsize, Ausize, Adsize)
            # HOTRG_loss = get_HOTRG_loss(ABABDagger, CDCDDagger, Projectors)
            
            return P1, P2, ori_norm, new_norm, before_sigma, after_sigma
            
        else:
            HOTRG_loss = 0
            return P1, P2, 1.0, 1.0, np.array([1.0]), np.array([1.0])

        # original = Contract(
        #     "axi,bxj,ayl,byk->ijkl", Au, Ad, Aunn, Adnn
        # )
        # new = Contract(
        #     "axi,bxj,cyl,dyk,abz,cdz->ijkl", Au, Ad, Aunn, Adnn, P1, P2
        # )

        # real_loss = np.linalg.norm(original-new)/np.linalg.norm(original)

        # assert -1.e-5 < real_loss - \
        #     HOTRG_loss < 1.e-5, f"{real_loss=} != {HOTRG_loss=}"

        # print(f"{HOTRG_loss=} {loss=}")

        # print("HOTRG finished")


def get_HOTRG_loss(ABABDagger, CDCDDagger, Projectors) -> float:

    # print("get HOTRG loss", end=" ")

    Ausize = ABABDagger.shape[0]
    Adsize = ABABDagger.shape[1]

    EEdagger = Contract("ijab,klcd->ijklabcd", ABABDagger, CDCDDagger)
    EEdagger = EEdagger.reshape(Ausize**2*Adsize**2, Ausize**2*Adsize**2)

    # print(f"{EEdagger.shape=}")

    eigval, eigvec = EIGH(EEdagger)

    S = np.sqrt(eigval)
    U = eigvec.reshape(Ausize, Adsize, Ausize, Adsize, -1)

    t = Contract("ababi->i", U)
    tprime = Contract("abcdi,abcd->i", U, Projectors)

    original = Contract("a,ai->i", t, np.diag(S))
    new = Contract("a,ai->i", tprime, np.diag(S))

    return np.linalg.norm(original-new)/np.linalg.norm(original)
