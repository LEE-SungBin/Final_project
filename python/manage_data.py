from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
import platform

from python.utils import round_sig, geo_mean, geo_std, print_traceback, get_shape, expected_fid


def ensure_list(x):
    """Convert to list if not already a list or np.ndarray."""
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    return [x]


def load_data(
    Ms, Ns,
    Dcut1s,
    two_qubit_gate_modes,
    only_four_layer: bool = True,
    mode: str = "data",
    sort_values = ["M", "N", "K", "D1", "seed"],
    load_all: bool = False,
    **kwargs
) -> list[list[pd.DataFrame]] | list[pd.DataFrame]:
    """Load DataFrames based on the combinations of inputs."""
    # Ensure inputs are treated as lists
    two_qubit_gate_modes = ensure_list(two_qubit_gate_modes)
    Ms = ensure_list(Ms)
    Ns = ensure_list(Ns)
    Dcut1s = ensure_list(Dcut1s)
        
    # Initialize list to store DataFrames
    list_of_list_of_df = [[] for _ in two_qubit_gate_modes]
    
    for it, two_qubit_gate_mode in enumerate(two_qubit_gate_modes):
        for M, N in zip(Ms, Ns):
            for Dcut1 in Dcut1s:
                try:
                    if platform.node() == 'Laptop-of-Physics' or platform.node() == "Desktop-of-Physics":
                        location = f"/mnt/d/OneDrive/Research/TensorNetwork/Seung-SupLEE/save/new_result/Total={M*N}, {M}x{N}/{mode}/TQG={two_qubit_gate_mode}/D1={Dcut1}"
                    else:
                        location = f"/home/sungbinlee/save/new_result/Total={M*N}, {M}x{N}/{mode}/TQG={two_qubit_gate_mode}/D1={Dcut1}"
                    
                    if load_all:
                        df = load_result(
                            location,
                            sort_values = sort_values,
                        )

                    else:
                        conditions = get_conditions(
                            M=M, N=N, **kwargs,
                            Dcut1=Dcut1,
                            two_qubit_gate_mode=two_qubit_gate_mode,
                            only_four_layer=only_four_layer,
                        )
                        df = load_result(
                            location,
                            conditions,
                            sort_values = sort_values,
                        )
                    
                    if isinstance(df, pd.DataFrame):
                        print(f"{M=} {N=} {two_qubit_gate_mode=} {Dcut1=} loaded")
                        list_of_list_of_df[it].append(df)
                    else:
                        print(f"{M=} {N=} {two_qubit_gate_mode=} {Dcut1=} not exists, {df=}")

                except Exception as e:
                    print(e)
                    continue
    
    # If there is only one two_qubit_gate_mode, return a flat list of DataFrames
    if len(two_qubit_gate_modes) == 1:
        return list_of_list_of_df[0]
    
    # Otherwise, return the list of lists of DataFrames
    return list_of_list_of_df


def load_result(
    location: str = ".",
    conditions: list[str] | None = None,
    sort_values: list[str] = ["M", "N", "K", "D1"],
    verbose: bool = False,
) -> pd.DataFrame:

    # * Scan the result directory and gather result files
    result_dir = Path(f"{location}")
    # result_keys = get_setting(location=location, conditions=conditions)
    # result_files = [result_dir /
    #                 f"{result_key}.pkl" for result_key in result_keys]
    result_files = [f for f in result_dir.iterdir() if filter_file(f, ".pkl")]

    # * Read files
    results: list[dict[str, Any]] = []
    for it, file in enumerate(result_files):
        if verbose:
            print(f"{it} ", end="")
        
        if file.is_file():
            with open(file, "rb") as f:
                results.append(pickle.load(f))

    # * Concatenate to single dataframe

    # print(f"{results=}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n")

    if conditions == None:
        if sort_values is None:
            return df
        else:
            return df.sort_values(by=sort_values, ascending=True)

    else:
        if verbose:
            print(f"Query in progress")
        for condition in conditions:
            df = df.query(condition)
    
        if len(df) > 0:
            if sort_values is None:
                return df
            else:
                return df.sort_values(by=sort_values, ascending=True)
        else:
            return None


def get_conditions(
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
    lower_depth: bool = False,
    rand: bool | None = None,
    Dcut1: int | None = None,
    Dcut2: int | None = None,
    use_GILT: bool | None = None,
    epsilon: float | None = None,
    gauging_tol: float | None = None,
    get_exact_state: bool | None = None,
    get_exact_fid: bool | None = None,
    get_patch_state: bool | None = None,
    Layer_wise_bitstring_contraction: bool | None = None,
    single_qubit_gate_mode: str | None = None,
    two_qubit_gate_mode: str | None = None,
    only_four_layer: bool | None = None,
    theta: float | None = None,
    phi: float | None = None,
    correlated: bool | None = None,
) -> list[str]:
    
    conditions: list[str] = []
    
    if M is not None:
        conditions.append(f"M=={M}")
    if N is not None:
        conditions.append(f"N=={N}")
    if K is not None:
        if lower_depth:
            conditions.append(f"K<={K}")
        else:
            conditions.append(f"K=={K}")
    if rand is not None:
        conditions.append(f"rand=={rand}")
    if Dcut1 is not None:
        conditions.append(f"D1=={Dcut1}")
    if Dcut2 is not None:
        conditions.append(f"D2=={Dcut2}")
    if use_GILT is not None:
        conditions.append(f"GILT=={use_GILT}")
    if epsilon is not None:
        conditions.append(f"e=={epsilon}")
    if get_exact_state is not None:
        conditions.append(f"exact=={get_exact_state}")
    if get_exact_fid is not None:
        conditions.append(f"exact_fid=={get_exact_fid}")
    if Layer_wise_bitstring_contraction is not None:
        conditions.append(f"contract=={Layer_wise_bitstring_contraction}")
    if gauging_tol is not None:
        conditions.append(f"tol=={gauging_tol}")
    if get_patch_state is not None:
        conditions.append(f"patch=={get_patch_state}")
    if single_qubit_gate_mode is not None:
        conditions.append(f"SQG=='{single_qubit_gate_mode}'")
    if two_qubit_gate_mode is not None:
        conditions.append(f"TQG=='{two_qubit_gate_mode}'")
    if theta is not None:
        conditions.append(f"t=={theta}")
    if phi is not None:
        conditions.append(f"p=={phi}")
    if correlated is not None:
        conditions.append(f"corr=={correlated}")
    if only_four_layer is not None:
        conditions.append(f"only_four_layer=={only_four_layer}")
    
    return conditions


def filter_file(f: Path, suffix: str) -> bool:
    return f.is_file() and (f.suffix == suffix) and f.stat().st_size > 0


def prepare_plots():
    
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    
    nrows, ncols = 3, 3
    axs = [0 for _ in range(nrows*ncols)]

    axs[0] = fig.add_subplot(nrows, ncols, 1)  # Spans across the first row
    # axs[0].set_title('Big Plot on Top')
    axs[0].set_xlabel(f"Layer K")
    axs[0].set_ylabel(rf"Fidelity $F$")

    axs[1] = fig.add_subplot(nrows, ncols, 2)  # Spans across the first row
    # axs[1].set_title('Big Plot on Top')
    axs[1].set_xlabel(f"Layer K")
    axs[1].set_ylabel(rf"$<\chi>$")

    axs[2] = fig.add_subplot(nrows, ncols, 3)  # Bottom left
    # axs[2].set_title('Small Plot 1')
    axs[2].set_xlabel(f"Layer K")
    axs[2].set_ylabel(f"Total time [s]")

    axs[3] = fig.add_subplot(nrows, ncols, 4)  # Bottom left
    # axs[3].set_title('Small Plot 2')
    axs[3].set_xlabel(f"Layer K")
    axs[3].set_ylabel(f"SU time [s]")

    axs[4] = fig.add_subplot(nrows, ncols, 5)  # Bottom right
    # axs[4].set_title('Small Plot 3')
    axs[4].set_xlabel(f"Layer K")
    axs[4].set_ylabel(f"GILT time [s]")

    axs[5] = fig.add_subplot(nrows, ncols, 6)  # Bottom left
    # axs[5].set_title('Small Plot 4')
    axs[5].set_xlabel(f"Layer K")
    axs[5].set_ylabel(f"SU Gauging time [s]")

    axs[6] = fig.add_subplot(nrows, ncols, 7)  # Bottom right
    # axs[6].set_title('Small Plot 5')
    axs[6].set_xlabel(f"Layer K")
    axs[6].set_ylabel(f"Renorm time [s]")
    
    axs[7] = fig.add_subplot(nrows, ncols, 8)  # Bottom right
    # axs[9].set_title('Small Plot 6')
    axs[7].set_xlabel(f"Layer K")
    axs[7].set_ylabel(f"Overlap time [s]")

    axs[8] = fig.add_subplot(nrows, ncols, 9)  # Bottom right
    # axs[8].set_title('Small Plot 6')
    axs[8].set_xlabel(f"Layer K")
    axs[8].set_ylabel(f"Sampling time [s]")
    
    return fig, axs


def finish_plots(fig, axs, K):
    
    axs[0].set_yscale("log")
    axs[0].axhline(1.0, linestyle="--", color="black")

    # axs[1].axhline(Dcut1, linestyle="--", color="black", label=rf"$\chi_1$={Dcut1}")
    # axs[1].axhline(Dcut1/2, linestyle="-.", color="black", label=rf"$\chi_1/2$={int(Dcut1/2)}")

    # axs[0].set_ylim([None, 2])
    axs[0].set_ylim([1.e-15, 2])
    axs[1].set_ylim([0.8, None])
    axs[1].set_yscale("log")
    
    x = np.linspace(1, K, K)
    axs[1].plot(x, 2**(x/8), linestyle="--", color="black")
    
    axs[2].set_ylim([0, None])
    axs[3].set_ylim([0, None])
    axs[4].set_ylim([1.e-2, None])
    axs[4].set_yscale("log")
    axs[5].set_ylim([0, None])
    axs[6].set_ylim([0, None])
    axs[7].set_ylim([0, None])
    # axs[7].set_yscale("log")
    axs[8].set_ylim([0, None])
    # axs[8].set_yscale("log")

    for i, ax in enumerate(axs):
        # if i == 1:
        # ax.axvline(int(8*np.log2(Dcut1)), linestyle="-", color="black", label=rf"8x$log_2$ ($\chi_1$)={int(8*np.log2(Dcut1))}")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlim([0, K+1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.legend(fontsize=7)

    return fig, axs


def single_plot(
    location: str,
    fig,
    axs,
    color,
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
    rand: bool | None = None,
    Dcut1: int | None = None,
    Dcut2: int | None = None,
    use_GILT: bool | None = None,
    epsilon: float | None = None,
    batch_size: int | None = None,
    get_exact_fid: bool | None = None,
    sampling: bool | None = None,
    n_bitstrings: int | None = None,
    error: float = 1.e-2
):
    
    conditions = get_conditions(
        M=M, N=N, K=K, rand=rand, Dcut1=Dcut1, Dcut2=Dcut2,
        use_GILT=use_GILT, epsilon=epsilon, batch_size=batch_size,
        get_exact_fid=get_exact_fid,
        sampling=sampling, n_bitstrings=n_bitstrings
    )
    
    # print(f"{location=}")
    # print(f"{conditions=}")
    
    df = load_result(location, conditions)
    # print(f"{df=}")
    
    if df.empty:
        print("df empty")
        print(f"{conditions=}")
        return fig, axs

    Ks1 = np.arange(K+1)
    Ks2 = np.arange(1, K+1)
    
    product_fid = [[] for _ in range(K+1)]
    exact_fid = [[] for _ in range(K+1)]
    avg_leg_dim = [[] for _ in range(K+1)]
    full_time = [[] for _ in range(K)]
    SU_time = [[] for _ in range(K)]
    GILT_time = [[] for _ in range(K)]
    SUgauging_time = [[] for _ in range(K)]
    renorm_time = [[] for _ in range(K)]
    overlap_time = [[] for _ in range(K)]
    sampling_time = [[] for _ in range(K)]
    
    # samp_Ks = Ks1[Ks1%8 == 0]
    # PorterThomas_time = [[] for _ in range(int(K/8)+1)]
    # sampling_time = [[] for _ in range(int(K/8)+1)]
    # samp_Ks = np.arange(1, K+1)
    # PorterThomas_time  = [[] for _ in range(K)]
    # sampling_time = [[] for _ in range(K)]
    
    raw_product_fids = df['product_fids'].to_numpy()
    raw_exact_fids = df['exact_fids'].to_numpy()
    raw_avg_leg_dims = df['avg_leg_dims'].to_numpy()
    raw_full_times = df['full_times'].to_numpy()
    raw_SU_time = df['SU_time'].to_numpy()
    raw_GILT_time = df['GILT_time'].to_numpy()
    raw_SUgauging_times = df['SUgauging_time'].to_numpy()
    raw_renorm_times = df['renorm_times'].to_numpy()
    raw_overlap_times = df['overlap_times'].to_numpy()
    raw_sampling_times = df['sampling_times'].to_numpy()
    
    # print(f"{raw_PorterThomas_times}")
    
    product_fid = stack_full_list(raw_product_fids, product_fid)
    exact_fid = stack_full_list(raw_exact_fids, exact_fid)
    avg_leg_dim = stack_full_list(raw_avg_leg_dims, avg_leg_dim)
    full_time = stack_full_list(raw_full_times, full_time)
    SU_time = stack_full_list(raw_SU_time, SU_time)
    GILT_time = stack_full_list(raw_GILT_time, GILT_time)
    SUgauging_time = stack_full_list(raw_SUgauging_times, SUgauging_time)
    renorm_time = stack_full_list(raw_renorm_times, renorm_time)
    overlap_time = stack_full_list(raw_overlap_times, overlap_time)
    sampling_time = stack_full_list(raw_sampling_times, sampling_time)

    if Dcut1 == 0:
        if use_GILT:
            single_errorbar(axs[0], Ks1, product_fid, "--", "o", color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, Overlap")
            # axs[0].errorbar(Ks1, geo_mean(product_fid), linestyle="--", marker="o", capsize=5, color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, Overlap")
            axs[0].fill_between(
            Ks1, geo_mean(product_fid)-np.std(product_fid, axis=1), geo_mean(product_fid)+np.std(product_fid, axis=1),
            color=color, alpha=0.2)
            if get_exact_fid:
                axs[0].errorbar(Ks1, geo_mean(exact_fid), linestyle="solid", marker="^", capsize=5, color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, Exact")
        else:
            axs[0].errorbar(Ks1, geo_mean(product_fid), linestyle="--", marker="o", capsize=5, color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, No GILT, Overlap")
            if get_exact_fid:
                axs[0].errorbar(Ks1, geo_mean(exact_fid), linestyle="solid", marker="^", capsize=5, color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, No GILT, Exact")
        axs[0].plot(Ks1, expected_fid(M, N, Ks1, error), linestyle="--", color=color)
        # axs[0].errorbar(Ks, geo_mean(HOTRG_fid), linestyle="--", marker="x", capsize=5, color=color, label=f"{M}x{N}, Projectors")
        # if use_GILT:
        #     axs[0].errorbar(Ks, geo_mean(filtering_fid), linestyle="--", marker=">", color=color, label=f"{M}x{N}, GILT")
        if use_GILT:
            axs[1].errorbar(Ks1, geo_mean(avg_leg_dim), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\varepsilon$={epsilon}")
        else:
            axs[1].errorbar(Ks1, geo_mean(avg_leg_dim), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, No GILT")
        axs[2].errorbar(Ks2, geo_mean(full_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, N="+r"$2^{" + f"{round_sig(np.log2(n_bitstrings))}" + r"}$")
        try:
            axs[3].errorbar(Ks2, geo_mean(SU_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\varepsilon$={epsilon}")
        except Exception as e:
            print(f"{Ks2=}\n{SU_time=}")
            print_traceback(e)
        if use_GILT:
            axs[4].errorbar(Ks2, geo_mean(GILT_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\varepsilon$={epsilon}")
        else:
            axs[4].errorbar(Ks2, geo_mean(GILT_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, No GILT")

    else:
        axs[0].errorbar(Ks1, geo_mean(product_fid), linestyle="--", marker="o", capsize=5, color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, Overlap")
        if get_exact_fid:
            axs[0].errorbar(Ks1, geo_mean(exact_fid), linestyle="solid", marker="^", capsize=5, color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, Exact")
        axs[0].plot(Ks1, expected_fid(M, N, Ks1, error), linestyle="--", color=color, label=f"{M}x{N}, {error=}, Expected")
        # axs[0].errorbar(Ks, geo_mean(HOTRG_fid), linestyle="--", marker="x", capsize=5, color=color, label=f"{M}x{N}, Projectors")
        # if use_GILT:
        #     axs[0].errorbar(Ks, geo_mean(filtering_fid), linestyle="--", marker=">", color=color, label=f"{M}x{N}, GILT")
        if use_GILT:
            axs[1].errorbar(Ks1, geo_mean(avg_leg_dim), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, $\varepsilon$={epsilon}")
        else:
            axs[1].errorbar(Ks1, geo_mean(avg_leg_dim), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, No GILT")
        axs[2].errorbar(Ks2, geo_mean(full_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, $\chi_2$={Dcut2}, $\varepsilon$={epsilon}, N="+r"$2^{" + f"{round_sig(np.log2(n_bitstrings))}" + r"}$")
        try:
            axs[3].errorbar(Ks2, geo_mean(SU_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}")
        except Exception as e:
            print(f"{Ks2=}\n{SU_time=}")
            print_traceback(e)
        if use_GILT:
            axs[4].errorbar(Ks2, geo_mean(GILT_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, $\varepsilon$={epsilon}")
        else:
            axs[4].errorbar(Ks2, geo_mean(GILT_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_1$={Dcut1}, No GILT")
        
    axs[5].errorbar(Ks2, geo_mean(SUgauging_time), linestyle="--", marker="o", color=color, label=f"{M}x{N}")
    axs[6].errorbar(Ks2, geo_mean(renorm_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_2$={Dcut2}")
    axs[7].errorbar(Ks2, geo_mean(overlap_time), linestyle="--", marker="o", color=color, label=rf"{M}x{N}, $\chi_2={Dcut2}$")
    if sampling is not False:
        # axs[8].errorbar(samp_Ks, geo_mean(sampling_time), linestyle="--", marker="o", color=color, label=f"{M}x{N}, N="+r"$2^{" + f"{round_sig(np.log2(n_bitstrings))}" + r"}$ " +f", batch={batch_size}")
        axs[8].errorbar(Ks2, geo_mean(sampling_time), linestyle="--", marker="o", color=color, label=f"{M}x{N}, N="+r"$2^{" + f"{round_sig(np.log2(n_bitstrings))}" + r"}$")
        # axs[8].set_yscale("log")
    
    # fig, axs = finish_plots(fig, axs, K=40)
    
    return fig, axs


def single_errorbar(
    ax, Xs, data, linestyle: str = "solid", marker: str = "o", color: str = "blue",
    label: str | None = None, capsize: int = 5,
) -> None:
    
    ax.errorbar(Xs, geo_mean(data), linestyle=linestyle, marker=marker, color=color, label=label, capsize=capsize)


def stack_full_list(
    raw_datas: npt.NDArray, stack_list: list[list]
) -> npt.NDArray:
    
    for raw_data in raw_datas:
        for i, data in enumerate(raw_data):
            if i < len(stack_list):
                try:
                    stack_list[i].append(np.abs(data))
                except Exception as e:
                    print(f"stack error {i=}, stack_list[{i}]={stack_list[i]}\n{raw_data=}\n{raw_datas=}")
                    print_traceback(e)

    try:
        arr = np.array(stack_list)
        return arr

    except Exception as e:
        print(f"array error, {get_shape(stack_list)=}\n{stack_list=}")
        print_traceback(e)
