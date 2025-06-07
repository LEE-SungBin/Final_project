import numpy as np
import numpy.typing as npt
import sys
sys.path.append("../..")

# from python.utils import round_sig, get_topology, get_pattern, print_traceback
from python.plot import log_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter, MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import sys
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import List, Union
import pandas as pd
# from python.manage_data import *
from python.Backend import Backend


def ensure_list(x):
    """Convert to list if not already a list."""
    return x if isinstance(x, list) else [x]


def load_data(
    two_qubit_gate_modes,
    Ms, Ns,
    Dcut1s,
    **kwargs
):
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
                    location = f"/home/sungbinlee/save/result/Total={M*N}, {M}x{N}/data/TQG={two_qubit_gate_mode}/D1={Dcut1}"
                    
                    conditions = get_conditions(
                        M=M, N=N, **kwargs,
                        Dcut1=Dcut1,
                        two_qubit_gate_mode=two_qubit_gate_mode
                    )
                    
                    df = load_result(location, conditions)
                    
                    
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


def convert_array(
    series: pd.Series,
    dtype: type = float,
) -> npt.NDArray:

    return np.array(
        [np.array(sublist, dtype=dtype) for sublist in series.to_numpy()],
        dtype=dtype
    )


def get_log2_fids(
    Lambda_errors: npt.NDArray[np.float64]
) -> npt.NDArray:
    
    log2_fids = []
    
    Lambda_errors = np.array(Lambda_errors)
    
    if len(Lambda_errors.shape) == 4:
        for Lambda_error in Lambda_errors:
            """
            Lambda_errors = [instance, depths, loc, dirc]
            """
            
            single_fids = []
            
            for depth, single_layer in enumerate(Lambda_error):
                layer_errors = Lambda_error[:depth+1]
                log_fidelities = np.log2(1.0-layer_errors)
                layer_fidelity = np.sum(log_fidelities) / 2 # * Double counting
                single_fids.append(layer_fidelity)
            
            log2_fids.append(np.array(single_fids))

    elif len(Lambda_errors.shape) == 3:
        
        single_fids = []
        
        for depth, single_layer in enumerate(Lambda_errors):
            """
            Lambda_errors = [depths, loc, dirc]
            """
            
            layer_errors = Lambda_errors[:depth+1]
            log_fidelities = np.log2(1.0-layer_errors)
            layer_fidelity = np.sum(log_fidelities) / 2 # * Double counting
            single_fids.append(layer_fidelity)
        
        log2_fids.append(np.array(single_fids))
    
    else:
        raise ValueError(f"{Lambda_errors.shape=} not 3 or 4")
    
    return np.array(log2_fids)


def add_colorbar(
    fig,
    ax,
    norm,
    colormap = plt.get_cmap("turbo"),
    cbar_ticks: list[int] = [2, 4, 8, 16, 32, 64, 128],
    cbar_tick_labels = [2, 4, 8, 16, 32, 64, 128],
    tick_size: int = 15,
    colorbar_label: str = r"$\chi$",
    colorbar_size: int = 30,
    colorbar_pad: int = 10,
):

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_tick_labels)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(colorbar_label, fontsize=colorbar_size, labelpad=colorbar_pad)
    cbar.ax.minorticks_off()
    
    return fig, ax


def manual_colorbar(
    fig,
    cmap=plt.get_cmap("turbo"),
    norm=mcolors.Normalize(0, 40),
    position=[1.01, 0.07, 0.02, 0.92],
    cbar_ticks=[0, 8, 16, 24, 32, 40],
    cbar_ticklabels: list | None = None,
    tick_size=20,
    colorbar_label=r"$\mathcal{D}$",
    label_size=30,
    colorbar_pad=10,
    label_position='top'  # Adding an argument to specify the position of the label
):
    
    cax = fig.add_axes(position)  # [left, bottom, width, height]

    # Create the colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is None:
        cbar_ticklabels = cbar_ticks
    cbar.set_ticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=tick_size)  # Adjust tick label size if needed
    cbar.ax.minorticks_off()
    
    # Move the label to the top
    cbar.ax.set_title(
        colorbar_label, fontsize=label_size, pad=colorbar_pad, loc="center"
    )
    
    return fig


def add_ons(
    fig: Figure,
    ax: Axes,
    colormap = None,
    norm = None,
    cbar_ticks: list = [1, 2, 4, 8, 16, 32, 64, 128],
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_font: int = 15,
    xticks: list | None = None,  # Manually set x-ticks
    xticklabels: list | None = None,  # Manually set x-tick labels
    yticks: list | None = None,  # Manually set y-ticks
    yticklabels: list | None = None,  # Manually set y-tick labels
    tick_size: int = 15,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    add_alphabet: str | None = None,  # Option 1: Input alphabet on top left corner
    alpha_size: int = 20,
    alpha_loc: list[float] = [0, 1.02],
    annotate_text: str | None = None,  # Option 2: Text with boundary at specific location
    annotate_loc: tuple[float, float] = (0.5, 0.5),  # Location for annotation text
    annotate_size: int = 15,
    annotate_color: str | None = None,
    annotate_line_color: str | None = None,
    plot_colorbar: bool = True,  # Option 3: Toggle colorbar on or off,
    colorbar_label: str = r"$\chi$",
    colorbar_size: int = 20,
    colorbar_pad: int = 10,
    vlines: list[float] | None = None,  # List of x positions for vertical lines
    vline_colors: list[str] | None = None,  # List of colors for vertical lines
    vline_styles: list[str] | None = None,  # List of linestyles for vertical lines
    vline_widths: list[float] | None = None,  # List of linewidths for vertical lines
    vline_labels: list[str] | None = None,  # List of labels for vertical lines
    hlines: list[float] | None = None,  # List of y positions for horizontal lines
    hline_colors: list[str] | None = None,  # List of colors for horizontal lines
    hline_styles: list[str] | None = None,  # List of linestyles for horizontal lines
    hline_widths: list[float] | None = None,  # List of linewidths for horizontal lines
    hline_labels: list[str] | None = None,  # List of labels for horizontal lines
) -> tuple[Figure, Axes]:
    
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    # Set x-ticks and labels if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    # Set y-ticks and labels if provided
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
        
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            legend = ax.legend(
                title='', edgecolor='black', framealpha=1, loc=legend_loc, fontsize=legend_font
            )
            legend.get_frame().set_linewidth(1)
            
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            transformed_bbox = ax.transAxes.inverted().transform_bbox(bbox)

    # Option 1: Add input alphabet on the top left corner
    if add_alphabet:
        ax.text(
            alpha_loc[0], alpha_loc[1], add_alphabet,
            transform=ax.transAxes, fontsize=alpha_size, fontweight='bold', va='bottom', ha='left'
        )

    # Option 2: Write text on a specific location with boundary
    if annotate_text:
        if annotate_color is None:
            ax.text(
                annotate_loc[0], annotate_loc[1], annotate_text, transform=ax.transAxes,
                fontsize=annotate_size, color='black',
                # bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=0)
                )
        else:
            ax.text(
                annotate_loc[0], annotate_loc[1], annotate_text, transform=ax.transAxes,
                fontsize=annotate_size, color='black', bbox=dict(facecolor=annotate_color, linewidth=0)
                )

    # Option 3: Add vertical lines
    if vlines is not None:
        for i, vline in enumerate(vlines):
            color = vline_colors[i] if vline_colors and i < len(vline_colors) else 'black'
            style = vline_styles[i] if vline_styles and i < len(vline_styles) else '--'
            width = vline_widths[i] if vline_widths and i < len(vline_widths) else 2
            ax.axvline(vline, color=color, linestyle=style, linewidth=width)
            if vline_labels and i < len(vline_labels):
                ax.text(
                    vline, ax.get_ylim()[1], vline_labels[i], color=color, 
                    ha='center', va='bottom', fontsize=10
                )

    # Option 4: Add horizontal lines
    if hlines is not None:
        for i, hline in enumerate(hlines):
            color = hline_colors[i] if hline_colors and i < len(hline_colors) else 'black'
            style = hline_styles[i] if hline_styles and i < len(hline_styles) else '--'
            width = hline_widths[i] if hline_widths and i < len(hline_widths) else 2
            ax.axhline(hline, color=color, linestyle=style, linewidth=width)
            if hline_labels and i < len(hline_labels):
                ax.text(ax.get_xlim()[1], hline, hline_labels[i], color=color, ha='right', va='center', fontsize=10)

    # Option 5: Add a colorbar with a logarithmic scale (toggle on or off)
    if plot_colorbar and colormap is not None:
        fig, ax = add_colorbar(
            fig,
            ax,
            colormap,
            norm,
            cbar_ticks,
            tick_size,
            colorbar_label,
            colorbar_size,
            colorbar_pad,
        )
    
    return fig, ax


def get_error_per_gate(
    M, N, fidelities_per_qubit: npt.NDArray
):
    """
    fidelities_per_qubit: [instances, depth]
    """
    
    errors = []
    stds = []
    prefactor = []
    
    for it in range(len(fidelities_per_qubit[0])):
        point1_dirc_list, point1_point2_list = get_pattern(M, N, it)
        if it == 0:
            prefactor.append(len(point1_dirc_list))
        else:
            prefactor.append(prefactor[-1]+len(point1_dirc_list))
    prefactor = np.array(prefactor)
    
    try:
        # errors = - np.log(fidelities_per_qubit.mean(axis=0)) * M * N / prefactor
        # stds = - fidelities_per_qubit.std(axis=0) * M * N / errors / prefactor
        
        depths_factor = M*N / prefactor
        
        errors = 1 - fidelities_per_qubit.mean(axis=0)**depths_factor
        stds = depths_factor * fidelities_per_qubit.mean(axis=0)**(depths_factor-1) * fidelities_per_qubit.std(axis=0)
        
        stds[errors < 1.e-2] = 0
        
        errors[fidelities_per_qubit.mean(axis=0) < 0.4] = np.nan
        stds[fidelities_per_qubit.mean(axis=0) < 0.4] = np.nan
        
    except Exception as e:
        print(f"{fidelities_per_qubit.shape=}\n{prefactor.shape=}")
    
    return errors, stds


def theoretical_error_per_gate(
    bond_dim, depth: int, a: float = np.log(2)/4, b: float = 2, open: bool = True
):
    if isinstance(bond_dim, int):
        bond_dims = [bond_dim]
    else:
        bond_dims = bond_dim
    
    errors = []
    for D in bond_dims:
        if open:
            error = a * (1 - b * np.log2(D) / depth)
        else:
            error = a * (1 - 2*b * np.log2(D) / depth)
        if error < 0:
            error = 0
        errors.append(error)
    
    if isinstance(bond_dim, int):
        return errors[0]
    else:
        return errors


def theoretical_bond_dim(
    error_per_gate: float, depth: int, a: float = np.log(2)/4, b: float = 2, open: bool = True,
):
    
    Dcut = 1
    if error_per_gate < np.log(2)/4:
        Dcut = 2**(depth / b * (1 - error_per_gate/a))
        
    return Dcut


def calculate_degeneracy(Lambdas, abs_tol=1.e-3, rel_tol=1.e-3):
    if len(Lambdas) <= 1:
        return np.array([1] * len(Lambdas))  # Single element has degeneracy of 1
    
    # Calculate pairwise relative differences
    def combined_metric(u, v):
        abs_diff = np.abs(u - v)
        rel_diff = np.abs(u - v) / np.abs(u)
        
        if rel_tol == 0:
            return abs_diff / abs_tol
        elif abs_tol == 0:
            return rel_diff / rel_tol
        else:
            return min(abs_diff / abs_tol, rel_diff / rel_tol)
    
    pairwise_combined_diff = pdist(Lambdas[:, None], combined_metric)
    
    # Perform hierarchical clustering based on combined differences
    Z = linkage(pairwise_combined_diff, method='single')
    
    # Form clusters based on the combined tolerance measure
    clusters = fcluster(Z, t=1, criterion='distance')  # t=1 because metric already accounts for tolerance
    
    # Calculate degeneracy as the number of elements in each cluster
    degeneracy = np.array([np.sum(clusters == cluster_id) for cluster_id in clusters])
    
    return degeneracy


def truncation_starting_point(
    Dcut: int, M: int, N: int, loc_dirc: list[int] = [0, 0], CZ: bool = True
):
    
    loc, dirc = loc_dirc
    
    nn_points, _, _ = get_topology(M, N)
    nn_loc = nn_points[loc][dirc]
    
    pair_to_search = sorted([loc, nn_loc])
    
    if CZ:
        layer = 4*int(np.log2(Dcut))
    else:
        layer = 4*int(np.log2(Dcut)/2)
    
    for it in np.arange(1, 5):
        point1_dirc_list, point1_point2_list = get_pattern(M, N, layer=layer+it)
        exists = any(sorted(pair) == pair_to_search for pair in point1_point2_list)

        if exists == True:
            break
    
    return layer + it


def format_memory(value):
    """Format memory value to appropriate units"""
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
    for unit in units:
        if value < 1000:
            return f"{value:.0f} {unit}"
        value /= 1000
    return f"{value:.0f} YB"  # Beyond ZB


def format_flops(value):
    """Format FLOPs value to appropriate units"""
    units = ['FLOP', 'KFLOP', 'MFLOP', 'GFLOP', 'TFLOP', 'PFLOP', 'EFLOP', 'ZFLOP']
    for unit in units:
        if value < 1000:
            return f"{value:.0f} {unit}"
        value /= 1000
    return f"{value:.0f} YFLOP"  # Beyond ZFLOP


def calculate_slope(contour_line, position):
    """Calculate the slope of the contour line at a given position"""
    # Extract the contour path closest to the position
    path = contour_line.collections[0].get_paths()[0]
    vertices = path.vertices
    distances = np.sqrt((vertices[:, 0] - position[0])**2 + (vertices[:, 1] - position[1])**2)
    nearest_index = np.argmin(distances)
    
    # Use the two nearest points to calculate the slope
    if nearest_index == 0:
        p1, p2 = vertices[0], vertices[1]
    elif nearest_index == len(vertices) - 1:
        p1, p2 = vertices[-2], vertices[-1]
    else:
        p1, p2 = vertices[nearest_index - 1], vertices[nearest_index + 1]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < -90 or angle > 90:
        angle += 180
        
    return angle + 60


def contour_plot(
    ax: Axes,
    x_grid,
    y_grid,
    colorbar_grid,
    label_positions,
    memory_usage,
    label_angle: list[int] | None = None,
):
    # Define memory usage levels for MB, GB, TB, PB, EB, ZB within the range of memory_usage
    memory_levels = [1e7, 1e9, 1e11, 1e13, 1e15, 7e17, 1e19]

    # Filter memory levels to those within the min and max of memory_usage
    memory_levels = [level for level in memory_levels if np.min(memory_usage) <= level <= np.max(memory_usage)]

    # Calculate corresponding FLOPs for each memory level
    # memory_usage = 2 * 16 * (qubit_grid - 4*np.sqrt(qubit_grid) + 4) * bond_dim_grid**4
    flops_levels = [5e4 * (1e9)**(-1/4) * level**(5/4) for level in memory_levels]
    hardware_levels = ["Laptop", "Desktop", "GPU", "GPU cluster", "Supercomputer", "Frontier"]
    
    # flops_grid = 5e4 * (1e9)**(-1/4) * memory_usage**(5/4)
    
    try:
        print(f"{np.log10(colorbar_grid.max())=}")
    except Exception as e:
        print("nplog10 error")
    
    color_plot = ax.contourf(
        x_grid, y_grid * 100, colorbar_grid,
        # levels=np.logspace(np.log10(colorbar_grid.min()), np.log10(colorbar_grid.max()), 30), 
        levels=np.logspace(np.log10(colorbar_grid.min()), np.log10(colorbar_grid.max()), 15), 
        cmap=plt.get_cmap('turbo'), norm=LogNorm(vmin=colorbar_grid.min()), alpha=1.0
    )

    if label_positions:
        manual_label_positions = iter(label_positions)
    
    for it, data in enumerate(zip(hardware_levels, memory_levels, flops_levels)):
        hardware, level, flops = data
        
        contour_line = ax.contour(
            x_grid, y_grid * 100, memory_usage, levels=[level], colors='black', linestyles='solid'
        )
        
        if contour_line.collections and len(contour_line.collections) > 0:
            if label_positions:
                label_pos = next(manual_label_positions)
                try:
                    clabels = ax.clabel(
                        contour_line, fmt={level: f'{hardware}\n{format_memory(level)}'},# {format_flops(flops)}'},
                        inline=True, fontsize=16, manual=[label_pos]
                    )
                    if label_angle is not None:
                        for txt in clabels:
                            txt.set_rotation(label_angle[it])
                
                except Exception as e:
                    print(e)
                    print(f"Contour level: {level} | {np.min(memory_usage)=} | {np.max(memory_usage)=}")
                    print_traceback(e)
                    sys.exit()
            else:
                ax.clabel(
                    contour_line, fmt={level: f'{hardware}\n{format_memory(level)}'},# {format_flops(flops)}'},
                    inline=True, fontsize=13
                )

    return ax, color_plot

