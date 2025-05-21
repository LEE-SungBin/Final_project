import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap

arr = npt.NDArray[np.generic]


def log_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    log-log scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()

    poly, residual, _, _, _ = np.polyfit(
        np.log10(raw_x[(start <= raw_x) & (raw_x <= end)]), np.log10(raw_y[
            (start <= raw_x) & (raw_x <= end)]), 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = pow(10.0, poly[1] - offset) * np.power(fit_x, poly[0])
    
    n = len(raw_x[(start <= raw_x) & (raw_x <= end)])
    df = n - 2  # degrees of freedom (n - number of parameters)
    
    if df == 0:
        print(f"{raw_x[(start <= raw_x) & (raw_x <= end)]=}")
        print(f"{raw_x=}")
        print(f"{start=} {end=}")
    
    # Residual sum of squares
    rss = residual[0] if len(residual) > 0 else 0.0
    
    # Mean squared error
    mse = rss / df
    
    # Standard error of the regression (variance of residuals)
    stderr = np.sqrt(mse)
    
    return fit_x, fit_y, poly[0], stderr


def lin_log_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    lin-log scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(raw_x, np.log10(raw_y), 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = np.power(10.0, poly[0] * fit_x + poly[1] - offset)
    return fit_x, fit_y, poly[0], residual[0]


def log_lin_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    log-lin scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(np.log10(raw_x), raw_y, 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = poly[0] * np.log10(fit_x) + poly[1] - offset
    return fit_x, fit_y, poly[0], residual[0]


def lin_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    lin-lin scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(raw_x, raw_y, 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = poly[0] * fit_x + poly[1] - offset
    return fit_x, fit_y, poly[0], residual[0]


def log_log_line(
    x0: float, y0: float, slope: float, x1: float, ax: Axes | None = None, **kwargs
) -> float:
    """
    Draw line at log-log plot with passing (x0, y0), with slope.
    Another end point is x1
    """
    y1 = np.power(x1 / x0, slope) * y0
    if ax:
        ax.plot([x0, x1], [y0, y1], **kwargs)

    return x1


def get_color(value, cvals, colors, mode="log"):
    """
    Return a color from a custom colormap for a given value, with either linear
    or logarithmic scaling.

    Parameters:
    - value: The data value to convert to a color.
    - cvals: The control values defining the color transitions.
    - colors: The colors corresponding to the control values.
    - mode: "linear" or "log" to specify the type of scaling.

    Returns:
    - A RGBA color for the given value.
    """
    # Ensure cvals and colors are sorted together based on cvals
    cvals, colors = zip(*sorted(zip(cvals, colors)))
    
    # Determine normalization based on mode
    if mode == "log":
        norm = LogNorm(vmin=min(cvals), vmax=max(cvals))
    elif mode == "linear":
        norm = Normalize(vmin=min(cvals), vmax=max(cvals))
    else:
        raise ValueError("mode must be 'linear' or 'log'")

    # Create a colormap from the provided control values and colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Normalize the value to [0, 1] range according to the chosen normalization
    normalized_value = norm(value)

    # Get the corresponding color from the colormap
    return cmap(normalized_value)


def get_colormap(cvals: list[float], colors: list[str], mode: str = "linear") -> tuple[LinearSegmentedColormap, Normalize]:
    """
    Create a LinearSegmentedColormap with specified control values and colors.
    Optionally apply a logarithmic normalization.

    Parameters:
    - cvals: Control values for the colormap transitions.
    - colors: List of matplotlib color strings for each control value.
    - mode: "linear" or "log" for the type of normalization.

    Returns:
    - A tuple of (LinearSegmentedColormap, Normalize or LogNorm instance)
    """

    # Normalize control values to range [0, 1]
    cvals_norm = [float(val - min(cvals)) / (max(cvals) - min(cvals)) for val in cvals]

    if mode == "linear":
        norm = Normalize(min(cvals), max(cvals))
    elif mode == "log":
        norm = LogNorm(min(cvals), max(cvals))
    else:
        raise ValueError("mode must be either 'linear' or 'log'")

    # Create a colormap from the provided control values and colors.
    cmap = LinearSegmentedColormap.from_list("", list(zip(cvals_norm, colors)))

    return cmap, norm

