import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend for Streamlit


# -----------------------------------------------------------------------
# Population firing rates
# -----------------------------------------------------------------------
def population_rate(R, NE, NI):
    """
    Compute mean population firing rates.

    Parameters
    ----------
    R   : (T_steps, N) firing-rate array
    NE  : number of excitatory neurons
    NI  : number of inhibitory neurons

    Returns
    -------
    r_total, r_E, r_I : each shape (T_steps,)
    """
    r_E = R[:, :NE].mean(axis=1)
    r_I = R[:, NE:].mean(axis=1)
    r_total = R.mean(axis=1)
    return r_total, r_E, r_I


# -----------------------------------------------------------------------
# Autocorrelation
# -----------------------------------------------------------------------
def autocorrelation(x, max_lag=200):
    """
    Compute normalised autocorrelation C(tau) = <x(t) x(t+tau)> / <x^2>.

    Parameters
    ----------
    x       : 1-D time series
    max_lag : maximum lag (in samples)

    Returns
    -------
    lags : (max_lag+1,)  lag values in samples
    C    : (max_lag+1,)  normalised autocorrelation
    """
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)

    lags = np.arange(max_lag + 1)
    C = np.array(
        [np.mean(x[: len(x) - lag] * x[lag:]) if lag < len(x) else 0.0 for lag in lags]
    )
    C /= var
    return lags, C


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------
def plot_population_rate(t, r_total, r_E, r_I, ax=None):
    """Line plot of population firing rates vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, r_total, color="k", lw=1.5, label="All neurons", alpha=0.8)
    ax.plot(t, r_E, color="#e74c3c", lw=1, label="Excitatory", alpha=0.7)
    ax.plot(t, r_I, color="#3498db", lw=1, label="Inhibitory", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean firing rate phi(x)")
    ax.set_title("Population Activity")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_figure()

def plot_raster(t, spikes, NE, n_show=100, ax=None):
    """
    Raster plot of pseudo-Poisson spikes.
    E neurons: red   I neurons: blue
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    N = spikes.shape[1]
    n_show = min(n_show, N)
    indices = np.linspace(0, N - 1, n_show, dtype=int)

    for plot_idx, neuron_idx in enumerate(indices):
        spike_times = t[spikes[:, neuron_idx]]
        color = "#e74c3c" if neuron_idx < NE else "#3498db"
        ax.scatter(
            spike_times,
            np.full_like(spike_times, plot_idx),
            s=0.5,
            color=color,
            alpha=0.6,
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", label="Excitatory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", label="Inhibitory"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"Raster Plot (showing {n_show} neurons)")
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_figure()

def plot_single_neuron(t, X, n_neurons=5, ax=None):
    """Traces of x(t) for a handful of neurons, vertically offset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    N = X.shape[1]
    rng = np.random.default_rng(0)
    chosen = rng.choice(N, size=min(n_neurons, N), replace=False)
    chosen.sort()

    offset_scale = np.std(X) * 3 if np.std(X) > 0 else 1.0
    cmap = plt.cm.tab10

    for k, idx in enumerate(chosen):
        offset = k * offset_scale
        ax.plot(t, X[:, idx] + offset, lw=0.8, color=cmap(k), label=f"Neuron {idx}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("State x(t) + offset")
    ax.set_title("Single Neuron Traces")
    ax.legend(loc="upper right", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_figure()

def plot_autocorrelation(lags, C, dt, g, ax=None):
    """
    Autocorrelation C(tau) with decay annotation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    tau_axis = lags * dt
    ax.plot(tau_axis, C, color="#8e44ad", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(1 / np.e, color="orange", lw=0.8, ls=":", label="1/e threshold")

    decay_idx = np.where(C < 1 / np.e)[0]
    if len(decay_idx) > 0:
        decay_time = tau_axis[decay_idx[0]]
        ax.axvline(decay_time, color="orange", lw=0.8, ls=":")
        regime = "fast decay (chaotic)" if g >= 1.2 else "slow decay (stable)"
        ax.text(
            decay_time + 1,
            0.6,
            f"tau_decay ~{decay_time:.1f} ms\n{regime}",
            fontsize=8,
            color="darkorange",
        )

    ax.set_xlabel("Lag tau (ms)")
    ax.set_ylabel("C(tau)")
    ax.set_title("Population Autocorrelation")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_figure()