import numpy as np


class EIRateNetwork:
    """
    E-I balanced random rate network (Sompolinsky 1988 style).

    Parameters
    ----------
    N        : total number of neurons
    frac_E   : fraction of excitatory neurons (default 0.8)
    J_EE     : E→E coupling strength (>0)
    J_EI     : E→I coupling strength (<0, set to -g*J_EE)
    J_IE     : I→E coupling strength (>0)
    J_II     : I→I coupling strength (<0)
    p        : connection probability (used for Sompolinsky scaling)
    tau      : membrane time constant (ms)
    dt       : integration time step (ms)
    phi_type : activation function: 'tanh' | 'relu' | 'sigmoid'
    seed     : random seed for reproducibility
    """

    def __init__(
        self,
        N=200,
        frac_E=0.8,
        J_EE=1.0,
        J_EI=-1.2,
        J_IE=1.0,
        J_II=-1.0,
        p=0.5,
        tau=10.0,
        dt=0.1,
        phi_type="tanh",
        seed=42,
    ):
        self.N = N
        self.NE = int(N * frac_E)
        self.NI = N - self.NE
        self.J_EE = J_EE
        self.J_EI = J_EI
        self.J_IE = J_IE
        self.J_II = J_II
        self.p = p
        self.tau = tau
        self.dt = dt
        self.phi_type = phi_type
        self.rng = np.random.default_rng(seed)

        self.J = self.build_weight_matrix()

    # ------------------------------------------------------------------
    # Weight matrix
    # ------------------------------------------------------------------
    def build_weight_matrix(self):
        """
        Build block-structured random weight matrix with Sompolinsky scaling:
            J_ij = (J_ab / sqrt(K)) * xi_ij,  xi ~ N(0,1),  K = p*N
        """
        N, NE, NI = self.N, self.NE, self.NI
        K = self.p * N  # effective in-degree

        J = np.zeros((N, N))

        # Helper: random Gaussian block scaled by J_ab / sqrt(K)
        def block(rows, cols, strength):
            return (strength / np.sqrt(K)) * self.rng.standard_normal((rows, cols))

        # E→E  (top-left)
        J[:NE, :NE] = block(NE, NE, self.J_EE)
        # I→E  (top-right: E rows, I cols)
        J[:NE, NE:] = block(NE, NI, self.J_EI)
        # E→I  (bottom-left: I rows, E cols)
        J[NE:, :NE] = block(NI, NE, self.J_IE)
        # I→I  (bottom-right)
        J[NE:, NE:] = block(NI, NI, self.J_II)

        # Zero out self-connections
        np.fill_diagonal(J, 0.0)
        return J

    # ------------------------------------------------------------------
    # Activation function
    # ------------------------------------------------------------------
    def phi(self, x):
        if self.phi_type == "tanh":
            return np.tanh(x)
        elif self.phi_type == "relu":
            return np.maximum(0.0, x)
        elif self.phi_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        else:
            raise ValueError(f"Unknown phi_type: {self.phi_type}")

    # ------------------------------------------------------------------
    # Single Euler step
    # ------------------------------------------------------------------
    def step(self, x, I):
        """
        Advance state by one time step.

        Parameters
        ----------
        x : (N,) current state
        I : (N,) external input

        Returns
        -------
        x_new  : (N,) updated state
        r      : (N,) firing rates phi(x)
        spikes : (N,) bool, Poisson-sampled pseudo-spikes
        """
        r = self.phi(x)
        dx = (-x + self.J @ r + I) * (self.dt / self.tau)
        x_new = x + dx

        # Poisson spike sampling: P(spike) = r * dt  (clip to [0,1])
        prob = np.clip(np.abs(r) * self.dt, 0.0, 1.0)
        spikes = self.rng.random(self.N) < prob

        return x_new, r, spikes

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------
    def run(self, T=500.0, input_type="noisy", mu=0.0, sigma=0.1, I0=0.0):
        """
        Run the simulation for T ms.

        Parameters
        ----------
        T          : simulation duration (ms)
        input_type : 'constant' or 'noisy'
        mu         : mean of external input
        sigma      : std of noise (used when input_type='noisy')
        I0         : constant input amplitude

        Returns
        -------
        dict with keys: t, X, R, spikes
        """
        n_steps = int(T / self.dt)
        t = np.arange(n_steps) * self.dt

        X = np.zeros((n_steps, self.N))
        R = np.zeros((n_steps, self.N))
        Spikes = np.zeros((n_steps, self.N), dtype=bool)

        # Initial condition: small random state
        x = 0.1 * self.rng.standard_normal(self.N)

        for i in range(n_steps):
            # Build input
            if input_type == "constant":
                I = np.full(self.N, I0 + mu)
            else:  # noisy
                I = mu + sigma * self.rng.standard_normal(self.N)

            x, r, spikes = self.step(x, I)

            X[i] = x
            R[i] = r
            Spikes[i] = spikes

        return {"t": t, "X": X, "R": R, "spikes": Spikes}