import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# --- Webpage Header ---
st.title("Transient Heat Conduction (1D Slab)")
st.markdown("**Rapid non-uniform decay, Bi = 10**")

# --- Mathematical Logic ---
@st.cache_data
def get_eigenvalues(Bi, num_terms=10):
    lams = []
    for n in range(1, num_terms + 1):
        # The n-th root is bounded between (n-1)*pi and (n-0.5)*pi
        # We solve lam*sin(lam) - Bi*cos(lam) = 0 to avoid tan() asymptotes
        bracket = [(n - 1) * np.pi + 1e-5, (n - 0.5) * np.pi - 1e-5]
        
        def f(lam):
            return lam * np.sin(lam) - Bi * np.cos(lam)
            
        res = root_scalar(f, bracket=bracket, method='brentq')
        if res.converged:
            lams.append(res.root)
    return np.array(lams)

# --- Interactive Slider ---
# slider(Label, minimum, maximum, default_value, step_size)
tau = st.slider("Fourier Number (τ)", min_value=0.001, max_value=3.0, value=0.05, step=0.05)

# --- Computation ---
Bi = 10
X = np.linspace(0, 1, 50) # Calculates 50 points along the X axis
lam = get_eigenvalues(Bi, num_terms=10)

# Calculate the coefficients A_n
Acoef = (4 * np.sin(lam)) / (2 * lam + np.sin(2 * lam))

# Vectorized computation of the infinite series
X_grid, lam_grid = np.meshgrid(X, lam)
A_grid = Acoef[:, np.newaxis]
terms = A_grid * np.cos(lam_grid * X_grid) * np.exp(-(lam_grid**2) * tau)
Theta = np.sum(terms, axis=0)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, Theta, color="darkgreen", linewidth=2.5)

# Graph Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.3)
ax.set_xlabel("X (Dimensionless spatial coordinate)")
ax.set_ylabel("Θ (Dimensionless temperature)")
ax.grid(True, linestyle="--", alpha=0.6)

# Send the plot to the web page
st.pyplot(fig)
