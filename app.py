import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import root_scalar

# --- Webpage Header ---
st.set_page_config(page_title="Heat Conduction Model", layout="centered")
st.title("Transient Heat Conduction (1D Slab)")
st.markdown("Interactive visualization of temperature decay and characteristic eigenvalue roots.")

# --- Controls (Sidebar) ---
st.sidebar.header("System Parameters")
Bi = st.sidebar.slider("Biot Number (Bi)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
tau = st.sidebar.slider("Fourier Number (τ)", min_value=0.001, max_value=3.0, value=0.05, step=0.05)

# --- Mathematical Logic ---
# Caching makes the slider lightning fast by memorizing roots for a given Bi
@st.cache_data
def get_eigenvalues(Bi_val, num_terms=10):
    lams = []
    for n in range(1, num_terms + 1):
        bracket = [(n - 1) * np.pi + 1e-5, (n - 0.5) * np.pi - 1e-5]
        
        def f(lam):
            return lam * np.sin(lam) - Bi_val * np.cos(lam)
            
        res = root_scalar(f, bracket=bracket, method='brentq')
        if res.converged:
            lams.append(res.root)
    return np.array(lams)

# Compute roots based on the current Bi slider value
lam = get_eigenvalues(Bi, num_terms=10)

# ==========================================
# PLOT 1: Temperature Profile
# ==========================================
st.subheader("1. Temperature Profile")

X = np.linspace(0, 1, 100)
Acoef = (4 * np.sin(lam)) / (2 * lam + np.sin(2 * lam))

# Vectorized math for the infinite series
X_grid, lam_grid = np.meshgrid(X, lam)
A_grid = Acoef[:, np.newaxis]
terms = A_grid * np.cos(lam_grid * X_grid) * np.exp(-(lam_grid**2) * tau)
Theta = np.sum(terms, axis=0)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=X, 
    y=Theta, 
    mode='lines', 
    line=dict(color='darkgreen', width=3),
    name="Temperature Profile"
))
