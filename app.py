import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import root_scalar

# --- Webpage Header ---
st.title("Transient Heat Conduction (1D Slab)")
st.markdown("**Rapid non-uniform decay, Bi = 10**")

# --- Mathematical Logic ---
@st.cache_data
def get_eigenvalues(Bi, num_terms=10):
    lams = []
    for n in range(1, num_terms + 1):
        bracket = [(n - 1) * np.pi + 1e-5, (n - 0.5) * np.pi - 1e-5]
        
        def f(lam):
            return lam * np.sin(lam) - Bi * np.cos(lam)
            
        res = root_scalar(f, bracket=bracket, method='brentq')
        if res.converged:
            lams.append(res.root)
    return np.array(lams)

# --- Interactive Slider ---
tau = st.slider("Fourier Number (τ)", min_value=0.001, max_value=3.0, value=0.05, step=0.05)

# --- Computation ---
Bi = 10
X = np.linspace(0, 1, 100) # Increased resolution for smoother Plotly rendering
lam = get_eigenvalues(Bi, num_terms=10)

Acoef = (4 * np.sin(lam)) / (2 * lam + np.sin(2 * lam))

X_grid, lam_grid = np.meshgrid(X, lam)
A_grid = Acoef[:, np.newaxis]
terms = A_grid * np.cos(lam_grid * X_grid) * np.exp(-(lam_grid**2) * tau)
Theta = np.sum(terms, axis=0)

# --- Plotly Rendering ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=X, 
    y=Theta, 
    mode='lines', 
    line=dict(color='darkgreen', width=3),
    name="Temperature Profile"
))

fig.update_layout(
    xaxis_title="X (Dimensionless spatial coordinate)",
    yaxis_title="Θ (Dimensionless temperature)",
    yaxis=dict(range=[0, 1.3]),
    xaxis=dict(range=[0, 1]),
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20)
)

# Send the interactive plot to the web page
st.plotly_chart(fig, use_container_width=True)
