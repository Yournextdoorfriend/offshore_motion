import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Offshore Motion - Wave Dynamics", layout="wide")

st.title("Wave Dynamics Simulator")
st.markdown("Visualize superimposed sea waves by adjusting amplitude and period.")

# Sidebar controls
st.sidebar.header("Wave Parameters")

st.sidebar.subheader("Wave 1")
amp1 = st.sidebar.slider("Amplitude (m)", 0.0, 10.0, 2.0, 0.1, key="amp1")
period1 = st.sidebar.slider("Period (s)", 1.0, 20.0, 8.0, 0.5, key="period1")

st.sidebar.subheader("Wave 2")
amp2 = st.sidebar.slider("Amplitude (m)", 0.0, 10.0, 1.0, 0.1, key="amp2")
period2 = st.sidebar.slider("Period (s)", 1.0, 20.0, 12.0, 0.5, key="period2")

st.sidebar.subheader("Time Settings")
duration = st.sidebar.slider("Duration (s)", 10, 120, 60, 5)

# Generate time array
t = np.linspace(0, duration, 1000)

# Calculate wave elevations: η(t) = A * cos(ωt) where ω = 2π/T
omega1 = 2 * np.pi / period1
omega2 = 2 * np.pi / period2

wave1 = amp1 * np.cos(omega1 * t)
wave2 = amp2 * np.cos(omega2 * t)
combined = wave1 + wave2

# Create plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=wave1,
    mode='lines',
    name=f'Wave 1 (A={amp1}m, T={period1}s)',
    line=dict(color='#1f77b4', width=1.5, dash='dash'),
    opacity=0.7
))

fig.add_trace(go.Scatter(
    x=t, y=wave2,
    mode='lines',
    name=f'Wave 2 (A={amp2}m, T={period2}s)',
    line=dict(color='#ff7f0e', width=1.5, dash='dash'),
    opacity=0.7
))

fig.add_trace(go.Scatter(
    x=t, y=combined,
    mode='lines',
    name='Combined Wave',
    line=dict(color='#2ca02c', width=2.5)
))

fig.update_layout(
    title="Sea Surface Elevation at Fixed Point",
    xaxis_title="Time (s)",
    yaxis_title="Elevation (m)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    hovermode='x unified',
    height=500
)

# Add zero line
fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

st.plotly_chart(fig, use_container_width=True)

# Display statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Max Elevation", f"{combined.max():.2f} m")
with col2:
    st.metric("Min Elevation", f"{combined.min():.2f} m")
with col3:
    significant_height = 4 * np.std(combined)
    st.metric("Significant Wave Height (Hs)", f"{significant_height:.2f} m")
