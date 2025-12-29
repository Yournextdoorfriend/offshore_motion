# Offshore Motion Simulator

A Streamlit app for simulating FPSO vessel motion response to sea waves, with focus on drone contact point analysis.

## Features

- **Wave Input**: Define 2 superimposed wave components (amplitude, period, direction)
- **Vessel & RAO**: Configure FPSO dimensions, compute Response Amplitude Operators using Capytaine BEM solver
- **Motion Response**: Visualize 6-DOF vessel motions (heave, pitch, roll, surge, sway, yaw)
- **Point Motion**: Analyze velocity/acceleration at specific hull contact points in drone-centric coordinates

## Installation

```bash
cd offshore_motion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Workflow

1. **Wave Input tab**: Set wave parameters (amplitude, period, direction for 2 wave components)
2. **Vessel & RAO tab**: Adjust FPSO dimensions if needed, click "Compute RAOs" (takes ~10-30s)
3. **Motion Response tab**: View vessel motions in all 6 degrees of freedom
4. **Point Motion tab**: Select a contact point on the hull, view velocities/accelerations in drone frame

### Drone Contact Frame

For the Point Motion analysis, velocities are expressed in a drone-centric frame:
- **Vx**: Tangential (fore-aft along hull surface)
- **Vy**: Normal (perpendicular to hull, + = away from surface)
- **Vz**: Tangential (vertical along hull surface)

### Export

The Point Motion tab includes a CSV export button for 2 minutes of data at 20 Hz, containing velocities and accelerations in the drone frame.

## Project Structure

```
offshore_motion/
├── app.py           # Streamlit application
├── hull.py          # Hull geometry generation
├── rao.py           # RAO computation with Capytaine
├── requirements.txt # Python dependencies
└── README.md
```

## Hull Model

The FPSO hull is modeled with:
- Tapered bow (15% of length, parabolic)
- Parallel mid-body (75% of length)
- Tapered stern (10% of length)
- Default dimensions: 300m × 54m × 22m draft

## Dependencies

- Streamlit (UI)
- Capytaine (BEM hydrodynamic solver)
- NumPy, SciPy (numerical computation)
- Plotly (visualization)
- xarray, pandas (data handling)
