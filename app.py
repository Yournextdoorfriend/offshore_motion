import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Offshore Motion Simulator", layout="wide")

st.title("Offshore Motion Simulator")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Wave Input", "Vessel & RAO", "Motion Response", "Point Motion"])

# Session state for RAO data
if 'rao_data' not in st.session_state:
    st.session_state.rao_data = None
if 'vessel_params' not in st.session_state:
    st.session_state.vessel_params = {'length': 300.0, 'beam': 54.0, 'draft': 22.0}

# ============== TAB 1: WAVE INPUT ==============
with tab1:
    st.header("Sea State Definition")
    st.markdown("Define up to 2 superimposed wave components.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Wave 1 (Primary)")
        amp1 = st.slider("Amplitude (m)", 0.0, 10.0, 2.0, 0.1, key="amp1")
        period1 = st.slider("Period (s)", 3.0, 25.0, 10.0, 0.5, key="period1")
        direction1 = st.slider("Direction (deg)", 0, 180, 180, 15, key="dir1",
                               help="0°=Following, 90°=Beam, 180°=Head seas")

    with col2:
        st.subheader("Wave 2 (Secondary)")
        amp2 = st.slider("Amplitude (m)", 0.0, 10.0, 0.5, 0.1, key="amp2")
        period2 = st.slider("Period (s)", 3.0, 25.0, 6.0, 0.5, key="period2")
        direction2 = st.slider("Direction (deg)", 0, 180, 90, 15, key="dir2",
                               help="0°=Following, 90°=Beam, 180°=Head seas")

    duration = st.slider("Simulation Duration (s)", 30, 300, 120, 10)

    # Generate time array
    t = np.linspace(0, duration, 1000)

    # Calculate wave elevations
    omega1 = 2 * np.pi / period1
    omega2 = 2 * np.pi / period2

    wave1 = amp1 * np.cos(omega1 * t)
    wave2 = amp2 * np.cos(omega2 * t)
    combined_wave = wave1 + wave2

    # Wave plot
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=wave1, mode='lines',
                                   name=f'Wave 1 (A={amp1}m, T={period1}s, {direction1}°)',
                                   line=dict(color='#1f77b4', width=1.5, dash='dash'), opacity=0.7))
    fig_wave.add_trace(go.Scatter(x=t, y=wave2, mode='lines',
                                   name=f'Wave 2 (A={amp2}m, T={period2}s, {direction2}°)',
                                   line=dict(color='#ff7f0e', width=1.5, dash='dash'), opacity=0.7))
    fig_wave.add_trace(go.Scatter(x=t, y=combined_wave, mode='lines',
                                   name='Combined Wave', line=dict(color='#2ca02c', width=2.5)))
    fig_wave.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_wave.update_layout(title="Sea Surface Elevation", xaxis_title="Time (s)",
                           yaxis_title="Elevation (m)", height=400,
                           legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_wave, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Elevation", f"{combined_wave.max():.2f} m")
    with col2:
        st.metric("Min Elevation", f"{combined_wave.min():.2f} m")
    with col3:
        hs = 4 * np.std(combined_wave)
        st.metric("Significant Wave Height (Hs)", f"{hs:.2f} m")


# ============== TAB 2: VESSEL & RAO ==============
with tab2:
    st.header("Vessel Configuration")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("FPSO Dimensions")
        length = st.number_input("Length (m)", 100.0, 400.0, 300.0, 10.0)
        beam = st.number_input("Beam (m)", 20.0, 80.0, 54.0, 2.0)
        draft = st.number_input("Draft (m)", 5.0, 30.0, 22.0, 1.0)

        # Hull profile visualization
        st.subheader("Hull Profile")
        bow_len = 0.15 * length
        stern_len = 0.10 * length

        # Generate hull outline
        x_hull = np.linspace(-length/2, length/2, 100)
        y_hull = []
        for x in x_hull:
            bow_start = length/2 - bow_len
            stern_end = -length/2 + stern_len
            if x > bow_start:
                t = (x - bow_start) / bow_len
                y_hull.append((beam/2) * (1 - t**2))
            elif x < stern_end:
                t = (stern_end - x) / stern_len
                y_hull.append((beam/2) * (1 - 0.5*t**2))
            else:
                y_hull.append(beam/2)
        y_hull = np.array(y_hull)

        # Create 3-view figure
        fig_hull = make_subplots(rows=1, cols=3, subplot_titles=("Top View", "Side View", "Front View"),
                                  horizontal_spacing=0.08)

        # Top view (plan)
        fig_hull.add_trace(go.Scatter(x=x_hull, y=y_hull, mode='lines', line=dict(color='#1f77b4', width=2),
                                       showlegend=False), row=1, col=1)
        fig_hull.add_trace(go.Scatter(x=x_hull, y=-y_hull, mode='lines', line=dict(color='#1f77b4', width=2),
                                       showlegend=False), row=1, col=1)

        # Side view (profile)
        fig_hull.add_trace(go.Scatter(x=[-length/2, length/2, length/2, -length/2, -length/2],
                                       y=[0, 0, -draft, -draft, 0], mode='lines', fill='toself',
                                       fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4', width=2),
                                       showlegend=False), row=1, col=2)
        fig_hull.add_hline(y=0, line_dash="dash", line_color="blue", row=1, col=2)

        # Front view (section at midship)
        fig_hull.add_trace(go.Scatter(x=[-beam/2, beam/2, beam/2, -beam/2, -beam/2],
                                       y=[0, 0, -draft, -draft, 0], mode='lines', fill='toself',
                                       fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4', width=2),
                                       showlegend=False), row=1, col=3)
        fig_hull.add_hline(y=0, line_dash="dash", line_color="blue", row=1, col=3)

        fig_hull.update_xaxes(title_text="x (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
        fig_hull.update_xaxes(title_text="x (m)", row=1, col=2)
        fig_hull.update_xaxes(title_text="y (m)", row=1, col=3)
        fig_hull.update_yaxes(title_text="y (m)", row=1, col=1)
        fig_hull.update_yaxes(title_text="z (m)", row=1, col=2)
        fig_hull.update_yaxes(title_text="z (m)", row=1, col=3)
        fig_hull.update_layout(height=250, margin=dict(t=30, b=30))

        st.plotly_chart(fig_hull, use_container_width=True)

        st.subheader("Mesh Resolution")
        mesh_res = st.selectbox("Mesh quality", ["Coarse (fast)", "Medium", "Fine (slow)"],
                                help="Higher resolution = more accurate but slower")

        if st.button("Compute RAOs", type="primary"):
            with st.spinner("Computing hydrodynamic coefficients... This may take a minute."):
                try:
                    from hull import create_fpso_hull, get_vessel_mass_properties
                    from rao import compute_raos, solve_motion_raos

                    # Set mesh resolution
                    if mesh_res == "Coarse (fast)":
                        nx, ny, nz = 15, 6, 4
                    elif mesh_res == "Medium":
                        nx, ny, nz = 25, 10, 6
                    else:
                        nx, ny, nz = 40, 15, 8

                    # Create hull
                    body = create_fpso_hull(length, beam, draft, nx=nx, ny=ny, nz=nz)

                    # Compute RAOs for range of periods and directions
                    periods = np.linspace(4, 25, 15)
                    directions = np.array([0, 45, 90, 135, 180])

                    mass_props = get_vessel_mass_properties(length, beam, draft)
                    dataset = compute_raos(body, periods, directions, mass_props=mass_props)
                    dataset = solve_motion_raos(dataset)

                    st.session_state.rao_data = dataset
                    st.session_state.vessel_params = {'length': length, 'beam': beam, 'draft': draft}

                    st.success(f"RAOs computed! Mesh: {body.mesh.nb_faces} panels")
                except Exception as e:
                    st.error(f"Error computing RAOs: {e}")

    with col2:
        st.subheader("RAO Curves")

        if st.session_state.rao_data is not None:
            dataset = st.session_state.rao_data
            params = st.session_state.vessel_params

            st.caption(f"Vessel: {params['length']}m × {params['beam']}m × {params['draft']}m draft")

            # Select DOF and direction to plot
            dof_select = st.selectbox("Degree of Freedom", ["Heave", "Pitch", "Roll", "Surge", "Sway", "Yaw"])
            dir_select = st.selectbox("Wave Direction", [0, 45, 90, 135, 180],
                                      format_func=lambda x: f"{x}° ({'Head' if x==180 else 'Beam' if x==90 else 'Following' if x==0 else 'Quartering'})")

            # Get RAO data
            periods = 2 * np.pi / dataset['omega'].values
            dir_rad = np.deg2rad(dir_select)

            # Find nearest direction index manually (avoids monotonic index requirement)
            dir_idx = np.argmin(np.abs(dataset['wave_direction'].values - dir_rad))
            rao_amp = dataset['RAO_amplitude'].sel(influenced_dof=dof_select).values[:, dir_idx]

            # Convert rotational RAOs to deg/m
            if dof_select in ['Roll', 'Pitch', 'Yaw']:
                rao_amp = np.rad2deg(rao_amp)
                unit = "deg/m"
            else:
                unit = "m/m"

            # Plot RAO
            fig_rao = go.Figure()
            fig_rao.add_trace(go.Scatter(x=periods, y=rao_amp, mode='lines+markers',
                                         name=f'{dof_select} RAO', line=dict(width=2)))
            fig_rao.update_layout(
                title=f"{dof_select} RAO - {dir_select}° waves",
                xaxis_title="Wave Period (s)",
                yaxis_title=f"RAO ({unit})",
                height=400
            )
            st.plotly_chart(fig_rao, use_container_width=True)
        else:
            st.info("Click 'Compute RAOs' to generate response amplitude operators for the vessel.")


# ============== TAB 3: MOTION RESPONSE ==============
with tab3:
    st.header("Vessel Motion Response")

    if st.session_state.rao_data is None:
        st.warning("Please compute RAOs in the 'Vessel & RAO' tab first.")
    else:
        dataset = st.session_state.rao_data
        params = st.session_state.vessel_params

        st.caption(f"Vessel: {params['length']}m × {params['beam']}m × {params['draft']}m draft")

        # Recreate time array (ensures it's available regardless of tab order)
        t_motion = np.linspace(0, duration, 1000)

        # Get RAO values at the wave periods (interpolate)
        from scipy.interpolate import interp1d

        rao_periods = 2 * np.pi / dataset['omega'].values
        directions_available = np.rad2deg(dataset['wave_direction'].values)

        # Function to get interpolated RAO
        def get_rao(dof, period, direction):
            dir_rad = np.deg2rad(direction)
            # Find nearest direction
            dir_idx = np.argmin(np.abs(dataset['wave_direction'].values - dir_rad))
            rao_vals = dataset['RAO_amplitude'].sel(influenced_dof=dof).values[:, dir_idx]
            phase_vals = dataset['RAO_phase'].sel(influenced_dof=dof).values[:, dir_idx]

            # Interpolate
            if period < rao_periods.min() or period > rao_periods.max():
                return 0.0, 0.0
            f_amp = interp1d(rao_periods, rao_vals, kind='linear')
            f_phase = interp1d(rao_periods, phase_vals, kind='linear')
            return float(f_amp(period)), float(f_phase(period))

        # Compute vessel motions for each wave component
        def compute_motion(dof, amp, period, direction, time_arr, phase_offset=0):
            rao_amp, rao_phase = get_rao(dof, period, direction)
            omega = 2 * np.pi / period
            response_amp = amp * rao_amp
            # Motion = RAO * wave_amplitude * cos(ωt + phase)
            return response_amp * np.cos(omega * time_arr + np.deg2rad(rao_phase) + phase_offset)

        # Combined motion from both waves
        def total_motion(dof):
            m1 = compute_motion(dof, amp1, period1, direction1, t_motion)
            m2 = compute_motion(dof, amp2, period2, direction2, t_motion)
            return m1 + m2

        # Compute all motions
        heave = total_motion("Heave")
        pitch = np.rad2deg(total_motion("Pitch"))  # Convert to degrees
        roll = np.rad2deg(total_motion("Roll"))
        surge = total_motion("Surge")
        sway = total_motion("Sway")
        yaw = np.rad2deg(total_motion("Yaw"))

        # Create subplots
        fig = make_subplots(rows=3, cols=2, subplot_titles=(
            "Heave (m)", "Surge (m)", "Pitch (deg)", "Sway (m)", "Roll (deg)", "Yaw (deg)"
        ), vertical_spacing=0.15, horizontal_spacing=0.1)

        fig.add_trace(go.Scatter(x=t_motion, y=heave, name='Heave', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_motion, y=surge, name='Surge', line=dict(color='#ff7f0e')), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_motion, y=pitch, name='Pitch', line=dict(color='#2ca02c')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_motion, y=sway, name='Sway', line=dict(color='#d62728')), row=2, col=2)
        fig.add_trace(go.Scatter(x=t_motion, y=roll, name='Roll', line=dict(color='#9467bd')), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_motion, y=yaw, name='Yaw', line=dict(color='#8c564b')), row=3, col=2)

        fig.update_layout(height=800, showlegend=False, margin=dict(t=40))
        # Only add x-axis label to bottom row
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Motion statistics
        st.subheader("Motion Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Heave (max)", f"{np.max(np.abs(heave)):.2f} m")
            st.metric("Surge (max)", f"{np.max(np.abs(surge)):.2f} m")
        with col2:
            st.metric("Pitch (max)", f"{np.max(np.abs(pitch)):.2f}°")
            st.metric("Sway (max)", f"{np.max(np.abs(sway)):.2f} m")
        with col3:
            st.metric("Roll (max)", f"{np.max(np.abs(roll)):.2f}°")
            st.metric("Yaw (max)", f"{np.max(np.abs(yaw)):.2f}°")


# ============== TAB 4: POINT MOTION ==============
with tab4:
    st.header("Motion at Hull Point")
    st.markdown("Analyze motion at a specific point on the hull for drone contact scenarios.")

    if st.session_state.rao_data is None:
        st.warning("Please compute RAOs in the 'Vessel & RAO' tab first.")
    else:
        dataset = st.session_state.rao_data
        params = st.session_state.vessel_params
        L, B, D = params['length'], params['beam'], params['draft']

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Point Location")

            # Hull surface selection
            surface = st.selectbox("Hull Surface",
                ["Port Side", "Starboard Side", "Bow", "Stern", "Custom"])

            # Freeboard height (above waterline) - typical FPSO ~8-10m
            freeboard = 8.0

            if surface == "Port Side":
                default_x, default_y, default_z = 0.0, B/2, freeboard/2
                normal_vec = np.array([0, 1, 0])  # Pointing outward (port)
            elif surface == "Starboard Side":
                default_x, default_y, default_z = 0.0, -B/2, freeboard/2
                normal_vec = np.array([0, -1, 0])  # Pointing outward (starboard)
            elif surface == "Bow":
                default_x, default_y, default_z = L/2 * 0.9, 0.0, freeboard/2
                normal_vec = np.array([1, 0, 0])  # Pointing forward
            elif surface == "Stern":
                default_x, default_y, default_z = -L/2, 0.0, freeboard/2
                normal_vec = np.array([-1, 0, 0])  # Pointing aft
            else:
                default_x, default_y, default_z = 0.0, B/2, freeboard/2
                normal_vec = np.array([0, 1, 0])

            st.caption("Point coordinates (vessel frame)")
            point_x = st.slider("X - Along length (m)", -L/2, L/2, default_x, 1.0,
                               help="Positive = towards bow")
            point_y = st.slider("Y - Athwartships (m)", -B/2, B/2, default_y, 1.0,
                               help="Positive = port side")
            point_z = st.slider("Z - Vertical (m)", -D, 30.0, default_z, 0.5,
                               help="0 = waterline, + above, - below. Typical freeboard ~8m, deck ~10-15m")

            if surface == "Custom":
                st.caption("Custom normal direction")
                nx = st.number_input("Normal X", -1.0, 1.0, 0.0, 0.1)
                ny = st.number_input("Normal Y", -1.0, 1.0, 1.0, 0.1)
                nz = st.number_input("Normal Z", -1.0, 1.0, 0.0, 0.1)
                normal_vec = np.array([nx, ny, nz])
                normal_vec = normal_vec / np.linalg.norm(normal_vec)  # Normalize

            # Point position vector from center of rotation
            point_pos = np.array([point_x, point_y, point_z])
            cor = np.array([0, 0, -D * 0.4])  # Center of rotation
            r_vec = point_pos - cor  # Position relative to CoR

            st.markdown("---")
            st.caption(f"**Distance from CoR:** {np.linalg.norm(r_vec):.1f} m")

        with col2:
            st.subheader("Point Visualization")

            # Show point on hull (top and side view)
            fig_point = make_subplots(rows=1, cols=2, subplot_titles=("Top View", "Side View"),
                                       horizontal_spacing=0.12)

            # Hull outline (top view) - reuse the hull shape logic
            bow_len = 0.15 * L
            stern_len = 0.10 * L
            x_hull = np.linspace(-L/2, L/2, 100)
            y_hull = []
            for x in x_hull:
                bow_start = L/2 - bow_len
                stern_end = -L/2 + stern_len
                if x > bow_start:
                    t = (x - bow_start) / bow_len
                    y_hull.append((B/2) * (1 - t**2))
                elif x < stern_end:
                    t = (stern_end - x) / stern_len
                    y_hull.append((B/2) * (1 - 0.5*t**2))
                else:
                    y_hull.append(B/2)
            y_hull = np.array(y_hull)

            # Top view
            fig_point.add_trace(go.Scatter(x=x_hull, y=y_hull, mode='lines',
                line=dict(color='#1f77b4', width=2), showlegend=False), row=1, col=1)
            fig_point.add_trace(go.Scatter(x=x_hull, y=-y_hull, mode='lines',
                line=dict(color='#1f77b4', width=2), showlegend=False), row=1, col=1)
            # Point marker
            fig_point.add_trace(go.Scatter(x=[point_x], y=[point_y], mode='markers',
                marker=dict(size=12, color='red', symbol='x'), name='Contact Point'), row=1, col=1)
            # Normal arrow
            arrow_scale = 20
            fig_point.add_annotation(x=point_x + normal_vec[0]*arrow_scale, y=point_y + normal_vec[1]*arrow_scale,
                ax=point_x, ay=point_y, xref='x1', yref='y1', axref='x1', ayref='y1',
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='green')

            # Side view (showing freeboard above waterline)
            fig_point.add_trace(go.Scatter(
                x=[-L/2, L/2, L/2, -L/2, -L/2], y=[freeboard, freeboard, -D, -D, freeboard],
                mode='lines', fill='toself', fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1f77b4', width=2), showlegend=False), row=1, col=2)
            fig_point.add_hline(y=0, line_dash="dash", line_color="blue", row=1, col=2,
                               annotation_text="WL", annotation_position="right")
            # Point marker
            fig_point.add_trace(go.Scatter(x=[point_x], y=[point_z], mode='markers',
                marker=dict(size=12, color='red', symbol='x'), showlegend=False), row=1, col=2)

            fig_point.update_xaxes(title_text="x (m)", row=1, col=1)
            fig_point.update_xaxes(title_text="x (m)", row=1, col=2)
            fig_point.update_yaxes(title_text="y (m)", row=1, col=1, scaleanchor="x1")
            fig_point.update_yaxes(title_text="z (m)", row=1, col=2)
            fig_point.update_layout(height=300, margin=dict(t=30, b=30), showlegend=False)

            st.plotly_chart(fig_point, use_container_width=True)

            # Compute motion at point
            st.subheader("Motion Analysis")

            # Recreate time array
            t_point = np.linspace(0, duration, 1000)
            dt = t_point[1] - t_point[0]

            # Get RAO interpolation function (reuse from tab3)
            from scipy.interpolate import interp1d
            rao_periods = 2 * np.pi / dataset['omega'].values

            def get_rao(dof, period, direction):
                dir_rad = np.deg2rad(direction)
                dir_idx = np.argmin(np.abs(dataset['wave_direction'].values - dir_rad))
                rao_vals = dataset['RAO_amplitude'].sel(influenced_dof=dof).values[:, dir_idx]
                phase_vals = dataset['RAO_phase'].sel(influenced_dof=dof).values[:, dir_idx]
                if period < rao_periods.min() or period > rao_periods.max():
                    return 0.0, 0.0
                f_amp = interp1d(rao_periods, rao_vals, kind='linear')
                f_phase = interp1d(rao_periods, phase_vals, kind='linear')
                return float(f_amp(period)), float(f_phase(period))

            def compute_motion(dof, amp, period, direction, time_arr):
                rao_amp, rao_phase = get_rao(dof, period, direction)
                omega = 2 * np.pi / period
                response_amp = amp * rao_amp
                return response_amp * np.cos(omega * time_arr + np.deg2rad(rao_phase))

            def total_motion(dof):
                m1 = compute_motion(dof, amp1, period1, direction1, t_point)
                m2 = compute_motion(dof, amp2, period2, direction2, t_point)
                return m1 + m2

            # Get 6DOF motions
            surge_m = total_motion("Surge")
            sway_m = total_motion("Sway")
            heave_m = total_motion("Heave")
            roll_m = total_motion("Roll")    # radians
            pitch_m = total_motion("Pitch")  # radians
            yaw_m = total_motion("Yaw")      # radians

            # Velocity = derivative of position
            surge_v = np.gradient(surge_m, dt)
            sway_v = np.gradient(sway_m, dt)
            heave_v = np.gradient(heave_m, dt)
            roll_v = np.gradient(roll_m, dt)   # rad/s
            pitch_v = np.gradient(pitch_m, dt)
            yaw_v = np.gradient(yaw_m, dt)

            # Total velocity at point = translation + ω × r
            # ω = [roll_v, pitch_v, yaw_v] (angular velocity vector)
            # r = [r_x, r_y, r_z] (position from CoR)
            # ω × r gives additional velocity from rotation

            # Velocity contribution from rotation: v_rot = ω × r
            # [ω_x, ω_y, ω_z] × [r_x, r_y, r_z] =
            # [ω_y*r_z - ω_z*r_y, ω_z*r_x - ω_x*r_z, ω_x*r_y - ω_y*r_x]
            v_rot_x = pitch_v * r_vec[2] - yaw_v * r_vec[1]
            v_rot_y = yaw_v * r_vec[0] - roll_v * r_vec[2]
            v_rot_z = roll_v * r_vec[1] - pitch_v * r_vec[0]

            # Total velocity at point
            vx_total = surge_v + v_rot_x
            vy_total = sway_v + v_rot_y
            vz_total = heave_v + v_rot_z

            # Build drone-centric coordinate frame at contact point
            # Y_drone = normal (pointing away from hull)
            # X_drone = tangent (horizontal/fore-aft along hull)
            # Z_drone = tangent (vertical along hull)

            y_drone = normal_vec  # Normal to hull (away from surface)

            # Create tangent vectors orthogonal to normal
            # X_drone: horizontal tangent (cross product of normal with world Z)
            world_z = np.array([0, 0, 1])
            if abs(np.dot(y_drone, world_z)) > 0.99:
                # Normal is nearly vertical, use world X instead
                world_x = np.array([1, 0, 0])
                x_drone = np.cross(y_drone, world_x)
            else:
                x_drone = np.cross(world_z, y_drone)
            x_drone = x_drone / np.linalg.norm(x_drone)  # Normalize

            # Z_drone: vertical tangent (cross product of normal with X_drone)
            z_drone = np.cross(y_drone, x_drone)
            z_drone = z_drone / np.linalg.norm(z_drone)  # Normalize

            # Project velocities onto drone frame
            # Vx_drone = tangential (fore-aft)
            # Vy_drone = normal (into/away from hull)
            # Vz_drone = tangential (vertical)
            vx_drone = vx_total * x_drone[0] + vy_total * x_drone[1] + vz_total * x_drone[2]
            vy_drone = vx_total * y_drone[0] + vy_total * y_drone[1] + vz_total * y_drone[2]  # Normal
            vz_drone = vx_total * z_drone[0] + vy_total * z_drone[1] + vz_total * z_drone[2]

            # Total velocity magnitude
            v_total = np.sqrt(vx_drone**2 + vy_drone**2 + vz_drone**2)

            # Accelerations in drone frame
            ax_drone = np.gradient(vx_drone, dt)
            ay_drone = np.gradient(vy_drone, dt)  # Normal acceleration
            az_drone = np.gradient(vz_drone, dt)
            a_total = np.sqrt(ax_drone**2 + ay_drone**2 + az_drone**2)

            # Plot velocities in drone frame
            fig_vel = make_subplots(rows=2, cols=1, subplot_titles=(
                "Drone Frame: Tangential Velocities (Vx, Vz)", "Drone Frame: Normal Velocity (Vy)"
            ), vertical_spacing=0.18)

            fig_vel.add_trace(go.Scatter(x=t_point, y=vx_drone, name='Vx (tangent fore-aft)',
                line=dict(width=1.5, color='#1f77b4')), row=1, col=1)
            fig_vel.add_trace(go.Scatter(x=t_point, y=vz_drone, name='Vz (tangent vertical)',
                line=dict(width=1.5, color='#2ca02c')), row=1, col=1)

            fig_vel.add_trace(go.Scatter(x=t_point, y=vy_drone, name='Vy normal (+ = away)',
                line=dict(color='red', width=2)), row=2, col=1)
            fig_vel.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

            fig_vel.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
            fig_vel.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
            fig_vel.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig_vel.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))

            st.plotly_chart(fig_vel, use_container_width=True)

            # Statistics table - Drone frame
            st.subheader("Statistics (Drone Contact Frame)")
            st.caption("Vx, Vz = tangential to hull | Vy = normal to hull (+ away)")

            st.markdown("**Maximum Velocity (m/s)**")
            vel_cols = st.columns(4)
            with vel_cols[0]:
                st.metric("Vx (tangent)", f"{np.max(np.abs(vx_drone)):.3f}")
            with vel_cols[1]:
                st.metric("Vy (normal)", f"{np.max(np.abs(vy_drone)):.3f}")
            with vel_cols[2]:
                st.metric("Vz (tangent)", f"{np.max(np.abs(vz_drone)):.3f}")
            with vel_cols[3]:
                st.metric("Total", f"{np.max(v_total):.3f}")

            st.markdown("**Maximum Acceleration (m/s²)**")
            acc_cols = st.columns(4)
            with acc_cols[0]:
                st.metric("Ax (tangent)", f"{np.max(np.abs(ax_drone)):.3f}")
            with acc_cols[1]:
                st.metric("Ay (normal)", f"{np.max(np.abs(ay_drone)):.3f}")
            with acc_cols[2]:
                st.metric("Az (tangent)", f"{np.max(np.abs(az_drone)):.3f}")
            with acc_cols[3]:
                st.metric("Total", f"{np.max(a_total):.3f}")

            # Export section
            st.markdown("---")
            st.subheader("Export Data")

            # Generate 2 minutes of data at higher resolution
            t_export = np.linspace(0, 120, 2400)  # 2 min at 20 Hz
            dt_export = t_export[1] - t_export[0]

            def compute_motion_export(dof, amp, period, direction, time_arr):
                rao_amp, rao_phase = get_rao(dof, period, direction)
                omega = 2 * np.pi / period
                response_amp = amp * rao_amp
                return response_amp * np.cos(omega * time_arr + np.deg2rad(rao_phase))

            def total_motion_export(dof):
                m1 = compute_motion_export(dof, amp1, period1, direction1, t_export)
                m2 = compute_motion_export(dof, amp2, period2, direction2, t_export)
                return m1 + m2

            # Compute for export
            surge_exp = total_motion_export("Surge")
            sway_exp = total_motion_export("Sway")
            heave_exp = total_motion_export("Heave")
            roll_exp = total_motion_export("Roll")
            pitch_exp = total_motion_export("Pitch")
            yaw_exp = total_motion_export("Yaw")

            # Velocities
            surge_v_exp = np.gradient(surge_exp, dt_export)
            sway_v_exp = np.gradient(sway_exp, dt_export)
            heave_v_exp = np.gradient(heave_exp, dt_export)
            roll_v_exp = np.gradient(roll_exp, dt_export)
            pitch_v_exp = np.gradient(pitch_exp, dt_export)
            yaw_v_exp = np.gradient(yaw_exp, dt_export)

            # Rotation contribution (vessel frame)
            v_rot_x_exp = pitch_v_exp * r_vec[2] - yaw_v_exp * r_vec[1]
            v_rot_y_exp = yaw_v_exp * r_vec[0] - roll_v_exp * r_vec[2]
            v_rot_z_exp = roll_v_exp * r_vec[1] - pitch_v_exp * r_vec[0]

            vx_vessel_exp = surge_v_exp + v_rot_x_exp
            vy_vessel_exp = sway_v_exp + v_rot_y_exp
            vz_vessel_exp = heave_v_exp + v_rot_z_exp

            # Transform to drone frame
            vx_drone_exp = vx_vessel_exp * x_drone[0] + vy_vessel_exp * x_drone[1] + vz_vessel_exp * x_drone[2]
            vy_drone_exp = vx_vessel_exp * y_drone[0] + vy_vessel_exp * y_drone[1] + vz_vessel_exp * y_drone[2]
            vz_drone_exp = vx_vessel_exp * z_drone[0] + vy_vessel_exp * z_drone[1] + vz_vessel_exp * z_drone[2]

            # Accelerations in drone frame
            ax_drone_exp = np.gradient(vx_drone_exp, dt_export)
            ay_drone_exp = np.gradient(vy_drone_exp, dt_export)
            az_drone_exp = np.gradient(vz_drone_exp, dt_export)

            # Create DataFrame with drone frame data
            import pandas as pd
            df_export = pd.DataFrame({
                'time_s': t_export,
                'vx_tangent_m_s': vx_drone_exp,
                'vy_normal_m_s': vy_drone_exp,
                'vz_tangent_m_s': vz_drone_exp,
                'ax_tangent_m_s2': ax_drone_exp,
                'ay_normal_m_s2': ay_drone_exp,
                'az_tangent_m_s2': az_drone_exp
            })

            csv_data = df_export.to_csv(index=False)

            st.download_button(
                label="Download 2min data (CSV)",
                data=csv_data,
                file_name=f"hull_point_motion_x{point_x:.0f}_y{point_y:.0f}_z{point_z:.0f}.csv",
                mime="text/csv"
            )
