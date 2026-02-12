import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.integrate import simpson as simps
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dot, Softmax, Multiply, Concatenate, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import time
import io

# --- 1. GLOBAL SETTINGS ---
plt.rcParams['font.family'] = 'DejaVu Sans' # Times New Roman often missing on cloud servers
plt.rcParams['font.size'] = 10

# --- 2. CORE LOGIC (UNCHANGED MATH) ---

def Transform_ann_batch(time_arr, temp_arr, model):
    N1, N2, N3 = 90, 70, 26
    cycle = 10  
    
    n1 = int(cycle * 0.1 * N1 + 1)
    n2 = int(cycle * 0.3 * N2 + 1)
    n3 = int(cycle * 0.6 * N3 + 1)

    w1_base = np.linspace(1e-6, cycle * 0.1 * 2 * np.pi, n1)
    w2_base = np.linspace(cycle * 0.1 * 2 * np.pi, cycle * 0.4 * 2 * np.pi, n2)[1:]
    w3_base = np.linspace(cycle * 0.4 * 2 * np.pi, cycle * 2 * np.pi, n3)[1:]

    n1, n2, n3 = w1_base.shape[0], w2_base.shape[0], w3_base.shape[0]
    M = time_arr.size

    w1 = w1_base[None, :] / time_arr[:, None]
    w2 = w2_base[None, :] / time_arr[:, None]
    w3 = w3_base[None, :] / time_arr[:, None]

    freq_log1 = np.log10(w1 / (2 * np.pi)).ravel()
    freq_log2 = np.log10(w2 / (2 * np.pi)).ravel()
    freq_log3 = np.log10(w3 / (2 * np.pi)).ravel()

    temp_rep1 = np.repeat(temp_arr, n1)
    temp_rep2 = np.repeat(temp_arr, n2)
    temp_rep3 = np.repeat(temp_arr, n3)

    X_big = np.column_stack([
        np.concatenate([temp_rep1, temp_rep2, temp_rep3]),
        np.concatenate([freq_log1, freq_log2, freq_log3])
    ])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_big)
    
    preds_big = model.predict(X_train_scaled, verbose=0)[:, 0]

    idx1_end = M * n1
    idx2_end = idx1_end + M * n2

    pred1 = preds_big[:idx1_end].reshape(M, n1)
    pred2 = preds_big[idx1_end:idx2_end].reshape(M, n2)
    pred3 = preds_big[idx2_end:].reshape(M, n3)

    factor = 2.0 / np.pi
    y1 = factor * (pred1 / w1) * np.sin(w1 * time_arr[:, None])
    y2 = factor * (pred2 / w2) * np.sin(w2 * time_arr[:, None])
    y3 = factor * (pred3 / w3) * np.sin(w3 * time_arr[:, None])

    E1 = np.trapz(y1, w1, axis=1)
    E2 = np.trapz(y2, w2, axis=1)
    E3 = np.trapz(y3, w3, axis=1)

    return E1 + E2 + E3

def compute_modulus_matrix(model, temp_lower, temp_upper, temp_step, strain_rates):
    temps = np.arange(temp_lower, temp_upper + 1, temp_step)
    rates = np.array(strain_rates)
    strain_min, strain_max, num_steps = 1e-25, 0.0025, 100
    
    T, R, S = temps.size, rates.size, num_steps
    time_min = (strain_min / rates)[None, :, None]
    time_max = (strain_max / rates)[None, :, None]
    fractions = np.linspace(0.0, 1.0, S)[None, None, :]
    time_grid = np.tile(time_min + fractions * (time_max - time_min), (T, 1, 1))
    temp_grid = temps[:, None, None] * np.ones_like(time_grid)
    
    E_flat = Transform_ann_batch(time_grid.ravel(), temp_grid.ravel(), model)
    E_tensor = E_flat.reshape(T, R, S)
    Stress = E_tensor * rates[None, :, None]
    
    integral = np.empty((T, R))
    for ti in range(T):
        integral[ti, :] = simps(Stress[ti, :, :], x=time_grid[ti, :, :], axis=1)
        
    final_cumulative = integral / strain_max
    
    results_df = pd.DataFrame({
        'Strain Rate (1/s)': np.tile(rates, T),
        'Elastic Modulus (MPa)': final_cumulative.ravel(),
        'Temperature (°C)': np.repeat(temps, R)
    })
    return results_df

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="Material Analysis Tool", layout="wide")
st.title("🧬 ML to Elastic Modulus Analysis")

# Sidebar for inputs
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a DMA CSV file", type="csv")
    
    st.header("2. Hyperparameters")
    epochs = st.number_input("Epochs", min_value=10, max_value=5000, value=500)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=1)
    
    run_analysis = st.button("🚀 Start Full Analysis")

# Main Page Tabs
tab_vis, tab_results = st.tabs(["📊 Data Visualization", "📉 Analysis Results"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Visualization Tab
    with tab_vis:
        st.subheader("Raw Storage Modulus Data")
        fig_vis, ax = plt.subplots()
        temps = sorted(df['Temperature'].unique())
        for t in temps:
            subset = df[df['Temperature'] == t].sort_values(by='Frequency')
            ax.plot(subset['Frequency'], subset['Storage Modulus'], label=f"{t} °C")
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Storage Modulus (MPa)')
        ax.legend(title='Temperature')
        st.pyplot(fig_vis)

    # Analysis Logic
    if run_analysis:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare Data
        X = df[['Temperature', 'Frequency']].values
        y = df['Storage Modulus'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build Model
        inputs = Input(shape=(2,))
        q, k, v = [Dense(32, activation='relu')(inputs) for _ in range(3)]
        attn = Activation('sigmoid')(Dot(axes=-1)([q, k]))
        concat = Concatenate()([Multiply()([attn, v]), inputs])
        output = Dense(1, activation='linear')(Dense(32, activation='relu')(concat))
        model = Model(inputs, output)
        model.compile(optimizer='adam', loss='mse')

        # Train
        status_text.text("Training Model...")
        model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=0)
        progress_bar.progress(50)
        
        # Predict
        status_text.text("Calculating Elastic Modulus...")
        t_start, t_end = int(df['Temperature'].min()), int(df['Temperature'].max())
        t_step = 10
        strain_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        
        res_df = compute_modulus_matrix(model, t_start, t_end, t_step, strain_rates)
        progress_bar.progress(100)
        status_text.text("Analysis Complete!")

        # Display Results
        with tab_results:
            col1, col2 = st.columns(2)
            
            # 1. Strain Rate Plot
            with col1:
                st.write("### Strain Rate vs Modulus")
                fig1, ax1 = plt.subplots()
                for t in sorted(res_df['Temperature (°C)'].unique()):
                    sub = res_df[res_df['Temperature (°C)'] == t]
                    ax1.plot(sub['Strain Rate (1/s)'], sub['Elastic Modulus (MPa)'], label=f"{t}°C")
                ax1.set_xscale('log')
                ax1.set_xlabel('Strain Rate (1/s)')
                ax1.set_ylabel('Modulus (MPa)')
                st.pyplot(fig1)

            # 2. 3D Plot
            with col2:
                st.write("### 3D Modulus Surface")
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, projection='3d')
                pivot = res_df.pivot(index='Strain Rate (1/s)', columns='Temperature (°C)', values='Elastic Modulus (MPa)')
                X_m, Y_m = np.meshgrid(pivot.columns, np.log10(pivot.index))
                surf = ax3.plot_surface(X_m, Y_m, pivot.values, cmap='rainbow')
                ax3.set_xlabel('Temp')
                ax3.set_ylabel('Log Strain Rate')
                st.pyplot(fig3)

            # Download CSV
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "results.csv", "text/csv")
else:
    st.info("Please upload a CSV file in the sidebar to begin.")