import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dot, Softmax, Multiply, Concatenate, Activation
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import griddata
import os
import time
import tempfile

# --- 1. GLOBAL SETTINGS & STYLING ---
st.set_page_config(page_title="Material Analysis Tool", layout="wide")

# Set Matplotlib styles to match the original "Times New Roman" look
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Compatibility for integration
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# --- 2. LOGIC FUNCTIONS (Core Physics & ML) ---

def Transform_ann_batch(time_arr, temp_arr, model):
    """
    Performs the complex transformation from dynamic properties (ANN predictions)
    to static elastic modulus using numerical integration.
    """
    # Constants from original script
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

    # Avoid division by zero if time is 0 (though unlikely in this physics context)
    # Adding a tiny epsilon just in case
    safe_time = time_arr + 1e-15
    
    w1 = w1_base[None, :] / safe_time[:, None]
    w2 = w2_base[None, :] / safe_time[:, None]
    w3 = w3_base[None, :] / safe_time[:, None]

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

    # We need to scale this data using the SAME scaler used for training.
    # In the original script, a new scaler was fit on the batch, which is technically 
    # a slight deviation from standard ML practice (usually you transform with the training scaler),
    # but to maintain 1:1 fidelity with the original logic, we fit a new scaler here.
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
    time_grid = time_min + fractions * (time_max - time_min)
    time_grid = np.tile(time_grid, (T, 1, 1))
    
    temp_grid = temps[:, None, None] * np.ones_like(time_grid)
    
    time_flat = time_grid.ravel()
    temp_flat = temp_grid.ravel()
    
    # Run the batch transformation
    E_flat = Transform_ann_batch(time_flat, temp_flat, model)
    E_tensor = E_flat.reshape(T, R, S)
    
    Stress = E_tensor * rates[None, :, None]
    
    integral = np.empty((T, R))
    for ti in range(T):
        integral[ti, :] = simps(Stress[ti, :, :], x=time_grid[ti, :, :], axis=1)
        
    final_cumulative = integral / strain_max
    
    T_col = np.repeat(temps, R)
    rate_col = np.tile(rates, T)
    emod_col = final_cumulative.ravel()

    results_df = pd.DataFrame({
        'Strain Rate (1/s)': rate_col,
        'Elastic Modulus (MPa)': emod_col,
        'Temperature (°C)': T_col
    })
    return results_df

# --- 3. CUSTOM CALLBACK FOR STREAMLIT ---
class StreamlitCallback(Callback):
    def __init__(self, epochs, progress_bar, status_text):
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            p = (epoch + 1) / self.epochs
            loss = logs.get('loss', 0)
            self.progress_bar.progress(p)
            self.status_text.text(f"Training Progress: {int(p*100)}% | Loss: {loss:.4f}")

# --- 4. MAIN APPLICATION ---

def main():
    st.title("Material Analysis Tool")
    
    # --- Session State Management ---
    if 'shared_df' not in st.session_state:
        st.session_state.shared_df = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'train_results' not in st.session_state:
        st.session_state.train_results = None
    if 'pred_results' not in st.session_state:
        st.session_state.pred_results = None

    # Tabs
    tab1, tab2 = st.tabs(["1. Visualization", "2. Model Training & Prediction"])

    # ==========================================
    # TAB 1: VISUALIZATION
    # ==========================================
    with tab1:
        st.header("Data Visualization")
        
        uploaded_file = st.file_uploader("Load Dataset (CSV)", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load Data
                df = pd.read_csv(uploaded_file)
                st.session_state.shared_df = df
                st.success(f"Loaded: {uploaded_file.name}")
                
                # Check columns
                required_cols = ['Temperature', 'Storage Modulus', 'Frequency']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {required_cols}")
                else:
                    # Plotting
                    temps = sorted(df['Temperature'].unique())
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    for t in temps:
                        subset = df[df['Temperature'] == t].sort_values(by='Frequency')
                        ax.plot(subset['Frequency'], subset['Storage Modulus'], label=f"{t} °C")
                    
                    ax.set_xscale('log')
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Storage Modulus (MPa)')
                    ax.legend(title='Temperature', bbox_to_anchor=(1.02, 1), loc='upper left')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Save Plot functionality (Streamlit handles download)
                    fn = "visualization_plot.png"
                    fig.savefig(fn)
                    with open(fn, "rb") as img:
                        st.download_button(
                            label="Download Plot as PNG",
                            data=img,
                            file_name=fn,
                            mime="image/png"
                        )
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ==========================================
    # TAB 2: ANALYSIS
    # ==========================================
    with tab2:
        st.header("Analysis Configuration")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            epochs = st.number_input("Epochs", min_value=10, value=500, step=10)
        with col2:
            batch_size = st.number_input("Batch Size", min_value=1, value=1, step=1)
        
        start_btn = st.button("Start Analysis", type="primary")
        
        status_placeholder = st.empty()
        progress_bar = st.empty()

        if start_btn:
            if st.session_state.shared_df is None:
                st.warning("Please upload data in Tab 1 first.")
            else:
                # --- STEP 1: TRAINING ---
                start_time = time.time()
                status_placeholder.text("Learning the trend... (Initializing)")
                progress_bar.progress(0)
                
                try:
                    # Data Prep
                    data = st.session_state.shared_df.copy()
                    X = data[['Temperature', 'Frequency']].values
                    y = data['Storage Modulus'].values
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Model Architecture (Exact Copy)
                    inputs = Input(shape=(2,))
                    q = Dense(32, activation='relu')(inputs)
                    k = Dense(32, activation='relu')(inputs)
                    v = Dense(32, activation='relu')(inputs)
                    attn = Dot(axes=-1)([q, k])
                    attn_w = Activation('sigmoid')(attn)
                    context = Multiply()([attn_w, v])
                    concat = Concatenate()([context, inputs])
                    hidden = Dense(32, activation='relu')(concat)
                    output = Dense(1, activation='linear')(hidden)
                    
                    model = Model(inputs, output)
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # Training
                    cb = StreamlitCallback(epochs, progress_bar, status_placeholder)
                    
                    history = model.fit(
                        X_scaled, y, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=0,
                        callbacks=[cb]
                    )
                    
                    st.session_state.trained_model = model
                    
                    # Training Eval
                    y_pred = model.predict(X_scaled, verbose=0)
                    r2 = r2_score(y, y_pred)
                    mse = mean_squared_error(y, y_pred)
                    st.session_state.train_results = {'y_true': y, 'y_pred': y_pred, 'r2': r2, 'mse': mse}
                    
                    # --- STEP 2: PREDICTION (The "Thread 2" part) ---
                    status_placeholder.text("Training Complete. Transforming to Elastic Modulus...")
                    progress_bar.progress(100) # Training done
                    
                    with st.spinner("Calculating Modulus Matrix... (This involves heavy integration)"):
                        # Determine ranges
                        temps = sorted(st.session_state.shared_df['Temperature'].unique())
                        t_start, t_end = int(min(temps)), int(max(temps))
                        t_step = int(temps[1] - temps[0]) if len(temps) > 1 else 10
                        strain_rates = [1e-2, 1e-3, 1e-4, 1e-5]

                        res_df = compute_modulus_matrix(
                            model, t_start, t_end, t_step, strain_rates
                        )
                        st.session_state.pred_results = res_df
                    
                    total_time = time.time() - start_time
                    status_placeholder.success(f"Analysis Finished Successfully. (Total Time: {total_time:.1f} s)")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # --- RESULTS DISPLAY ---
        if st.session_state.train_results is not None and st.session_state.pred_results is not None:
            st.divider()
            
            # Downloads Section
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                # Save Model
                # We save to a temp file then read bytes for download
                if st.session_state.trained_model:
                    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                        st.session_state.trained_model.save(tmp.name)
                        tmp.seek(0)
                        model_bytes = tmp.read()
                        st.download_button("Download Trained Model (.h5)", data=model_bytes, file_name="model.h5")
            
            with d_col2:
                # Save Results CSV
                csv = st.session_state.pred_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results (.csv)", data=csv, file_name="results.csv", mime="text/csv")

            # Result Tabs
            rtab1, rtab2, rtab3, rtab4 = st.tabs([
                "1. Accuracy", "2. Strain Rate vs Modulus", "3. Temp vs Modulus", "4. 3D Surface"
            ])

            # 1. Training Accuracy
            with rtab1:
                res = st.session_state.train_results
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(res['y_true'], res['y_pred'], alpha=0.5, label='Data Points')
                
                min_val = min(np.min(res['y_true']), np.min(res['y_pred']))
                max_val = max(np.max(res['y_true']), np.max(res['y_pred']))
                ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-.', label='45° Line')
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                
                textstr = f'R² = {res["r2"]:.4f}\nMSE = {res["mse"]:.4f}'
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                ax.legend()
                st.pyplot(fig)

            # 2. Strain Rate vs Modulus
            with rtab2:
                res_df = st.session_state.pred_results
                temps_res = sorted(res_df['Temperature (°C)'].unique())
                fig, ax = plt.subplots(figsize=(6, 5))
                for t in temps_res:
                    subset = res_df[res_df['Temperature (°C)'] == t]
                    ax.plot(subset['Strain Rate (1/s)'], subset['Elastic Modulus (MPa)'], label=str(t))
                
                ax.set_xscale('log')
                ax.set_xlabel(r'Strain Rate (s$^{-1}$)')
                ax.set_ylabel('Elastic Modulus (MPa)')
                ax.legend(title='Temperature (°C)', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)

            # 3. Temp vs Modulus
            with rtab3:
                res_df = st.session_state.pred_results
                fig, ax = plt.subplots(figsize=(6, 5))
                unique_rates = sorted(res_df['Strain Rate (1/s)'].unique(), reverse=True)
                for rate in unique_rates:
                    subset = res_df[res_df['Strain Rate (1/s)'] == rate].sort_values(by='Temperature (°C)')
                    exponent = int(np.round(np.log10(rate)))
                    ax.plot(subset['Temperature (°C)'], subset['Elastic Modulus (MPa)'], label=fr"$10^{{{exponent}}}$")
                
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel('Elastic Modulus (MPa)')
                ax.legend(title=r'Strain Rate (s$^{-1}$)', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)

            # 4. 3D Surface
            with rtab4:
                res_df = st.session_state.pred_results
                pivot_df = res_df.pivot(index='Strain Rate (1/s)', columns='Temperature (°C)', values='Elastic Modulus (MPa)')
                
                X_temps = pivot_df.columns.values
                Y_rates = pivot_df.index.values
                Y_rates_log = np.log10(Y_rates)
                Z_data = pivot_df.values
                
                # Grid for interpolation
                X_data_grid, Y_data_grid = np.meshgrid(X_temps, Y_rates_log)
                points = np.column_stack([X_data_grid.ravel(), Y_data_grid.ravel()])
                values = Z_data.ravel()
                
                x_fine = np.linspace(X_temps.min(), X_temps.max(), 100)
                y_fine = np.linspace(Y_rates_log.min(), Y_rates_log.max(), 100)
                X_grid_fine, Y_grid_fine = np.meshgrid(x_fine, y_fine)
                
                Z_grid_fine = griddata(points, values, (X_grid_fine, Y_grid_fine), method='cubic')
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                surf = ax.plot_surface(X_grid_fine, Y_grid_fine, Z_grid_fine, cmap='rainbow', edgecolor='none', alpha=0.8)
                
                ax.set_xlabel('Temperature (°C)', labelpad=10)
                ax.set_ylabel(r'Strain Rate (s$^{-1}$)', labelpad=10)
                ax.set_yticks(Y_rates_log)
                ax.set_yticklabels([f"$10^{{{int(y)}}}$" for y in Y_rates_log])
                ax.set_zlabel('Elastic Modulus (MPa)', labelpad=10)
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
