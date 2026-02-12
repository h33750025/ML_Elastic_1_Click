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
import threading

# --- 1. GLOBAL SETTINGS & STYLING ---
st.set_page_config(page_title="Material Analysis Tool", layout="wide")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Compatibility checks
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# --- 2. LOGIC FUNCTIONS ---

def Transform_ann_batch(time_arr, temp_arr, model):
    """
    Performs the complex transformation from dynamic properties (ANN predictions)
    to static elastic modulus using numerical integration.
    """
    # Constants
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

    return pd.DataFrame({
        'Strain Rate (1/s)': rate_col,
        'Elastic Modulus (MPa)': emod_col,
        'Temperature (°C)': T_col
    })

# --- 3. CALLBACK ---
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

# --- 4. PLOTTING HELPERS ---
def plot_accuracy(y_true, y_pred, r2, mse):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.5, label='Data Points', color='blue')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-.', label='45° Line')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Training Accuracy')
    textstr = f'R² = {r2:.4f}\nMSE = {mse:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    return fig

def plot_strain_vs_elastic(df):
    temps = sorted(df['Temperature (°C)'].unique())
    # Slightly wider figure to accommodate outside legend
    fig, ax = plt.subplots(figsize=(7, 4)) 
    
    for t in temps:
        subset = df[df['Temperature (°C)'] == t]
        ax.plot(subset['Strain Rate (1/s)'], subset['Elastic Modulus (MPa)'], label=str(t), marker='o', markersize=3)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Strain Rate (s$^{-1}$)')
    ax.set_ylabel('Elastic Modulus (MPa)')
    ax.set_title("Strain Rate vs. Elastic Modulus")
    
    # 1. REMOVE GAPS
    ax.margins(x=0) 
    
    # 2. LEGEND OUTSIDE
    ax.legend(title='Temperature (°C)', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    return fig

def plot_temp_vs_elastic(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    unique_rates = sorted(df['Strain Rate (1/s)'].unique(), reverse=True)
    
    for rate in unique_rates:
        subset = df[df['Strain Rate (1/s)'] == rate].sort_values(by='Temperature (°C)')
        exponent = int(np.round(np.log10(rate)))
        ax.plot(subset['Temperature (°C)'], subset['Elastic Modulus (MPa)'], label=fr"$10^{{{exponent}}}$")
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Elastic Modulus (MPa)')
    ax.set_title("Temperature vs. Elastic Modulus")
    
    # 1. REMOVE GAPS
    ax.margins(x=0)
    
    # 2. LEGEND OUTSIDE
    ax.legend(title=r'Strain Rate (s$^{-1}$)', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    return fig

# --- 5. MAIN APP ---

def main():
    st.title("Material Analysis Tool")

    # Initialize Session State
    if 'shared_df' not in st.session_state: st.session_state.shared_df = None
    if 'trained_model' not in st.session_state: st.session_state.trained_model = None
    if 'train_results' not in st.session_state: st.session_state.train_results = None
    if 'pred_results' not in st.session_state: st.session_state.pred_results = None

    # ==========================================
    # SIDEBAR: CONFIGURATION & UPLOAD
    # ==========================================
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.shared_df = df
                st.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        st.header("2. Analysis Settings")
        epochs = st.number_input("Epochs", min_value=10, value=500, step=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=1, step=1)
        
        st.divider()
        start_btn = st.button("Start Analysis", type="primary", use_container_width=True)

    # ==========================================
    # MAIN AREA
    # ==========================================

    # 1. Visualization of Input Data
    if st.session_state.shared_df is not None:
        with st.expander("Raw Data Visualization", expanded=(st.session_state.train_results is None)):
            df = st.session_state.shared_df
            temps = sorted(df['Temperature'].unique())
            fig, ax = plt.subplots(figsize=(8, 4))
            for t in temps:
                subset = df[df['Temperature'] == t].sort_values(by='Frequency')
                ax.plot(subset['Frequency'], subset['Storage Modulus'], label=f"{t} °C")
            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Storage Modulus (MPa)')
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig)

    # 2. Results Containers
    status_container = st.empty()
    accuracy_container = st.container()
    prediction_status_container = st.empty()
    final_results_container = st.container()

    if start_btn:
        if st.session_state.shared_df is None:
            st.sidebar.error("Please upload data first.")
        else:
            # --- PHASE 1: TRAINING ---
            start_time_total = time.time()
            progress_bar = status_container.progress(0)
            status_text = status_container.empty()
            status_text.text("Learning the trend... (Initializing)")

            try:
                # Data Prep
                data = st.session_state.shared_df.copy()
                X = data[['Temperature', 'Frequency']].values
                y = data['Storage Modulus'].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Model Setup
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
                
                # Train
                cb = StreamlitCallback(epochs, progress_bar, status_text)
                model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[cb])
                
                st.session_state.trained_model = model
                
                # Calc Accuracy Stats
                y_pred = model.predict(X_scaled, verbose=0)
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                st.session_state.train_results = {'y_true': y, 'y_pred': y_pred, 'r2': r2, 'mse': mse}

                # --- SHOW ACCURACY GRAPH IMMEDIATELY ---
                with accuracy_container:
                    st.subheader("1. Training Accuracy")
                    fig_acc = plot_accuracy(y, y_pred, r2, mse)
                    st.pyplot(fig_acc)

                status_text.text("Training Complete. Starting Prediction Integration...")
                
                # --- PHASE 2: PREDICTION (Threaded Timer) ---
                temps = sorted(st.session_state.shared_df['Temperature'].unique())
                t_start, t_end = int(min(temps)), int(max(temps))
                t_step = int(temps[1] - temps[0]) if len(temps) > 1 else 10
                strain_rates = [1e-2, 1e-3, 1e-4, 1e-5]

                result_wrapper = {}
                
                def run_calc():
                    result_wrapper['df'] = compute_modulus_matrix(model, t_start, t_end, t_step, strain_rates)

                calc_thread = threading.Thread(target=run_calc)
                calc_thread.start()

                calc_start = time.time()
                while calc_thread.is_alive():
                    elapsed = int(time.time() - calc_start)
                    prediction_status_container.markdown(f"**Calculating Modulus Matrix... (This involves heavy integration) ({elapsed}s)**")
                    time.sleep(1)
                
                calc_thread.join()
                st.session_state.pred_results = result_wrapper['df']
                
                total_time = time.time() - start_time_total
                prediction_status_container.success(f"Analysis Finished Successfully. (Total Time: {total_time:.1f} s)")

            except Exception as e:
                st.error(f"Error during analysis: {e}")

    # ==========================================
    # DISPLAY RESULTS (PERSISTENT)
    # ==========================================
    if st.session_state.train_results is not None and not start_btn:
        with accuracy_container:
            st.subheader("1. Training Accuracy")
            res = st.session_state.train_results
            fig_acc = plot_accuracy(res['y_true'], res['y_pred'], res['r2'], res['mse'])
            st.pyplot(fig_acc)

    if st.session_state.pred_results is not None:
        with final_results_container:
            st.divider()
            
            # Downloads
            c1, c2 = st.columns(2)
            with c1:
                if st.session_state.trained_model:
                    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                        st.session_state.trained_model.save(tmp.name)
                        tmp.seek(0)
                        st.download_button("Download Model (.h5)", data=tmp.read(), file_name="model.h5")
            with c2:
                csv = st.session_state.pred_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results (.csv)", data=csv, file_name="results.csv", mime="text/csv")

            # Prediction Graphs
            st.subheader("2. Prediction Results")
            ptab1, ptab2, ptab3 = st.tabs(["Strain Rate vs Modulus", "Temp vs Modulus", "3D Surface"])
            
            # Tab 1: Strain Rate vs Modulus
            with ptab1:
                fig_strain = plot_strain_vs_elastic(st.session_state.pred_results)
                st.pyplot(fig_strain)
            
            # Tab 2: Temp vs Modulus
            with ptab2:
                fig_temp = plot_temp_vs_elastic(st.session_state.pred_results)
                st.pyplot(fig_temp)

            # Tab 3: 3D Surface
            with ptab3:
                res_df = st.session_state.pred_results
                pivot_df = res_df.pivot(index='Strain Rate (1/s)', columns='Temperature (°C)', values='Elastic Modulus (MPa)')
                X_temps = pivot_df.columns.values
                Y_rates = pivot_df.index.values
                Y_rates_log = np.log10(Y_rates)
                Z_data = pivot_df.values
                
                x_fine = np.linspace(X_temps.min(), X_temps.max(), 100)
                y_fine = np.linspace(Y_rates_log.min(), Y_rates_log.max(), 100)
                X_grid_fine, Y_grid_fine = np.meshgrid(x_fine, y_fine)
                
                # Griddata needs points as (N, 2) array
                X_orig, Y_orig = np.meshgrid(X_temps, Y_rates_log)
                points = np.column_stack([X_orig.ravel(), Y_orig.ravel()])
                values = Z_data.ravel()
                
                Z_grid_fine = griddata(points, values, (X_grid_fine, Y_grid_fine), method='cubic')
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X_grid_fine, Y_grid_fine, Z_grid_fine, cmap='rainbow', edgecolor='none', alpha=0.8)
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel(r'Strain Rate (s$^{-1}$)')
                ax.set_yticks(Y_rates_log)
                ax.set_yticklabels([f"$10^{{{int(y)}}}$" for y in Y_rates_log])
                ax.set_zlabel('Elastic Modulus (MPa)')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
