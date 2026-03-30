import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import signal


EEG_CHANNELS = [
    'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8',
    'P3', 'P4', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h'
]
METADATA_COLUMNS = ['Trigger', 'SubjectID', 'SubjectSEX', 'TargetCODE', 'TargetNATURE', 'Time_ms']
PCA_COLUMNS = ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']

# Define constant for sampling frequency
S_FREQ = 512

# Define constant for target epoch samples
# The times range varies from ~200ms to ~600ms, which is 400ms duration.
# At 512Hz, 400ms = 0.4 * 512 = 204.8 samples. We'll aim for 205 samples.
TARGET_EPOCH_SAMPLES = 205

# Define constant for statistical threshold for PCA components
Z_THRESHOLD = 4.0

# ─────────────────────────────────────────────
# ARTIFACT CHECK
# Plot histogram of max amplitudes to visually identify potential artifacts and set rejection thresholds.
# ─────────────────────────────────────────────

def check_artifacts(data: pd.DataFrame, eeg_cols: list) -> None:
    # Check if there are some artifacts in the amplitudes distribution in order to apply some type of rejection threshold
    amplitudes = data[eeg_cols].abs().max(axis=1)
    plt.hist(amplitudes, bins=100, range=(0, 100))
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('Distribution of the max amplitudes')
    plt.show()

# ─────────────────────────────────────────────
# TEMPORAL CLEANING (Valid for both EEG and PCA features)
# Apply low-pass filtering, detrending, and debiasing to the EEG and PCA data to reduce noise and artifacts.
# ─────────────────────────────────────────────

def apply_temporal_cleaning(epoch_data: np.ndarray) -> np.ndarray:
    # Low-pass filter
    nyq = 0.5 * S_FREQ
    b, a = signal.butter(4, 40.0 / nyq, btype='low')

    filtered = signal.filtfilt(b, a, epoch_data, axis=0)
    
    # Detrending
    cleaned = signal.detrend(filtered, axis=0)

    # Debiasing
    baseline_window = cleaned[:25, :] 
    baseline_mean = np.mean(baseline_window, axis=0)

    return cleaned - baseline_mean

# ─────────────────────────────────────────────
# TENSOR CREATION
# Create a 3D tensor (Epochs, Channels, Time) by grouping the data by SubjectID and Trigger, applying temporal cleaning to both EEG and PCA features, and enforcing consistent epoch lengths.
# Also applies an exclusion criterion based on the cleaned PCA components to remove epochs with extreme values.
# ─────────────────────────────────────────────

def create_tensor(data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    eeg_cols = EEG_CHANNELS
    PCA_cols = PCA_COLUMNS

    EEG_list = [] # EEG data
    PCA_list = [] # PCA data
    y_list = [] # Labels (TargetNATURE and SubjectSEX)
    sub_list = [] # Subject ID

    # Group by epochs
    for (sub_id, trig), group in data.groupby(['SubjectID', 'Trigger']):
        # Order by the time dimension
        group = group.sort_values('Time_ms')

        # Ensure enough samples for filtering and baseline for both EEG and PCA components.
        # For a 4th order filter, filtfilt requires at least 16 samples.
        # Baseline correction uses 25 samples.
        min_samples_required = max(int(0.05 * S_FREQ), 16)
        if len(group) < min_samples_required:
            continue

        # 1. Clean EEG data
        eeg_raw_values = group[eeg_cols].values
        cleaned_eeg_data = apply_temporal_cleaning(eeg_raw_values)

        # 2. Clean PCA components
        PCA_raw_values = group[PCA_cols].values
        cleaned_pca_data = apply_temporal_cleaning(PCA_raw_values)

        # Enforce consistent epoch length for cleaned EEG data
        if cleaned_eeg_data.shape[0] > TARGET_EPOCH_SAMPLES:
            cleaned_eeg_data = cleaned_eeg_data[:TARGET_EPOCH_SAMPLES, :]
        elif cleaned_eeg_data.shape[0] < TARGET_EPOCH_SAMPLES:
            continue # Skip this epoch if EEG data is too short after cleaning/truncation

        # Enforce consistent epoch length for cleaned PCA data
        if cleaned_pca_data.shape[0] > TARGET_EPOCH_SAMPLES:
            cleaned_pca_data = cleaned_pca_data[:TARGET_EPOCH_SAMPLES, :]
        elif cleaned_pca_data.shape[0] < TARGET_EPOCH_SAMPLES:
            continue # Skip this epoch if PCA data is too short after cleaning/truncation

        # Enforce the exclusion of too extreme data based on cleaned PCA components
        if np.max(np.abs(cleaned_pca_data)) < (np.std(cleaned_pca_data) * Z_THRESHOLD):
            # List the cleaned data
            EEG_list.append(cleaned_eeg_data.T)
            PCA_list.append(cleaned_pca_data.T)
            y_list.append((group['TargetNATURE'].iloc[0], group['SubjectSEX'].iloc[0]))
            sub_list.append(sub_id)

    # Final transformation in a Tensor format (Epochs, Channels, Time)
    combined_x_list = []
    for i in range(len(EEG_list)):
        # Vertically stack the EEG channels and PCA components for each epoch
        # Resulting shape for each epoch will be (14 EEG_channels + 4 PCA_channels, Time)
        combined_epoch_data = np.vstack((EEG_list[i], PCA_list[i]))
        combined_x_list.append(combined_epoch_data)

    x = np.array(combined_x_list) # 3D Tensor (Epochs, Channels, Time)
    y = np.array(y_list) # Labels
    subjects = np.array(sub_list) # Subject ID
    return x, y, subjects

# ─────────────────────────────────────────────
# CHECK CLEANED DATA
# Plot the average of all epochs for a specific class to visually check the effect of temporal cleaning on both EEG and PCA features.
# ─────────────────────────────────────────────

def check_cleaned_data(x: np.ndarray, y: np.ndarray) -> None:
    # Mean standard deviation for EEG and for PCA (to check if the cleaning worked and reduced the noise in the data)
    print("Mean STD EEG:", np.std(x[:, :14, :]))
    print("Mean STD PCA:", np.std(x[:, 14:, :]))

    # Check the cleaned data by plotting the average of all epochs for a specific class (e.g., TargetNATURE=0 and SubjectSEX=0)
    times = np.linspace(200, 600, x.shape[2])
    avg_all = np.mean(x[np.all(y == [0, 0], axis=1)], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot EEG
    for i in range(14):
        ax1.plot(times, avg_all[i, :], color='blue', alpha=0.3)
    ax1.set_title("Cleaned EEG channels")
    ax1.invert_yaxis()

    # Plot PCA
    colors = ['red', 'green', 'orange', 'purple']
    for i in range(4):
        ax2.plot(times, avg_all[14+i, :], color=colors[i], label=f'PCA {i+1}', linewidth=2)
    ax2.set_title("Cleaned PCA components")
    ax2.invert_yaxis()

    plt.show()

# ─────────────────────────────────────────────
# SAVE TENSOR
# Save the created tensor and labels in a compressed .npz format for efficient storage and later use in model training.
# ─────────────────────────────────────────────
def save_tensor(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, filename: str) -> None:
    np.savez(filename, x=x, y=y, subjects=subjects)

# ─────────────────────────────────────────────
# MAIN FUNCTION
# Load preprocessed data, create tensor, check cleaned data, and save the tensor for later use in model training.
# ─────────────────────────────────────────────
def main():
    # Load preprocessed data
    preprocessed_data = pd.read_parquet('data/dataset_eeg_preprocessed.parquet')

    # Create tensor
    x, y, subjects = create_tensor(preprocessed_data)

    # Check cleaned data
    check_cleaned_data(x, y)

    # Save tensor
    save_tensor(x, y, subjects, 'data/final_tensor.npz')

if __name__ == '__main__':
    main()