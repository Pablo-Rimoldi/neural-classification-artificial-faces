import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd


EEG_CHANNELS = [
    'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8',
    'P3', 'P4', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h'
]
METADATA_COLUMNS = ['Trigger', 'SubjectID', 'SubjectSEX', 'TargetCODE', 'TargetNATURE', 'Time_ms']
PCA_COLUMNS = ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']

# Sampling frequency in Hz
S_FREQ = 512 

# Statistical threshold for PCA components
Z_THRESHOLD = 4.0

# ─────────────────────────────────────────────
# CHECK AMPLITUDE DISTRIBUTION
# Plots a histogram of the maximum absolute amplitudes across EEG channels to identify potential artifacts.
# ─────────────────────────────────────────────
def check_amplitude_distribution(data):
    # Check if there are some artifacts in the amplitudes distribution in order to apply some type of rejection treshold
    amplitudes = data[EEG_CHANNELS].abs().max(axis=1)
    plt.hist(amplitudes, bins=100, range=(0, 100))
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('Distribution of the max amplitudes')
    plt.show()

# ─────────────────────────────────────────────
# TEMPORAL CLEANING
# Applies temporal filtering and detrending to the epoch data.
# ─────────────────────────────────────────────

def apply_temporal_cleaning(epoch_data):
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
# CREATE LISTS
# Iterates through the dataset grouped by SubjectID and Trigger, applies temporal cleaning to both EEG and PCA data, and collects the cleaned data along with labels and metadata into separate lists.
# ─────────────────────────────────────────────

def create_lists(data):
    EEG_list = [] # EEG data
    PCA_list = [] # PCA data
    y_list = [] # Labels (TargetNATURE)
    sub_list = [] # Subject ID
    subject_sex_list = [] # Subject SEX

    # Define a target number of samples for each epoch
    # The Time_ms column indicates a range from ~200ms to ~600ms, which is 400ms duration.
    # At 512Hz, 400ms = 0.4 * 512 = 204.8 samples. We'll aim for 205 samples.
    target_epoch_samples = 205

    # Group by epochs
    for (sub_id, trig), group in data.groupby(['SubjectID', 'TargetCODE']):
        # Order by the time dimension
        group = group.sort_values('Time_ms')

        # Ensure enough samples for filtering and baseline for both EEG and PCA components.
        # For a 4th order filter, filtfilt requires at least 16 samples.
        # Baseline correction uses 25 samples
        min_samples_required = max(int(0.05 * S_FREQ), 16)
        if len(group) < min_samples_required:
            continue

        # 1. Clean EEG data
        eeg_raw_values = group[EEG_CHANNELS].values
        cleaned_eeg_data = apply_temporal_cleaning(eeg_raw_values)

        # 2. Clean PCA components
        PCA_raw_values = group[PCA_COLUMNS].values
        cleaned_pca_data = apply_temporal_cleaning(PCA_raw_values)

        # Enforce consistent epoch length for cleaned EEG data
        if cleaned_eeg_data.shape[0] > target_epoch_samples:
            cleaned_eeg_data = cleaned_eeg_data[:target_epoch_samples, :]
        elif cleaned_eeg_data.shape[0] < target_epoch_samples:
            continue # Skip this epoch if EEG data is too short after cleaning/truncation

        # Enforce consistent epoch length for cleaned PCA data
        if cleaned_pca_data.shape[0] > target_epoch_samples:
            cleaned_pca_data = cleaned_pca_data[:target_epoch_samples, :]
        elif cleaned_pca_data.shape[0] < target_epoch_samples:
            continue # Skip this epoch if PCA data is too short after cleaning/truncation

        # Enforce the exclusion of too extreme data based on cleaned PCA components
        if np.max(np.abs(cleaned_pca_data)) < (np.std(cleaned_pca_data) * Z_THRESHOLD):
            # List the cleaned data
            EEG_list.append(cleaned_eeg_data.T)
            PCA_list.append(cleaned_pca_data.T)
            y_list.append(group['TargetCODE'].iloc[0])
            sub_list.append(sub_id)
            subject_sex_list.append(group['SubjectSEX'].iloc[0])
        
    return EEG_list, PCA_list, y_list, sub_list, subject_sex_list

# ─────────────────────────────────────────────
# CREATE TENSOR
# Combines the cleaned EEG and PCA data along with the SubjectSEX metadata into a single 3D tensor for each epoch, and concatenates all epochs into final tensors for features, labels, and subject IDs.
# ─────────────────────────────────────────────

def create_tensor(data):
    x_final = None
    y_final = None
    subjects_final = None

    combined_X_list = []

    EEG_list, PCA_list, y_list, sub_list, subject_sex_list = create_lists(data)

    for j in range(len(EEG_list)):
        # Vertically stack the EEG channels and PCA components for each epoch
        # Resulting shape for each epoch will be (14 EEG_channels + 4 PCA_channels, Time)
        # Add an extra channel to track the face sex (FaceSEX: 1.0 for female face, 0.0 for male face)
        face_sex_val = 1.0 if str(y_list[j]).endswith('F') else 0.0
        new_channel_data = np.full((1, EEG_list[j].shape[1]), face_sex_val)
        combined_epoch_data = np.vstack((EEG_list[j], PCA_list[j], new_channel_data))
        combined_X_list.append(combined_epoch_data)

    x_final = np.array(combined_X_list) # 3D Tensor (Epochs, Channels, Time)
    y_final = np.array(y_list) # Labels
    subjects_final = np.array(sub_list) # Subject ID

    return x_final, y_final, subject_final

# ─────────────────────────────────────────────
# CHECK FINAL TENSOR
# Prints the shape of the final tensor for features and labels to verify that they are correctly structured for model training.
# ─────────────────────────────────────────────
def check_final_tensor(x, y):
    # Check the shape of the final tensor
    print("Final tensor shape (Epochs, Channels, Time):", x.shape)
    print("Labels shape:", y.shape)

# ─────────────────────────────────────────────
# SAVE TENSOR
# Saves the final tensors for features, labels, and subject IDs into a .npz file for later use in model training.
# ─────────────────────────────────────────────
def save_tensor(x, y, subjects):
    np.savez('data/final_tensor.npz', x=x, y=y, subjects=subjects)

def main():
    # Load preprocessed data
    preprocessed_data = pd.read_parquet('data/dataset_eeg_preprocessed.parquet')

    # Check amplitude distribution to identify potential artifacts
    check_amplitude_distribution(preprocessed_data)

    # Create tensor
    x, y, subjects = create_tensor(preprocessed_data)

    # Check the final tensor
    check_final_tensor(x, y)
    
    # Save the final tensor
    save_tensor(x, y, subjects)

if __name__ == '__main__':
    main()