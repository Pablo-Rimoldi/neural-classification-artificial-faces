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
    for (sub_id, trig), group in data.groupby(['SubjectID', 'Trigger']):
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

    AF_data = data[data['TargetCODE'] == '60AF']
    AM_data = data[data['TargetCODE'] == '50AM']
    RF_data = data[data['TargetCODE'] == '80RF']
    RM_data = data[data['TargetCODE'] == '70RM']

    for i, current_data_df in enumerate([AF_data, AM_data, RF_data, RM_data]):
        combined_x_list_current = []
        EEG_list_current, PCA_list_current, y_list_current, sub_list_current, subject_sex_list_current = create_lists(current_data_df)

        for j in range(len(EEG_list_current)):
            # Create a new channel filled with the SubjectSEX value, repeated for the time dimension
            subject_sex_value = subject_sex_list_current[j]
            sex_channel_data = np.full((1, EEG_list_current[j].shape[1]), subject_sex_value)
            # Vertically stack the EEG channels and PCA components for each epoch
            # Resulting shape for each epoch will be (14 EEG_channels + 4 PCA_channels + 1 sex_channel, Time)
            combined_epoch_data = np.vstack((EEG_list_current[j], PCA_list_current[j], sex_channel_data))
            combined_x_list_current.append(combined_epoch_data)

        x_current = np.array(combined_x_list_current) # 3D Tensor (Epochs, Channels, Time)
        y_current = np.array(y_list_current) # Labels
        subjects_current = np.array(sub_list_current) # Subject ID

        if x_final is None:
            x_final = x_current
            y_final = y_current
            subjects_final = subjects_current
        else:
            x_final = np.concatenate((x_final, x_current), axis=0)
            y_final = np.concatenate((y_final, y_current), axis=0)
            subjects_final = np.concatenate((subjects_final, subjects_current), axis=0)

    return x_final, y_final, subjects_final

# ─────────────────────────────────────────────
# CHECK CLEANING
# Plots the original and cleaned data for a sample epoch to visually verify the effect of temporal cleaning.
# ─────────────────────────────────────────────

def check_cleaning(data, cols):
    # Check the effect of temporal cleaning by plotting the original and cleaned data for a single epoch
    # Get a sample epoch to demonstrate cleaning
    sample_group = None
    for (sub_id, trig), group in data.groupby(['SubjectID', 'Trigger']):
        sample_group = group.sort_values('Time_ms')
        break # Take the first epoch

    if sample_group is not None:
        # Extract raw values for the sample epoch
        raw_values_sample = sample_group[cols].values
        
        # Define times based on the length of this specific epoch
        # Use the actual time values from the epoch if available, otherwise create a linspace
        if 'Time_ms' in sample_group.columns and len(sample_group['Time_ms']) == raw_values_sample.shape[0]:
            times = sample_group['Time_ms'].values
        else:
            times = np.linspace(200, 200 + (raw_values_sample.shape[0] / S_FREQ) * 1000, raw_values_sample.shape[0])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot original data for the selected epoch
        for i in range(raw_values_sample.shape[1]):
            ax1.plot(times, raw_values_sample[:, i], color='blue', alpha=0.3)
        ax1.set_title("Original channels (sample epoch)")
        ax1.invert_yaxis()

        # Plot cleaned data for the selected epoch
        cleaned_data_sample = apply_temporal_cleaning(raw_values_sample)
        for i in range(cleaned_data_sample.shape[1]):
            ax2.plot(times, cleaned_data_sample[:, i], color='red', alpha=0.3)
        ax2.set_title("Cleaned channels (sample epoch)")
        ax2.invert_yaxis()

        plt.show()
    else:
        print("No valid epoch found to plot.")

# ─────────────────────────────────────────────
# CHECK EEG-PCA RELATION
# Plots the average of all epochs for a specific class to visually verify the relationship between cleaned EEG channels and PCA components.
# ─────────────────────────────────────────────

def check_EEG_PCA_relation(x, y):
    # Mean standard deviation for EEG and for PCA (to check if the cleaning worked and reduced the noise in the data)
    print("Mean STD EEG:", np.std(x[:, :14, :]))
    print("Mean STD PCA:", np.std(x[:, 14:18, :]))

    # Check the cleaned data by plotting the average of all epochs for a specific class (e.g., TargetNATURE='70RM'))
    times = np.linspace(200, 600, x.shape[2])
    avg_all = np.mean(x[y == '70RM'], axis=0)

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

    # Check cleaning effect
    check_cleaning(preprocessed_data, EEG_CHANNELS) # Check cleaning on EEG channels
    check_cleaning(preprocessed_data, PCA_COLUMNS) # Check cleaning on PCA components

    # Check EEG-PCA relation
    check_EEG_PCA_relation(x, y)

    # Check the final tensor
    check_final_tensor(x, y)
    
    # Save the final tensor
    save_tensor(x, y, subjects)

if __name__ == '__main__':
    main()