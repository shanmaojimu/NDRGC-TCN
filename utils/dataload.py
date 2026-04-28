import os
import argparse
import torch
from collections import OrderedDict
import logging
import pickle
import scipy.io as scio
from scipy import signal
import numpy as np
import mne
from utils.aug_utils import random_upsampling_transform, small_laplace_normalize

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def load_bciciv2a_data_single_subject(data_path, subject_id):
    """Load single-subject data for BCI Competition IV 2a."""
    subject = f"A{subject_id:02d}"

    # Load training data and labels
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy")) - 1

    # Load test data and labels
    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy")) - 1

    # Convert labels to PyTorch tensor
    test_Y = torch.tensor(test_Y, dtype=torch.int64).squeeze(-1)  # Convert (288, 1) → (288,)

    # Strategy 1: Random upsampling (currently commented out)
    aug_train_x1 = random_upsampling_transform(torch.from_numpy(train_X).to(torch.float32), ratio=0.1).numpy()
    aug_train_y1 = train_Y.copy()
    # train_X = np.concatenate((train_X, aug_train_x1))
    # train_Y = np.concatenate((train_Y, aug_train_y1))

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    # Apply Butterworth bandpass filter (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # Convert back to PyTorch tensors
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


def load_HandMI_single_subject(data_path, subject_id):
    """Load single-subject data for HandMI (HandMI-collected VR-MI) dataset."""
    subject = f"VR{subject_id:02d}"

    # Build file paths
    train_data_path = os.path.join(data_path, f"{subject}T_data.npy")
    train_label_path = os.path.join(data_path, f"{subject}T_label.npy")
    test_data_path = os.path.join(data_path, f"{subject}E_data.npy")
    test_label_path = os.path.join(data_path, f"{subject}E_label.npy")

    # Check file existence
    for path in [train_data_path, train_label_path, test_data_path, test_label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load data
    train_X = np.load(train_data_path)  # (n_trials, n_channels, n_samples)
    train_Y = np.load(train_label_path)
    test_X = np.load(test_data_path)
    test_Y = np.load(test_label_path)

    print(f"Loaded subject {subject}: Train shape {train_X.shape}, Labels shape {train_Y.shape}")
    print(f"Unique labels: {np.unique(train_Y)}")

    # Random upsampling (currently commented out)
    aug_train_x1 = random_upsampling_transform(torch.from_numpy(train_X).to(torch.float32), ratio=0.1).numpy()
    aug_train_y1 = train_Y.copy()
    # train_X = np.concatenate((train_X, aug_train_x1))
    # train_Y = np.concatenate((train_Y, aug_train_y1))

    # Convert to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).view(-1)

    # Apply Butterworth bandpass filter (0.5–40 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # Convert back to tensors
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


def load_HGD_single_subject(data_path, subject_id):
    """Load single-subject data for High Gamma Dataset (HGD)."""
    subject = f"H{subject_id:02d}"

    # Load training and test data
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy"))
    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy"))

    # Convert labels
    test_Y = torch.tensor(test_Y, dtype=torch.int64).squeeze(-1)

    # Random upsampling (currently commented out)
    aug_train_x1 = random_upsampling_transform(torch.from_numpy(train_X).to(torch.float32), ratio=0.1).numpy()
    aug_train_y1 = train_Y.copy()
    # train_X = np.concatenate((train_X, aug_train_x1))
    # train_Y = np.concatenate((train_Y, aug_train_y1))

    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    # Apply Butterworth bandpass filter (0.5–40 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


# ========================= Leave-One-Subject-Out (LOSO) =========================

def load_bciciv2a_data_cross_subject(data_path, subject_id):
    """
    Load data for Leave-One-Subject-Out (LOSO) cross-subject validation on BCI IV 2a.

    Returns:
        train_X, train_y : Training set from all other subjects (with augmentation)
        test_X,  test_y  : Test set from the left-out subject (no augmentation)
    """
    # ========================= Configurable Parameters =========================
    aug_ratio = 0.10  # Upsampling ratio
    fs = 250  # Sampling rate
    band = (0.5, 40)  # Bandpass range
    subject_ids = list(range(1, 10))  # A01 ~ A09

    # ========================= Internal Helper =========================
    def _load_raw_numpy(sid):
        """Load raw numpy data without augmentation or filtering."""
        subj = f"A{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy")) - 1
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy")) - 1

        if trY.ndim > 1:
            trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1:
            teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # Design Butterworth filter
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # ========================= Training Set (All other subjects) =========================
    train_ids = [sid for sid in subject_ids if sid != subject_id]
    train_X_list, train_y_list = [], []

    for sid in train_ids:
        trX, trY, teX, teY = _load_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # Data augmentation (upsampling)
        if aug_ratio and aug_ratio > 0:
            X_all_torch = torch.from_numpy(X_all).to(torch.float32)
            aug_X = random_upsampling_transform(X_all_torch, ratio=aug_ratio).numpy()
            aug_y = y_all.copy()
            X_all = np.concatenate([X_all, aug_X], axis=0)
            y_all = np.concatenate([y_all, aug_y], axis=0)

        # Bandpass filtering
        X_all = signal.lfilter(b, a, X_all, axis=-1)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ========================= Test Set (Left-out subject) =========================
    trX, trY, teX, teY = _load_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)

    X_te = signal.lfilter(b, a, X_te, axis=-1)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y


def load_HGD_data_cross_subject(data_path, subject_id):
    """
    Leave-One-Subject-Out (LOSO) data loader for High Gamma Dataset (HGD).
    Subjects range: 1 to 14.
    """
    aug_ratio = 0.10
    fs = 250
    band = (0.5, 40)
    time_axis = -1

    subject_ids = list(range(1, 15))
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} is not in valid range 1..14")

    train_ids = [sid for sid in subject_ids if sid != subject_id]
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    def _load_hgd_raw_numpy(sid):
        subj = f"H{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # Training set
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_hgd_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        if aug_ratio > 0:
            X_all_t = torch.from_numpy(X_all).to(torch.float32)
            aug_X = random_upsampling_transform(X_all_t, ratio=aug_ratio).numpy()
            aug_y = y_all.copy()
            X_all = np.concatenate([X_all, aug_X], axis=0)
            y_all = np.concatenate([y_all, aug_y], axis=0)

        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # Test set
    trX, trY, teX, teY = _load_hgd_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y


def load_HandMI_data_cross_subject(data_path, subject_id):
    """
    Leave-One-Subject-Out (LOSO) data loader for HandMI (VR-MI) dataset.
    Subjects range: 1 to 20.
    """
    aug_ratio = 0.10
    fs = 250
    band = (0.5, 40)
    time_axis = -1

    subject_ids = list(range(1, 21))
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} is not in valid range 1..20")

    train_ids = [sid for sid in subject_ids if sid != subject_id]
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    def _load_vr_raw_numpy(sid):
        subj = f"VR{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # Training set
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_vr_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        if aug_ratio > 0:
            X_all_t = torch.from_numpy(X_all).to(torch.float32)
            aug_X = random_upsampling_transform(X_all_t, ratio=aug_ratio).numpy()
            aug_y = y_all.copy()
            X_all = np.concatenate([X_all, aug_X], axis=0)
            y_all = np.concatenate([y_all, aug_y], axis=0)

        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # Test set
    trX, trY, teX, teY = _load_vr_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y