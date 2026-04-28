import torch
import os
import sys
import argparse
import time
import numpy as np
import openpyxl
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ======================== Visualization Functions ========================

def draw_tsne(model, dataloader, device, subject_id, save_dir, split_name="test"):
    """Draw t-SNE visualization using features before the final FC layer."""
    model.to(device)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Adjust data shape if necessary
            if X_batch.ndim == 4:
                B, W, C, L = X_batch.shape
                X_batch = X_batch.permute(0, 2, 1, 3).reshape(B, C, W * L)

            X_batch = X_batch.to(device)
            # Model returns four values; the second one is features before FC layer
            logits, features_before_fc, _, _ = model(X_batch)
            features.append(features_before_fc.cpu().numpy())
            labels.append(y_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot and save PNG
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=f'Class {label}', alpha=0.6)
    plt.legend()
    plt.title(f'Sub{subject_id:02d} t-SNE ({split_name})')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'tsne_{split_name}_Sub{subject_id:02d}.png')
    plt.savefig(png_path)
    plt.close()

    # Save as CSV for later custom plotting
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': labels
    })
    csv_path = os.path.join(save_dir, f'tsne_{split_name}_Sub{subject_id:02d}.csv')
    df.to_csv(csv_path, index=False)


def plot_confusion_matrix(model, dataloader, device, subject_id, save_dir):
    """Compute and save confusion matrix (PNG + CSV)."""
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            if X_batch.ndim == 4:
                B, W, C, L = X_batch.shape
                X_batch = X_batch.permute(0, 2, 1, 3).reshape(B, C, W * L)

            X_batch = X_batch.to(device)
            logits, _, _, _ = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs(save_dir, exist_ok=True)

    # Save PNG
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(cm.shape[0])],
                yticklabels=[f"C{i}" for i in range(cm.shape[0])])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Sub{subject_id:02d} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_Sub{subject_id:02d}.png'))
    plt.close()

    # Save CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(save_dir, f'confusion_matrix_Sub{subject_id:02d}.csv'),
                 index=False)


# ======================== Connectivity Computation ========================

from scipy.signal import hilbert


def compute_plv(data: np.ndarray) -> np.ndarray:
    """Compute Phase Locking Value (PLV) adjacency matrix."""
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (trials, channels, time), got {data.shape}")

    # Auto transpose if the time dimension is not the last one
    if data.shape[1] > data.shape[2]:
        data = np.transpose(data, (0, 2, 1))

    n_trials, n_channels, n_times = data.shape
    analytic_signal = hilbert(data, axis=-1)
    phase_data = np.angle(analytic_signal)

    plv_matrix = np.zeros((n_channels, n_channels), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(n_channels, k=1)

    for i, j in zip(idx_i, idx_j):
        phase_diff = phase_data[:, i, :] - phase_data[:, j, :]
        plv_val = np.abs(np.mean(np.exp(1j * phase_diff)))
        plv_matrix[i, j] = plv_val
        plv_matrix[j, i] = plv_val

    np.fill_diagonal(plv_matrix, 1.0)
    return plv_matrix


from scipy.signal import coherence, get_window


def compute_coh(
        data: np.ndarray,
        fs: float,
        band: tuple | None = None,  # None = full band, (8,12) = mu, (13,30) = beta
        nperseg: int | None = None,
        noverlap: int | None = None,
        window: str = 'hann',
        detrend: str | bool = False,
) -> np.ndarray:
    """
    Compute Coherence (COH) adjacency matrix with optional frequency band filtering.

    Parameters:
        data: (trials, channels, time)
        fs: sampling rate (Hz)
        band: (low, high) frequency band. If None, use full band.
    Returns:
        coh_matrix: (channels, channels) symmetric matrix with diagonal = 1
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (trials, channels, time), got {data.shape}")

    # Transpose to (trials, channels, time) if needed
    if data.shape[1] > data.shape[2]:
        data = np.transpose(data, (0, 2, 1))

    n_trials, n_channels, n_times = data.shape
    if nperseg is None:
        nperseg = min(256, n_times)
    if noverlap is None:
        noverlap = nperseg // 2
    win = get_window(window, nperseg)

    coh_matrix = np.zeros((n_channels, n_channels), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(n_channels, k=1)

    for i, j in zip(idx_i, idx_j):
        vals = []
        for t in range(n_trials):
            f, Cxy = coherence(
                data[t, i, :], data[t, j, :],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window=win,
                detrend=detrend
            )
            if band is None:
                vals.append(np.mean(Cxy))  # full frequency band
            else:
                f_low, f_high = band
                sel = (f >= f_low) & (f <= f_high)
                vals.append(np.mean(Cxy[sel]) if np.any(sel) else 0.0)

        coh_val = float(np.mean(vals)) if vals else 0.0
        coh_matrix[i, j] = coh_val
        coh_matrix[j, i] = coh_val

    np.fill_diagonal(coh_matrix, 1.0)
    np.clip(coh_matrix, 0.0, 1.0, out=coh_matrix)
    return coh_matrix


# ======================== Main Training Function ========================

def start_run(args):
    # -------------------------- Environment Setup --------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, f'information-{args.id}.txt'))
    tensorboard = SummaryWriter(args.tensorboard_path)

    start_epoch = 0

    # -------------------------- Device Setup --------------------------
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(device)

    # -------------------------- Data Loading --------------------------
    if args.data_type == 'bci2a':
        train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(args.data_path, subject_id=args.id)
    elif args.data_type == 'HGD':
        args.channel_num = 44
        args.n_class = 4
        args.out_chans = 32
        train_X, train_y, test_X, test_y = load_HGD_single_subject(args.data_path, subject_id=args.id)
    elif args.data_type == 'HandMI':
        args.channel_num = 32
        args.n_class = 2
        args.out_chans = 32
        train_X, train_y, test_X, test_y = load_HandMI_single_subject(args.data_path, subject_id=args.id)

    channel_num = args.channel_num
    slide_window_length = args.window_length
    slide_window_stride = args.window_padding

    slide_train_X, slide_train_y = sliding_window_eeg(train_X, train_y, slide_window_length, slide_window_stride)
    slide_test_X, slide_test_y = sliding_window_eeg(test_X, test_y, slide_window_length, slide_window_stride)

    slide_train_X = torch.tensor(slide_train_X, dtype=torch.float32)
    slide_test_X = torch.tensor(slide_test_X, dtype=torch.float32)
    slide_train_y = torch.tensor(slide_train_y, dtype=torch.int64)
    slide_test_y = torch.tensor(slide_test_y, dtype=torch.int64)

    print(slide_train_X.shape, slide_train_y.shape)
    print(slide_test_X.shape, slide_test_y.shape)
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    slide_window_num = slide_train_X.shape[0]

    # -------------------------- Build Adjacency Matrix --------------------------
    if args.spatial_adj_mode == 'l':
        Adj = torch.tensor(load_adj('hgd'), dtype=torch.float32)

    elif args.spatial_adj_mode == 'p':
        temp = train_X
        train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
        Adj = torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)
        print(Adj.shape)

    elif args.spatial_adj_mode == 'r':
        Adj = torch.randn(channel_num, channel_num)

    elif args.spatial_adj_mode == 'lp':
        temp = train_X
        train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
        Adj_p = torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)
        Adj_l = torch.tensor(load_adj('hgd'), dtype=torch.float32)
        Adj = torch.zeros((channel_num, channel_num), dtype=torch.float32)
        for i in range(channel_num):
            for j in range(channel_num):
                if Adj_l[i][j] == 0:
                    Adj[i][j] = Adj_p[i][j]
                else:
                    Adj[i][j] = 0.5 * Adj_l[i][j] + 0.5 * Adj_p[i][j]

    elif args.spatial_adj_mode == 'plv':
        temp = train_X.detach().cpu().numpy()
        plv_matrix = compute_plv(temp)
        Adj = torch.tensor(plv_matrix, dtype=torch.float32, device=device)
        print("PLV Adj shape:", Adj.shape, "device:", Adj.device)

    elif args.spatial_adj_mode == 'coh':
        temp = train_X.detach().cpu().numpy()
        coh_matrix = compute_coh(temp, fs=args.sampling_rate)
        Adj = torch.tensor(coh_matrix, dtype=torch.float32, device=device)
        print("COH Adj shape:", Adj.shape, "device:", Adj.device)

    elif args.spatial_adj_mode == 'lplv':
        Adj_l = torch.tensor(load_adj('hgd'), dtype=torch.float32, device=device)
        temp = train_X.detach().cpu().numpy()
        plv_matrix = compute_plv(temp)
        Adj_plv = torch.tensor(plv_matrix, dtype=torch.float32, device=device)
        Adj = torch.zeros_like(Adj_l)
        for i in range(channel_num):
            for j in range(channel_num):
                if Adj_l[i][j] == 0:
                    Adj[i][j] = Adj_plv[i][j]
                else:
                    Adj[i][j] = 0.5 * Adj_l[i][j] + 0.5 * Adj_plv[i][j]
        print("L+PLV Adj shape:", Adj.shape, "device:", Adj.device)

    elif args.spatial_adj_mode == 'lcoh':
        # 1) Layout prior
        Adj_l = torch.tensor(load_adj('hgd'), dtype=torch.float32, device=device)
        temp = train_X.detach().cpu().numpy()  # (trials, channels, time)
        fs = args.sampling_rate

        # 2) Choose frequency band mode
        mode = getattr(args, 'coh_band_mode', 'full')

        if mode == 'mu':
            coh_np = compute_coh(temp, fs=fs, band=(8, 12))
        elif mode == 'beta':
            coh_np = compute_coh(temp, fs=fs, band=(13, 30))
        elif mode == 'mubeta':
            mu_np = compute_coh(temp, fs=fs, band=(8, 12))
            beta_np = compute_coh(temp, fs=fs, band=(13, 30))
            w_mu, w_beta = 0.5, 0.5
            coh_np = (w_mu * mu_np + w_beta * beta_np) / (w_mu + w_beta)
        elif mode == 'full':
            coh_np = compute_coh(temp, fs=fs, band=None)
        else:
            raise ValueError(f"Unknown coh_band_mode: {mode}")

        Adj_coh = torch.tensor(coh_np, dtype=torch.float32, device=device)

        # 3) Blend with layout prior
        alpha = getattr(args, 'layout_alpha', 0.5)
        Adj = torch.empty_like(Adj_l)

        mask_zero = (Adj_l == 0)
        Adj[mask_zero] = Adj_coh[mask_zero]
        Adj[~mask_zero] = alpha * Adj_l[~mask_zero] + (1.0 - alpha) * Adj_coh[~mask_zero]

        print(f"L+COH({mode}) Adj shape: {Adj.shape}, device: {Adj.device}")

    else:
        raise ValueError(
            f"adj_mode only supports l, p, r, lp, plv, coh, lplv, lcoh but got {args.spatial_adj_mode}")

    # -------------------------- Model Construction --------------------------
    model_classifier = DGCNMamba(
        Adj=Adj, in_chans=channel_num, n_classes=args.n_class,
        time_window_num=slide_window_num, spatial_GCN=args.spatial_GCN,
        time_GCN=args.time_GCN, k_spatial=args.k_spatial, k_time=args.k_time,
        dropout=args.dropout, input_time_length=slide_window_length,
        out_chans=args.out_chans, kernel_size=args.kernel_size,
        slide_window=slide_window_num, sampling_rate=args.sampling_rate,
        device=args.device
    )

    print(model_classifier)
    print(f"target_id:{args.id}  spatial_GCN:{args.spatial_GCN}  time_GCN:{args.time_GCN}")

    # -------------------------- Optimizer & Loss --------------------------
    optimizer = torch.optim.AdamW(model_classifier.parameters(), lr=args.lr, weight_decay=args.w_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc, best_kappa = 0.0, 0.0
    best_f1, best_recall, best_precision = 0.0, 0.0, 0.0

    # DataLoaders
    train_loader = DataLoader(EEGDataSet(slide_train_X, slide_train_y),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(EEGDataSet(slide_test_X, slide_test_y),
                             batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    transform = build_tranforms()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2 ** -12)

    # -------------------------- Training Loop --------------------------
    acc_list = []
    train_loss_metrics = {'min_loss': [], 'mean_loss': [], 'max_loss': []}
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        node_weights, train_loss_tuple = train_one_epoch(
            epoch, train_loader, (slide_train_X, slide_train_y),
            model_classifier, args.device, optimizer, criterion,
            tensorboard, start_time, args
        )

        avg_acc, avg_loss, kappa, f1, rec, prec = evaluate_one_epoch(
            epoch, test_loader, (slide_test_X, slide_test_y),
            model_classifier, args.device, criterion, tensorboard, args, start_time
        )

        last_acc, last_kappa = avg_acc, kappa
        last_f1, last_recall, last_precision = f1, rec, prec

        train_loss_metrics['min_loss'].append(train_loss_tuple[0])
        train_loss_metrics['mean_loss'].append(train_loss_tuple[1])
        train_loss_metrics['max_loss'].append(train_loss_tuple[2])
        acc_list.append(avg_acc)

        save_checkpoints = {
            'model': model_classifier.state_dict(),
            'epoch': epoch + 1,
            'acc': avg_acc
        }

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_kappa = kappa
            best_f1, best_recall, best_precision = f1, rec, prec
            save(save_checkpoints, os.path.join(args.model_path, 'model_best.pth.tar'))

        print(f'best_acc:{best_acc:.4f} best_kappa:{best_kappa:.4f} '
              f'best_f1:{best_f1:.4f} best_recall:{best_recall:.4f} best_precision:{best_precision:.4f}')

        save(save_checkpoints, os.path.join(args.model_path, 'model_newest.pth.tar'))

    # -------------------------- Final Evaluation & Visualization --------------------------
    model_classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'model_best.pth.tar'))['model'])
    model_classifier.to(args.device)
    model_classifier.eval()

    # t-SNE visualizations
    draw_tsne(model_classifier, train_loader, args.device, args.id, args.result_path, split_name="train")
    draw_tsne(model_classifier, test_loader, args.device, args.id, args.result_path, split_name="test")

    # Confusion Matrix
    plot_confusion_matrix(model_classifier, test_loader, args.device, args.id, args.result_path)

    # Record best and last metrics
    all_kappa_record.append(best_kappa)
    all_acc_record.append(best_acc)
    all_f1_record.append(best_f1)
    all_recall_record.append(best_recall)
    all_precision_record.append(best_precision)

    last_acc_record.append(last_acc)
    last_kappa_record.append(last_kappa)
    last_f1_record.append(last_f1)
    last_recall_record.append(last_recall)
    last_precision_record.append(last_precision)

    # Save per-subject metrics
    pd.DataFrame({
        'Subject_ID': [f"{args.id:02d}"],
        'Best_Acc': [best_acc],
        'Best_Kappa': [best_kappa],
        'Best_F1': [best_f1],
        'Best_Recall': [best_recall],
        'Best_Precision': [best_precision],
        'Last_Acc': [last_acc],
        'Last_Kappa': [last_kappa],
        'Last_F1': [last_f1],
        'Last_Recall': [last_recall],
        'Last_Precision': [last_precision]
    }).to_csv(os.path.join(args.result_path, f"Sub{args.id:02d}_metrics.csv"),
              index=False, float_format='%.4f')

    # Save spatial node weights
    with open(os.path.join(args.spatial_adj_path, 'spatial_node_weights.txt'), 'a') as f:
        f.write(str(node_weights) + '\r\n')

    # Save initial adjacency matrix
    os.makedirs(args.spatial_adj_path, exist_ok=True)
    save_path = os.path.join(args.spatial_adj_path, 'initAdj.xlsx')
    adj_np = Adj.detach().cpu().numpy()
    pd.DataFrame(adj_np).to_excel(save_path, index=False, header=False)

    # Plot accuracy curve
    plt.figure()
    plt.plot(acc_list, label='test_acc')
    plt.legend()
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(args.result_path, f'test_acc_{args.id}.png'))

    # Save training loss statistics
    train_loss_df = pd.DataFrame(train_loss_metrics)
    train_loss_df.to_csv(os.path.join(args.result_path, f'train_loss_{args.id}.csv'), index=False)

    # Save test accuracy history
    pd.DataFrame(acc_list).to_csv(os.path.join(args.result_path, f'test_acc_{args.id}.csv'),
                                  header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0, help='GPU device.')
    parser.add_argument('-data_path', type=str, default='/root/autodl-tmp/datasets/2a_PMCOH',
                        help='The data path file.')
    parser.add_argument('-out_chans', type=int, default=64, help='Out channels.')
    parser.add_argument('-kernel_size', type=int, default=63, help='Kernel size.')

    parser.add_argument('-spatial_adj_mode', type=str, default='lp',
                        choices=['l', 'p', 'r', 'lp', 'plv', 'coh', 'lplv', 'lcoh'],
                        help='l: spatial layout; p: PCC; r: random; lp: PCC+layout; plv: phase locking value; coh: coherence; lplv: layout+PLV; lcoh: layout+coherence.')
    parser.add_argument('--coh_band_mode', type=str, default='beta',
                        choices=['mu', 'beta', 'mubeta', 'full'],
                        help='mu: 8–12 Hz; '
                             'beta: 13–30 Hz; '
                             'mubeta: μ+β mixed; '
                             'full: all')
    parser.add_argument('-sampling_rate', type=int, default=250, help='Data sampling rate.')
    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('-epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('-batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=2 ** -12, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('-max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping.')

    parser.add_argument('-log_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='The log files path.')
    parser.add_argument('-model_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='Path of saved model.')
    parser.add_argument('-result_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='Path of result.')
    parser.add_argument('-spatial_adj_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='Path of saved spatial_adj.')
    parser.add_argument('-time_adj_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='Path of saved time_adj.')
    parser.add_argument('-print_freq', type=int, default=1, help='The frequency to show training information.')
    parser.add_argument('-seed', type=int, default='2026', help='Random seed.')
    parser.add_argument('-father_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH')
    parser.add_argument('-tensorboard_path', type=str, default=None, help='Path of tensorboard.')
    parser.add_argument('-adj_save_path', type=str, default='/root/autodl-tmp/results/2a_PMCOH',
                        help='Path to save computed adjacency matrices.')

args_ = parser.parse_args()

    # Initialize metric recording lists
    all_kappa_record = []
    all_acc_record = []
    last_acc_record = []
    last_kappa_record = []
    all_f1_record = []
    all_recall_record = []
    all_precision_record = []
    last_f1_record = []
    last_recall_record = []
    last_precision_record = []

    for i in range(1, 10):
        args_.id = i
        start_run(args_)

    # Print summary
    fmt1 = lambda lst: [f"{x * 100:.2f}" for x in lst]
    fmt2 = lambda lst: [f"{x:.4f}" for x in lst]

    print("best_sub acc:", fmt1(all_acc_record))
    print("best_sub kappa:", fmt2(all_kappa_record))
    print("best_sub f1:", fmt2(all_f1_record))
    print("last_sub acc:", fmt1(last_acc_record))
    print("last_sub kappa:", fmt2(last_kappa_record))
    print("last_sub f1:", fmt2(last_f1_record))

    print("\n=== Best Overall Subject Statistics ===")
    print("best_avg_acc:", format(np.mean(all_acc_record), '.4f'))
    print("best_avg_kappa:", format(np.mean(all_kappa_record), '.4f'))
    print("best_std_acc:", format(np.std(all_acc_record), '.4f'))
    print("best_avg_f1:", format(np.mean(all_f1_record), '.4f'))
    print("best_avg_recall:", format(np.mean(all_recall_record), '.4f'))
    print("best_avg_precision:", format(np.mean(all_precision_record), '.4f'))

    print("\n=== Last Overall Subject Statistics ===")
    print("last_avg_acc:", format(np.mean(last_acc_record), '.4f'))
    print("last_avg_kappa:", format(np.mean(last_kappa_record), '.4f'))
    print("last_std_acc:", format(np.std(last_acc_record), '.4f'))
    print("last_avg_f1:", format(np.mean(last_f1_record), '.4f'))
    print("last_avg_recall:", format(np.mean(last_recall_record), '.4f'))
    print("last_avg_precision:", format(np.mean(last_precision_record), '.4f'))