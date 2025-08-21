import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ===== Constants: Adjust direction encoding here if different =====
CLIENT_TO_SERVER = 1   # Client -> Server (query direction)
SERVER_TO_CLIENT = -1  # Server -> Client (response direction)
QUERY_LEN_RANGE = (233, 350)    # Closed interval [233, 350]
RESP_LEN_RANGE  = (555, 777)   # Closed interval [555, 777]

def pad_to_len(arr, L, pad_val=-1.0):
    """Right-side padding to fixed length L. arr is a 1D iterable."""
    n = len(arr)
    if n >= L:
        return np.asarray(arr[:L], dtype=np.float32)
    out = np.full((L,), pad_val, dtype=np.float32)
    if n > 0:
        out[:n] = np.asarray(arr, dtype=np.float32)
    return out

def extract_relative_time_feature(
    input_dir="csv_closeworld",
    output_npz="odoh_e3_fcnn_gru_closed_world.npz",
    sequence_len=200
):
    # ===== Containers =====
    # GRU: Timestamp sequences
    time_q_series = []   # Query timestamp sequences (padded) shape -> (N, sequence_len)
    time_r_series = []   # Response timestamp sequences (padded) shape -> (N, sequence_len)
    q_valid_len = []     # Valid length of query timestamps per sample
    r_valid_len = []     # Valid length of response timestamps per sample

    # FCNN: Aggregated statistics per trace (sum/count + N + TT)
    fcnn_stats = []      # shape -> (N, 6): [sum_q, count_q, sum_r, count_r, N, TT]

    labels = []
    paths  = []

    # ===== Iterate over data =====
    for domain in os.listdir(input_dir):
        domain_path = os.path.join(input_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        label = domain

        for fname in os.listdir(domain_path):
            if not fname.endswith(".csv"):
                continue

            csv_path = os.path.join(domain_path, fname)
            try:
                df = pd.read_csv(csv_path, header=None)
                df.columns = [
                    "relative_time", "interval", "src_ip", "src_port",
                    "dst_ip", "dst_port", "length", "direction"
                ]
                # Retain necessary columns and convert types
                need_cols = ["relative_time", "length", "direction"]
                df = df[need_cols].apply(pd.to_numeric, errors="coerce").dropna()

                # Select queries/responses (based on direction and length)
                q_mask = (
                    (df["direction"] == CLIENT_TO_SERVER) &
                    (df["length"] >= QUERY_LEN_RANGE[0]) &
                    (df["length"] <= QUERY_LEN_RANGE[1])
                )
                r_mask = (
                    (df["direction"] == SERVER_TO_CLIENT) &
                    (df["length"] >= RESP_LEN_RANGE[0]) &
                    (df["length"] <= RESP_LEN_RANGE[1])
                )

                q_df = df.loc[q_mask].copy()
                r_df = df.loc[r_mask].copy()

                # Skip if both categories are completely empty
                if q_df.empty and r_df.empty:
                    continue

                # Sort by time
                q_df.sort_values("relative_time", inplace=True)
                r_df.sort_values("relative_time", inplace=True)

                # ===== 1) GRU Input: Timestamp sequences (no intervals) =====
                q_times = q_df["relative_time"].to_numpy(dtype=np.float32)
                r_times = r_df["relative_time"].to_numpy(dtype=np.float32)

                q_len = len(q_times)
                r_len = len(r_times)

                time_q_series.append(pad_to_len(q_times, sequence_len, pad_val=-1.0))
                time_r_series.append(pad_to_len(r_times, sequence_len, pad_val=-1.0))
                q_valid_len.append(q_len)
                r_valid_len.append(r_len)

                # ===== 2) FCNN Input: Aggregated per trace (sum/count + N + TT) =====
                # Sum/count of query/response packet lengths
                q_lengths = q_df["length"].to_numpy(dtype=np.float64)
                r_lengths = r_df["length"].to_numpy(dtype=np.float64)

                sum_q = float(q_lengths.sum()) if q_len > 0 else 0.0
                cnt_q = float(q_len)
                sum_r = float(r_lengths.sum()) if r_len > 0 else 0.0
                cnt_r = float(r_len)

                # N = Total number of packets (queries + responses)
                N_total = float(q_len + r_len)

                # TT = Time span from the first "query" to the last "response"
                # If one end is missing, use the overall min/max time of selected packets
                have_query  = q_len > 0
                have_resp   = r_len > 0

                if have_query and have_resp:
                    t_start = float(q_times[0])       # First query time
                    t_end   = float(r_times[-1])      # Last response time
                    TT = max(0.0, t_end - t_start)
                else:
                    # Fallback: Use (min_all, max_all) span
                    all_times = []
                    if q_len > 0:
                        all_times.extend(q_times.tolist())
                    if r_len > 0:
                        all_times.extend(r_times.tolist())
                    if len(all_times) >= 2:
                        TT = float(max(all_times) - min(all_times))
                    else:
                        TT = 0.0

                fcnn_stats.append([sum_q, cnt_q, sum_r, cnt_r, N_total, TT])

                # Record labels and paths
                labels.append(label)
                paths.append(csv_path)

            except Exception as e:
                print(f"[ERROR] {csv_path}: {e}")

    # ===== Convert to NumPy =====
    time_q_series = np.asarray(time_q_series, dtype=np.float32)
    time_r_series = np.asarray(time_r_series, dtype=np.float32)
    q_valid_len   = np.asarray(q_valid_len, dtype=np.int32)
    r_valid_len   = np.asarray(r_valid_len, dtype=np.int32)
    fcnn_stats    = np.asarray(fcnn_stats, dtype=np.float32)
    labels        = np.asarray(labels)
    paths         = np.asarray(paths)

    if len(labels) == 0:
        raise RuntimeError("No valid samples. Please check filter conditions and direction encoding.")

    # Label encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)

    # Train/test split
    (q_train, q_test,
     r_train, r_test,
     qlen_train, qlen_test,
     rlen_train, rlen_test,
     fcnn_train, fcnn_test,
     y_train, y_test,
     paths_train, paths_test) = train_test_split(
        time_q_series, time_r_series,
        q_valid_len, r_valid_len,
        fcnn_stats,
        y, paths,
        test_size=0.2, random_state=42, stratify=y
    )

    # Save .npz
    np.savez_compressed(
        output_npz,
        # GRU branch: Timestamp sequences
        time_q_train=q_train,
        time_q_test=q_test,
        time_r_train=r_train,
        time_r_test=r_test,
        q_valid_len_train=qlen_train,
        q_valid_len_test=qlen_test,
        r_valid_len_train=rlen_train,
        r_valid_len_test=rlen_test,

        # FCNN branch: Aggregated statistics per trace
        # Column order: [sum_q, count_q, sum_r, count_r, N_total, TT]
        fcnn_train=fcnn_train,
        fcnn_test=fcnn_test,

        # Labels/paths/classes
        y_train=y_train,
        y_test=y_test,
        train_paths=paths_train,
        test_paths=paths_test,
        classes=encoder.classes_,
    )

    print(f"[OK] Saved {output_npz}")
    print(f"GRU timestamps Q shape: Train {q_train.shape}, Test {q_test.shape}")
    print(f"GRU timestamps R shape: Train {r_train.shape}, Test {r_test.shape}")
    print(f"FCNN stats shape: Train {fcnn_train.shape}, Test {fcnn_test.shape}")
    print(f"Classes: {len(encoder.classes_)} | Samples: {len(y)}")

if __name__ == "__main__":
    extract_relative_time_feature()
