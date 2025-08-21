import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def extract_relative_time_feature(
    input_dir="csv_closeworld",
    output_npz="interval_time_closed_world_449.npz",
    sequence_len=200
):
    # Initialize data structures
    time_series = []    # Temporal features (N, sequence_len)
    stat_features = []  # Statistical features (N, 2): [Total packets, burstiness]
    valid_lengths = []   # Number of valid packets (N,)
    labels = []         # Sample labels
    paths = []          # File paths

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

                # Convert data types and handle missing values
                numeric_cols = ["length", "direction", "relative_time"]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                df = df.dropna(subset=numeric_cols)

                # Filtering conditions remain unchanged
                filtered = df[
                    (df["direction"] == 1) &
                    (df["length"] >= 233) & (df["length"] <= 350)
                ].copy()
                
                # Skip the file if there are no packets meeting the criteria
                if len(filtered) < 2:  # At least 2 packets are needed to calculate intervals
                    continue
                    
                # Sort by relative time
                filtered.sort_values("relative_time", inplace=True)
                
                # Calculate inter-arrival time
                # The interval of the first packet is set to 0
                filtered["inter_arrival"] = 0.0
                # From the second packet, calculate the time difference with the previous packet
                filtered.iloc[1:, filtered.columns.get_loc("inter_arrival")] = \
                    filtered["relative_time"].iloc[1:].values - \
                    filtered["relative_time"].iloc[:-1].values
                
                # Extract inter-arrival time sequence
                inter_arrivals = filtered["inter_arrival"].to_numpy()
                n = len(inter_arrivals)
                
                # Record the number of valid packets
                valid_len = n
                
                # ===== Statistical feature calculation =====
                # 1. Total number of packets (meeting the filter criteria)
                pkt_count = n
                
                # 2. Calculate burstiness (using coefficient of variation)
                # Coefficient of Variation (CV) = std / mean
                # Larger value means more bursty arrivals
                # Smaller value means more uniform arrivals
                
                # Exclude the first packet's interval (always 0)
                actual_intervals = inter_arrivals[1:]
                
                if len(actual_intervals) > 0:
                    std_dev = np.std(actual_intervals)
                    mean_val = np.mean(actual_intervals)
                    
                    # Handle mean=0 case (avoid division by zero)
                    if mean_val < 1e-10:  # Very small threshold
                        burstiness = 0.0
                    else:
                        burstiness = std_dev / mean_val
                else:
                    burstiness = 0.0
                
                # ===== Temporal feature processing =====
                if n >= sequence_len:
                    padded = inter_arrivals[:sequence_len]
                else:
                    padded = np.pad(inter_arrivals, (0, sequence_len - n), 
                                mode='constant', constant_values=-1)
                
                # Store features
                time_series.append(padded)
                stat_features.append([pkt_count, burstiness])
                valid_lengths.append(valid_len)
                labels.append(label)
                paths.append(csv_path)

            except Exception as e:
                print(f"[ERROR] {csv_path}: {e}")

    # Convert to NumPy arrays
    time_series = np.array(time_series, dtype=np.float32)
    stat_features = np.array(stat_features, dtype=np.float32)
    valid_lengths = np.array(valid_lengths, dtype=np.int32)
    labels = np.array(labels)
    paths = np.array(paths)

    # Label encoding
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels).astype(np.int64)

    # Split dataset
    (time_train, time_test, 
     stat_train, stat_test, 
     valid_len_train, valid_len_test,
     y_train, y_test, 
     paths_train, paths_test) = train_test_split(
        time_series, stat_features, valid_lengths, encoded_labels, paths,
        test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Save as .npz file
    np.savez_compressed(
        output_npz,
        # Temporal feature branch
        time_train=time_train,
        time_test=time_test,
        # Statistical feature branch
        stat_train=stat_train,
        stat_test=stat_test,
        # Number of valid packets
        valid_len_train=valid_len_train,
        valid_len_test=valid_len_test,
        # Labels and paths
        y_train=y_train,
        y_test=y_test,
        train_paths=paths_train,
        test_paths=paths_test,
        # Metadata
        classes=encoder.classes_
    )

    print(f"[OK] Saved {output_npz}")
    print(f"Time series shape: Train {time_train.shape}, Test {time_test.shape}")
    print(f"Stat features shape: Train {stat_train.shape}, Test {stat_test.shape}")
    print(f"Valid lengths: Train {valid_len_train.shape}, Test {valid_len_test.shape}")
    print(f"Classes: {len(encoder.classes_)}")

if __name__ == "__main__":
    extract_relative_time_feature()