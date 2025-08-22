import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def extract_relative_time_feature(
    closed_dir="captures_csv_legal",
    openworld_dir="captures_csv_openworld",
    output_npz="interval_time_openworld.npz",
    sequence_len=200
):
    # 初始化数据结构
    time_series = []
    stat_features = []
    valid_lengths = []
    labels = []
    paths = []

    def process_domain_folder(domain_path, label):
        """读取一个域名文件夹下的所有 CSV 并提取特征"""
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

                numeric_cols = ["length", "direction", "relative_time"]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                df = df.dropna(subset=numeric_cols)

                filtered = df[
                    (df["direction"] == 1) &
                    (df["length"] >= 233) & (df["length"] <= 322)
                ].copy()
                
                if len(filtered) < 2:
                    continue
                    
                filtered.sort_values("relative_time", inplace=True)
                filtered["inter_arrival"] = 0.0
                filtered.iloc[1:, filtered.columns.get_loc("inter_arrival")] = \
                    filtered["relative_time"].iloc[1:].values - \
                    filtered["relative_time"].iloc[:-1].values
                
                inter_arrivals = filtered["inter_arrival"].to_numpy()
                n = len(inter_arrivals)
                valid_len = n
                pkt_count = n

                actual_intervals = inter_arrivals[1:]
                if len(actual_intervals) > 0:
                    std_dev = np.std(actual_intervals)
                    mean_val = np.mean(actual_intervals)
                    burstiness = 0.0 if mean_val < 1e-10 else std_dev / mean_val
                else:
                    burstiness = 0.0
                
                if n >= sequence_len:
                    padded = inter_arrivals[:sequence_len]
                else:
                    padded = np.pad(inter_arrivals, (0, sequence_len - n), 
                                    mode='constant', constant_values=-1)
                
                time_series.append(padded)
                stat_features.append([pkt_count, burstiness])
                valid_lengths.append(valid_len)
                labels.append(label)
                paths.append(csv_path)

            except Exception as e:
                print(f"[ERROR] {csv_path}: {e}")

    # 处理 closed-world 数据
    for domain in os.listdir(closed_dir):
        domain_path = os.path.join(closed_dir, domain)
        if os.path.isdir(domain_path):
            process_domain_folder(domain_path, domain)

    # 处理 open-world 数据（标签固定为 'unknown'）
    for domain in os.listdir(openworld_dir):
        domain_path = os.path.join(openworld_dir, domain)
        if os.path.isdir(domain_path):
            process_domain_folder(domain_path, "unknown")

    # 转为数组
    time_series = np.array(time_series, dtype=np.float32)
    stat_features = np.array(stat_features, dtype=np.float32)
    valid_lengths = np.array(valid_lengths, dtype=np.int32)
    labels = np.array(labels)
    paths = np.array(paths)

    # 标签编码
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels).astype(np.int64)

    # 划分数据集
    (time_train, time_test, 
     stat_train, stat_test, 
     valid_len_train, valid_len_test,
     y_train, y_test, 
     paths_train, paths_test) = train_test_split(
        time_series, stat_features, valid_lengths, encoded_labels, paths,
        test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # 保存为 .npz
    np.savez_compressed(
        output_npz,
        time_train=time_train,
        time_test=time_test,
        stat_train=stat_train,
        stat_test=stat_test,
        valid_len_train=valid_len_train,
        valid_len_test=valid_len_test,
        y_train=y_train,
        y_test=y_test,
        train_paths=paths_train,
        test_paths=paths_test,
        classes=encoder.classes_
    )

    print(f"[OK] Saved {output_npz}")
    print(f"Time series shape: Train {time_train.shape}, Test {time_test.shape}")
    print(f"Stat features shape: Train {stat_train.shape}, Test {stat_test.shape}")
    print(f"Valid lengths: Train {valid_len_train.shape}, Test {valid_len_test.shape}")
    print(f"Classes: {list(encoder.classes_)}")

if __name__ == "__main__":
    extract_relative_time_feature()
