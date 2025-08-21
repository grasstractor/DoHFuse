import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def extract_length_series_signed_firstNfolders_all(
    input_dir="csv_closeworld",
    output_npz="packet_length_signed_449_all.npz",
    sequence_len=200,
    max_folders=449
):
    """
    Only read the first max_folders folders (sorted by name), and read "all" CSV samples in each folder:
      - direction ∈ {1, -1}; when adding samples, multiply length by direction to get signed sequences
      - Length filter range: [233,333] or [555,1200] (no absolute value taken)
      - Each CSV generates one sample; pad with -1 if less than sequence_len
      - Attempt stratified split first; if the number of samples in a class <2 causes failure, it will print the problematic classes and their paths and fall back to stratify=None
    """
    time_series, labels, paths = [], [], []

    # Only take the first N "directory names"
    all_domains = sorted([d for d in os.listdir(input_dir)
                          if os.path.isdir(os.path.join(input_dir, d))])
    selected_domains = all_domains[:max_folders]
    print(f"[INFO] Processing the first {len(selected_domains)} folders (reading all CSV samples in each folder)")

    total_files = 0
    used_files = 0

    for domain in selected_domains:
        dpath = os.path.join(input_dir, domain)
        csv_files = sorted([f for f in os.listdir(dpath) if f.endswith(".csv")])
        for fname in csv_files:
            total_files += 1
            fpath = os.path.join(dpath, fname)
            try:
                df = pd.read_csv(fpath, header=None)
                df.columns = ["relative_time","interval","src_ip","src_port",
                              "dst_ip","dst_port","length","direction"]

                # Numeric conversion and cleaning
                num_cols = ["length","direction","relative_time"]
                df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
                df = df.dropna(subset=num_cols)

                # Direction filter: {1, -1}
                df = df[df["direction"].isin([1, -1])]

                # Length range
                cond = ((df["length"].between(233, 333)) |
                        (df["length"].between(555, 1200)))
                df = df[cond].copy()
                if df.empty:
                    continue

                # Sort by time and generate signed lengths
                df.sort_values("relative_time", inplace=True)
                sign = np.where(df["direction"].to_numpy() == 1, 1.0, -1.0)
                lengths_signed = df["length"].to_numpy(dtype=np.float32) * sign

                # Fix length to sequence_len (pad with -1 if insufficient)
                if len(lengths_signed) >= sequence_len:
                    seq = lengths_signed[:sequence_len]
                else:
                    seq = np.pad(lengths_signed, (0, sequence_len - len(lengths_signed)),
                                 mode="constant", constant_values=-1)

                time_series.append(seq.astype(np.float32))
                labels.append(domain)
                paths.append(fpath)
                used_files += 1
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

    if len(time_series) < 2:
        raise ValueError("Valid sample count < 2, unable to perform train/test split. Please relax the filter or increase max_folders.")

    # Convert to arrays
    time_series = np.array(time_series, dtype=np.float32)
    labels = np.array(labels)
    paths = np.array(paths)

    # Label encoding
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels).astype(np.int64)

    # Stratified split; report problematic classes and paths on failure, then fallback
    try:
        (time_train, time_test,
         y_train, y_test,
         paths_train, paths_test) = train_test_split(
            time_series, encoded_labels, paths,
            test_size=0.2, random_state=42, stratify=encoded_labels
        )
    except ValueError as e:
        print(f"[WARN] Stratified split failed: {e}")
        # Locate classes with <2 samples
        vc = pd.Series(labels).value_counts()
        bad_classes = vc[vc < 2]
        if not bad_classes.empty:
            print("[WARN] The following classes have less than 2 samples in the selected data:")
            for cls, cnt in bad_classes.items():
                print(f"  - Class: {cls}, Sample count: {cnt}")
                idxs = np.where(labels == cls)[0]
                for i in idxs:
                    print(f"      Sample idx={i}, Path={paths[i]}")
        else:
            print("[WARN] No obvious small sample classes located, please check data distribution or filter conditions.")

        print("[WARN] Falling back to non-stratified split stratify=None")
        (time_train, time_test,
         y_train, y_test,
         paths_train, paths_test) = train_test_split(
            time_series, encoded_labels, paths,
            test_size=0.2, random_state=42, stratify=None
        )

    # Save
    np.savez_compressed(
        output_npz,
        time_train=time_train,
        time_test=time_test,
        y_train=y_train,
        y_test=y_test,
        train_paths=paths_train,
        test_paths=paths_test,
        classes=encoder.classes_
    )

    print(f"[OK] Saved {output_npz}")
    print(f"Folders used: {len(selected_domains)} / {len(all_domains)} total")
    print(f"CSV files scanned: {total_files}, samples kept: {used_files}")
    print(f"Train shape: {time_train.shape}, Test shape: {time_test.shape}")
    print(f"Unique classes: {len(encoder.classes_)}")

if __name__ == "__main__":
    extract_length_series_signed_firstNfolders_all()
