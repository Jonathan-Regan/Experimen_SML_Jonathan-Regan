import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_path: str):
    # 1. Load data
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()
    
    # 2. Pisahkan fitur & target
    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Definisi fitur numerik & kategorikal
    numerical_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]

    categorical_features = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]

    # 4. Scaling numerik
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(
        X[numerical_features]
    )

    # 5. Gabungkan kembali
    df_processed = X.copy()
    df_processed["target"] = y.values

    # 6. Simpan hasil
    df_processed.to_csv(output_path, index=False)

    return df_processed


if __name__ == "__main__":
    preprocess_data(
        "heart_raw/heart.csv",
        "preprocessing/heart_preprocessing/heart_preprocessed.csv"
    )

