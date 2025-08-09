import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings

warnings.filterwarnings("ignore")  # Supaya warning tidak tampil (opsional)

def main():
    # Baca dataset
    df = pd.read_csv("customer_feedback.csv")

    # Label Encoding untuk kolom kategorikal
    le_gender = LabelEncoder()
    le_country = LabelEncoder()
    le_feedback = LabelEncoder()
    le_loyalty = LabelEncoder()

    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Country"] = le_country.fit_transform(df["Country"])
    df["FeedbackScore"] = le_feedback.fit_transform(df["FeedbackScore"])
    df["LoyaltyLevel"] = le_loyalty.fit_transform(df["LoyaltyLevel"])

    # Fitur & Target
    X = df.drop(["CustomerID", "SatisfactionScore"], axis=1)
    y = (df["SatisfactionScore"] >= 75).astype(int)  # 1 = puas, 0 = tidak puas

    # Split data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling fitur agar range nilainya seragam
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inisialisasi dan training model KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Simpan model, scaler, dan encoder ke file .pkl
    with open("model_knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("encoders.pkl", "wb") as f:
        pickle.dump(
            {
                "gender": le_gender,
                "country": le_country,
                "feedback": le_feedback,
                "loyalty": le_loyalty,
            },
            f,
        )

    print("✅ Model berhasil disimpan!")

if __name__ == "__main__":
    main()
