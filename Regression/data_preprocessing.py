import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "melb_data(Uncleaned).csv"

RANDOM_STATE = 42
TARGET = "Price"

# الأعمدة اللي هنعالج فيها الـ Outliers (من غير الإحداثيات والتواريخ والـ Postcode)
OUTLIER_COLS = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
                "Landsize", "BuildingArea"]


def clean_data(df):
    df = df.copy()

    # استخراج السنة والشهر من التاريخ
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["SaleYear"] = date.dt.year
    df["SaleMonth"] = date.dt.month

    # حذف أعمدة غير مفيدة مباشرة في التوقع
    df = df.drop(columns=["Address", "Date"], errors="ignore")

    # سنة بناء غير منطقية
    if "YearBuilt" in df.columns:
        df.loc[(df["YearBuilt"] < 1800) |
               (df["YearBuilt"] > df["SaleYear"] + 3), "YearBuilt"] = np.nan

    # عدد غرف نوم غير منطقي مقارنة بعدد الغرف
    df.loc[df["Bedroom2"] > df["Rooms"] + 5, "Bedroom2"] = np.nan

    # اعتبار بعض القيم غير المنطقية Missing
    if "BuildingArea" in df.columns:
        df.loc[df["BuildingArea"] <= 10, "BuildingArea"] = np.nan

    if "Landsize" in df.columns:
        df.loc[df["Landsize"] <= 0, "Landsize"] = np.nan

    return df.reset_index(drop=True)


def fill_council(X, council_by_suburb):
    # تعويض CouncilArea من أكثر قيمة في نفس الـ Suburb
    X["CouncilArea"] = X["CouncilArea"].fillna(X["Suburb"].map(council_by_suburb))
    return X


def imbalance_weights(y_log, bins=10):
    # وزن أعلى للنطاقات السعرية النادرة (SMOTE خاص بالـ Classification)
    groups = pd.cut(y_log, bins=bins, labels=False)
    weights = 1 / np.sqrt(groups.map(groups.value_counts()))
    weights = (weights / weights.mean()).clip(upper=5)
    return weights / weights.mean()


def transform_new(df_new, info):
    # تطبيق نفس المعالجة على بيانات جديدة (بدون fit)
    X = clean_data(df_new).drop(columns=[TARGET], errors="ignore")
    X = fill_council(X, info["council_by_suburb"])

    num = info["numeric_cols"]
    X[num] = X[num].astype(float).fillna(info["numeric_medians"])
    for col, value in info["categorical_modes"].items():
        X[col] = X[col].fillna(value)

    for col, (lower, upper) in info["outlier_limits"].items():
        X[col] = X[col].clip(lower, upper)

    X = pd.get_dummies(X, columns=info["categorical_cols"], dtype=int)
    X = X.reindex(columns=info["dummy_columns"], fill_value=0)
    X = X[info["selected_features"]]

    return pd.DataFrame(info["scaler"].transform(X),
                        columns=info["selected_features"], index=X.index)


def main():
    # 2) قراءة ملف CSV
    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape)

    # 3) تنظيف البيانات
    # حذف التكرار (على البيانات الأصلية قبل حذف Address و Date)
    rows_before = len(df)
    df = df.drop_duplicates()
    print("Duplicates removed:", rows_before - len(df))

    # حذف الصفوف اللي مفيهاش سعر صحيح
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    df = clean_data(df)
    print("Shape after cleaning:", df.shape)

    # فصل الـ Features عن الـ Target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # تحويل السعر لتقليل تأثير انحراف الأسعار
    y_log = np.log1p(y)

    # 4) Train/Test Split (80% تدريب و20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.20, random_state=RANDOM_STATE
    )

    X_train, X_test = X_train.copy(), X_test.copy()

    # تحديد الأعمدة الرقمية والفئوية اعتماداً على بيانات التدريب
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # حدود الـ Outliers تتحسب من القيم الأصلية قبل تعويض الناقص
    raw_quantiles = X_train[OUTLIER_COLS].quantile([0.25, 0.75])

    # 5) Missing Values
    council_by_suburb = (X_train.dropna(subset=["CouncilArea"])
                         .groupby("Suburb")["CouncilArea"]
                         .agg(lambda s: s.mode().iloc[0]).to_dict())
    X_train = fill_council(X_train, council_by_suburb)
    X_test = fill_council(X_test, council_by_suburb)

    X_train[numeric_cols] = X_train[numeric_cols].astype(float)
    X_test[numeric_cols] = X_test[numeric_cols].astype(float)
    # الأرقام: نعوض القيم الناقصة بالـ Median المحسوب من train
    numeric_medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(numeric_medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(numeric_medians)

    # النصوص: نعوض القيم الناقصة بأكثر قيمة متكررة في train
    categorical_modes = {}
    for col in categorical_cols:
        mode_values = X_train[col].mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        if col == "CouncilArea":
            fill_value = "Unknown"
        categorical_modes[col] = fill_value
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    # 6) Outliers Detection باستخدام IQR ثم Capping
    # نحسب الحدود من train فقط، ثم نطبقها على train و test
    outlier_limits = {}
    print("\nOutliers (train):")
    for col in OUTLIER_COLS:
        q1 = raw_quantiles.loc[0.25, col]
        q3 = raw_quantiles.loc[0.75, col]
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # عدد الغرف: حدود أوسع عشان البيوت الكبيرة الطبيعية متتقصش
        if col in ["Rooms", "Bedroom2", "Bathroom", "Car"]:
            lower, upper = X_train[col].quantile([0.001, 0.995])

        count = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
        print(f"  {col:13s} limits=({lower:.1f}, {upper:.1f}) outliers={count}")

        outlier_limits[col] = (lower, upper)
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    # 7) Encoding: تحويل الأعمدة النصية إلى أرقام باستخدام One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, dtype=int)

    # توحيد الأعمدة بين train و test
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 8) Feature Selection: أعلى 25 feature ارتباطاً بالسعر
    # مع تخطي أي feature مرتبط جداً (> 0.90) بواحد مختار قبله
    all_dummy_columns = X_train.columns.tolist()
    correlations = X_train.corrwith(y_train).abs().fillna(0)
    selected_features = []
    for col in correlations.sort_values(ascending=False).index:
        if all(abs(X_train[col].corr(X_train[s])) <= 0.90 for s in selected_features):
            selected_features.append(col)
        if len(selected_features) == 25:
            break

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # 9) Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=selected_features,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=selected_features,
        index=X_test.index
    )

    # Handle Imbalance: أوزان للـ train فقط
    sample_weights = imbalance_weights(y_train)
    print(f"\nSample weights: min={sample_weights.min():.2f}, "
          f"max={sample_weights.max():.2f}")

    # 10) حفظ النتائج
    train_output = X_train_scaled.copy()
    train_output["Price_log"] = y_train
    train_output["Price"] = np.expm1(y_train)

    train_output["sample_weight"] = sample_weights

    test_output = X_test_scaled.copy()
    test_output["Price_log"] = y_test
    test_output["Price"] = np.expm1(y_test)

    train_output.to_csv("train_preprocessed.csv", index=False)
    test_output.to_csv("test_preprocessed.csv", index=False)

    # ملف كامل: كل الصفوف بعد المعالجة بنفس ترتيبها
    # عمود Split بيوضح الصف تبع train ولا test
    # وزن صفوف الـ test = 1 (الأوزان بتستخدم في تدريب الـ train بس)
    full_output = pd.concat([
        train_output.assign(Split="train"),
        test_output.assign(sample_weight=1.0, Split="test")
    ]).sort_index()
    full_output.to_csv("melb_preprocessed_full.csv", index=False)

    # حفظ المعلومات اللازمة لتطبيق نفس المعالجة لاحقاً على بيانات جديدة
    preprocessing_info = {
        "council_by_suburb": council_by_suburb,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "outlier_limits": outlier_limits,
        "dummy_columns": all_dummy_columns,
        "selected_features": selected_features,
        "scaler": scaler
    }
    joblib.dump(preprocessing_info, "preprocessing_info.joblib")

    print("Missing after preprocessing:",
          int(X_train_scaled.isna().sum().sum() + X_test_scaled.isna().sum().sum()))

    print("\nPreprocessing completed successfully.")
    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Selected features:", len(selected_features))
    print("Saved files:")
    print("- train_preprocessed.csv")
    print("- test_preprocessed.csv")
    print("- preprocessing_info.joblib")
    print("- melb_preprocessed_full.csv", full_output.shape)


if __name__ == "__main__":
    main()