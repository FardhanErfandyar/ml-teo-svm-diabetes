# Mengimport library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)
import seaborn as sns

# 1. Input
# Mengimport dataset
data = pd.read_csv("diabetes.csv")

# Menampilkan data awal
print("Data awal".center(75, "="))
print(data.head())
print("=" * 75)

# 2. Preprocessing
# Pengecekan missing value
print("Pengecekan missing value".center(75, "="))
print(data.isnull().sum())
print("=" * 75)

# Memeriksa duplikat data
print("Jumlah baris duplikat dalam dataset:", data.duplicated().sum())

# Menampilkan boxplot untuk setiap fitur dalam dataset sebelum menangani outlier
plt.figure(figsize=(15, 10))
data.boxplot(rot=45)
plt.title("Boxplot untuk Setiap Fitur")
plt.show()

# Deteksi outliers dengan Z-score
z_scores = np.abs(stats.zscore(data._get_numeric_data()))
threshold = 3
outliers = (z_scores > threshold).any(axis=1)

# Menampilkan data yang memiliki outlier
print("Data yang memiliki outlier:".center(75, "="))
print(data[outliers])
print("=" * 75)

# Penanganan outliers dengan IQR (Interquartile Range)
for column in data._get_numeric_data().columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[column] = np.where(
        data[column] < lower_bound,
        lower_bound,
        np.where(data[column] > upper_bound, upper_bound, data[column]),
    )

# Menampilkan boxplot untuk setiap fitur dalam dataset setelah penanganan outlier
plt.figure(figsize=(15, 10))
data.boxplot(rot=45)
plt.title("Boxplot untuk Setiap Fitur Setelah Penanganan Outlier")
plt.show()

# Menampilkan data setelah menangani outlier
print("Data setelah menangani outlier".center(75, "="))
print(data.head())
print("=" * 75)

# 3. Transformasi
# Normalisasi data menggunakan Min-Max Scaling
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data._get_numeric_data())

# Konstruksi DataFrame dari hasil normalisasi
data_normalized = pd.DataFrame(data_minmax, columns=data._get_numeric_data().columns)

# Menampilkan hasil normalisasi Min-Max Scaling
print("Hasil Normalisasi Min-Max Scaling".center(75, "="))
print(data_normalized.head())
print("=" * 75)

# 4. Klasifikasi
# Grouping yang dibagi menjadi dua
print("Grouping Variable".center(75, "="))
X = data_normalized.iloc[:, 0:-1].values
y = data_normalized.iloc[:, -1].values
y = np.where(y > 0, 1, 0)
print("Data Variable".center(75, "="))
print(X)
print("Data Kelas".center(75, "="))
print(y)
print("=" * 75)

# Pembagian training dan testing
print("Splitting Data 43-57".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=260 / 460, random_state=0
)
print("Instance variable data training".center(75, "="))
print(X_train)
print("Instance kelas data training".center(75, "="))
print(y_train)
print("Instance variable data testing".center(75, "="))
print(X_test)
print("Instance kelas data testing".center(75, "="))
print(y_test)
print("=" * 75)
print()

# Pemodelan SVM
svm_model = SVC(random_state=0)
svm_model.fit(X_train, y_train)

# Prediksi SVM pada data training
print("Prediksi SVM pada data training".center(75, "="))
y_train_pred = svm_model.predict(X_train)
print(y_train_pred)
print("=" * 75)
print()

# Memilih persentase tertentu dari data pelatihan yang diprediksi dengan buruk (misalnya, 10%)
# Hitung kesalahan prediksi
errors = y_train != y_train_pred

# Jumlah total kesalahan prediksi
num_errors = np.sum(errors)

# Menghitung persentase data yang diprediksi buruk yang ingin dipilih
x_percent = 10  # misalnya, kita ingin memilih 10% dari data yang diprediksi buruk
num_selected = int(x_percent / 100 * num_errors)

# Memilih data yang diprediksi buruk
if num_selected > 0:
    selected_indices = np.random.choice(
        np.where(errors)[0], num_selected, replace=False
    )
    selected_badly_predicted_data = X_train[selected_indices]
    selected_badly_predicted_labels = y_train[selected_indices]

    print("Data yang diprediksi buruk:")
    print(selected_badly_predicted_data)
    print(selected_badly_predicted_labels)

    # Menambahkan data yang diprediksi buruk kembali ke dalam data training
    X_train = np.concatenate([X_train, selected_badly_predicted_data], axis=0)
    y_train = np.concatenate([y_train, selected_badly_predicted_labels], axis=0)

    print("Ukuran data training setelah ditambahkan data yang diprediksi buruk:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
else:
    print("Tidak ada data yang diprediksi buruk.")


# Prediksi SVM pada seluruh data training
print("Prediksi SVM pada seluruh data training".center(75, "="))
y_all_train_pred = svm_model.predict(X_train)
print(y_all_train_pred)
print("=" * 75)
print()

# Prediksi SVM pada data testing
print("Instance prediksi SVM: ")
Y_pred = svm_model.predict(X_test)
print(Y_pred)
print("=" * 75)
print()

from sklearn.model_selection import cross_val_score

# Melakukan 10-fold cross-validation
cross_val_scores = cross_val_score(svm_model, X_train, y_train, cv=10)

# Menampilkan hasil cross-validation
print("Hasil 10-fold Cross-Validation:")
print(cross_val_scores)

# Menampilkan rata-rata akurasi dari cross-validation
print("Rata-rata Akurasi 10-fold Cross-Validation:", cross_val_scores.mean())


# 5 Evaluasi
# Menghitung confusion matrix
cm = confusion_matrix(y_test, Y_pred)

print("CLASSIFICATION REPORT SVM".center(75, "="))
# Menghitung akurasi
accuracy = accuracy_score(y_test, Y_pred)
# Menghitung presisi
precision = precision_score(y_test, Y_pred)
# Menampilkan precision, recall, f1

print(classification_report(y_test, Y_pred))

# Menghitung sensitivity (true positive rate)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
sens = TN / (TN + FP) * 100

# Menghitung specificity (true negative rate)
spec = TP / (TP + FN) * 100

print("Akurasi : ", accuracy * 100, "%")
print("Sensitivity : " + str(sens))
print("Specificity : " + str(spec))
print("=" * 75)
print()

# Menampilkan confusion matrix
print("Confusion matrix for SVM\n", cm)
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("=" * 75)
print()

# Cross-validation accuracy
cross_val_accuracies = []
training_sizes = np.linspace(0.1, 0.9, 9)
for size in training_sizes:
    X_train_cv, _, y_train_cv, _ = train_test_split(
        X_train, y_train, test_size=1 - size, random_state=0
    )
    svm_model_cv = SVC(random_state=0)
    svm_model_cv.fit(X_train_cv, y_train_cv)
    y_train_cv_pred = svm_model_cv.predict(X_train_cv)
    accuracy_cv = accuracy_score(y_train_cv, y_train_cv_pred)
    cross_val_accuracies.append(accuracy_cv)

plt.figure(figsize=(10, 6))
plt.plot(training_sizes * 100, cross_val_accuracies, marker="o")
plt.title("Training Set Accuracy of the Diabetes Data Set")
plt.xlabel("Percentage of Training Data")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, Y_pred)
roc_auc = roc_auc_score(y_test, Y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC of SVM using Diabetes Data Set")
plt.legend(loc="lower right")
plt.show()
