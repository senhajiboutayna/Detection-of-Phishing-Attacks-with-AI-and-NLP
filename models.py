import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler


file_path = 'data/cleaned_phishing_data.csv'
data = pd.read_csv(file_path)


print(data.columns)


features = ['sender_name_length', 'special_char_density', 'sender_domain_encoded','num_urls','suspicious_urls','phishing_words','receiver_domain_encoded']
X = data[features]


y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = MinMaxScaler()
X_train[['sender_name_length', 'special_char_density']] = scaler.fit_transform(X_train[['sender_name_length', 'special_char_density']])
X_test[['sender_name_length', 'special_char_density']] = scaler.transform(X_test[['sender_name_length', 'special_char_density']])

# ------------------------------ XGBoost Model ------------------------------

xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=6)
xgb_model.fit(X_train, y_train)


y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"XGBoost - Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost - ROC AUC: {roc_auc_xgb:.4f}")

cm = confusion_matrix(y_test, y_pred_xgb)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb)


plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matrice de confusion - XGBoost")
plt.colorbar()
plt.ylabel('Vrai')
plt.xlabel('Prédiction')
plt.xticks([0, 1], ['Non-Spam', 'Spam'])
plt.yticks([0, 1], ['Non-Spam', 'Spam'])
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - XGBoost')
plt.legend(loc="lower right")
plt.show()

# ------------------------------ SVM Model ------------------------------
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Vérification des NaN dans les données d'entrée
print(f"NaN dans X_train: {np.isnan(X_train).sum()}")
print(f"NaN dans X_test: {np.isnan(X_test).sum()}")
print(f"NaN dans y_train: {np.isnan(y_train).sum()}")
print(f"NaN dans y_test: {np.isnan(y_test).sum()}")

# Si des NaN existent dans les données, vous pouvez les gérer en les remplissant par 0
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test, nan=0)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # 'rbf' est le noyau radial
svm_model.fit(X_train, y_train)


y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]

accuracy_svm = accuracy_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_proba_svm)

print(f"SVM - Accuracy: {accuracy_svm:.4f}")
print(f"SVM - ROC AUC: {roc_auc_svm:.4f}")


cm_svm = confusion_matrix(y_test, y_pred_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_proba_svm)


plt.figure(figsize=(8, 6))
plt.imshow(cm_svm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matrice de confusion - SVM")
plt.colorbar()
plt.ylabel('Vrai')
plt.xlabel('Prédiction')
plt.xticks([0, 1], ['Non-Spam', 'Spam'])
plt.yticks([0, 1], ['Non-Spam', 'Spam'])
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - SVM')
plt.legend(loc="lower right")
plt.show()

# ------------------------------RandomForest Model ------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]


accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Random Forest - Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest - ROC AUC: {roc_auc_rf:.4f}")


cm_rf = confusion_matrix(y_test, y_pred_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)


plt.figure(figsize=(8, 6))
plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matrice de confusion - Random Forest")
plt.colorbar()
plt.ylabel('Vrai')
plt.xlabel('Prédiction')
plt.xticks([0, 1], ['Non-Spam', 'Spam'])
plt.yticks([0, 1], ['Non-Spam', 'Spam'])
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()

models = ['XGBoost', 'SVM', 'Random Forest']
accuracies = [accuracy_xgb, accuracy_svm, accuracy_rf]
roc_aucs = [roc_auc_xgb, roc_auc_svm, roc_auc_rf]


results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'ROC AUC': roc_aucs
})

print(results_df)
print(f'Le modèle le plus performant : {results_df["Model"][results_df["Accuracy"].idxmax()]}')