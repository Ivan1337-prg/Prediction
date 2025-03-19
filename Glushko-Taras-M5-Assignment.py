import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo

df = fetch_ucirepo(id=320).data.features.copy()
df['final_grade'] = fetch_ucirepo(id=320).data.targets['G3']
df.dropna(inplace=True)

X, y = shuffle(pd.get_dummies(df.drop(columns=['final_grade'])), df['final_grade'], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(kernel='linear').fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}\nVariance Score: {explained_variance_score(y_test, y_pred):.4f}")

y_pred_bin, y_test_bin = (y_pred >= 12).astype(int), (y_test >= 12).astype(int)
conf_mat = confusion_matrix(y_test_bin, y_pred_bin)

plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['<12', '≥12']), plt.yticks([0, 1], ['<12', '≥12'])
plt.ylabel("Actual"), plt.xlabel("Predicted"), plt.show()

print(classification_report(y_test_bin, y_pred_bin))
