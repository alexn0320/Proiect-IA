import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Baza de date
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'

#citirea datelor si separarea etichetelor
df = pd.read_csv(url, names=['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne',
                'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
                'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type'])

x = df.drop(['animal_name', 'class_type'], axis=1)
y = df['class_type']

#test_size 25%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

#rulam toti hiperparametrii pentru cost si gamma
for c_exp in range(-5, 8, 2):
    for g_exp in range(-15, 4, 2):
        cost = 2 ** c_exp
        gamma = 2 ** g_exp

        print(f"Hiperparametrii: C=2^{c_exp}, gamma=2^{g_exp}")

        model = SVC(kernel='rbf', C=cost, gamma=gamma)
        model.fit(x_train_s, y_train)

        y_pred = model.predict(x_test_s)
        print("Acuratețea modelului:", accuracy_score(y_test, y_pred))
        print("Matricea de confuzie:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))