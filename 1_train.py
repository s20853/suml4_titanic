from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    base_data = pd.read_csv("DSP_1.csv")  # ładujemy dane z pliku (używamy pandas)
    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    data = base_data[cols].copy()
    data["Age"].fillna((data["Age"].mean()),
                       inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)
    encoder = LabelEncoder()
    data.loc[:, "Sex"] = encoder.fit_transform(data.loc[:, "Sex"])
    print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    data.loc[:, "Embarked"] = encoder.fit_transform(data.loc[:, "Embarked"])
    print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

    print(min(data["Fare"]))
    print(max(data["Fare"]))


    X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis=1),
                                                        data['Survived'], test_size=0.2,
                                                        random_state=101)

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=58)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    filename = 'model-titanic.sv'
    pickle.dump(model, open(filename, 'wb'))

    print(score)


if __name__ == '__main__':
    main()
