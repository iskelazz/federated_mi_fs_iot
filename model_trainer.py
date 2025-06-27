import time
import numpy as np
from sklearn.metrics import f1_score, recall_score

class ModelTrainer:
    def __init__(self, clf_type="logreg", random_state=42):
        self.clf_type = clf_type
        self.random_state = random_state
        self.clf = self._init_clf()
    
    def _init_clf(self):
        if self.clf_type == "logreg":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=500, random_state=self.random_state)
        elif self.clf_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(n_neighbors=5)
        elif self.clf_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Tipo de clasificador no soportado: {self.clf_type}")

    def fit_predict(self, X_train, y_train, X_test, y_test):
        import numpy as np
        import time

        t_train0 = time.time()
        self.clf.fit(X_train, y_train)
        t_train1 = time.time()
        y_pred = self.clf.predict(X_test)
        t_pred1 = time.time()
        acc = np.mean(y_pred == y_test)
        train_time = t_train1 - t_train0
        pred_time = t_pred1 - t_train1

        # --- MÉTRICAS CLAVE ---
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        return {
            "accuracy": acc,
            "y_pred": y_pred,
            "train_time": train_time,
            "pred_time": pred_time,
            "recall": recall,
            "f1_score": f1
        }
