#!/usr/bin/env python3
"""
OneClassSNN — одноклассовый детектор аномалий на импульсной нейросети
для эмулятора нейроморфного процессора AltAI-2 (Kaspersky Neuromorphic Platform).

Нейроморфный аналог One-Class SVM с RBF-ядром:
  - fit()      — прототипы нормального класса (k-means) квантуются в веса ±255,
                 порог калибруется по квантилю contamination на обучающих данных;
  - predict()  — образец кодируется частотой спайков и исполняется на бэкенде
                 (CPU для отладки, libknp-altai-backend.so — эмулятор AltAI-2),
                 низкий спайковый отклик = аномалия.

Архитектура сети (см. ARCH):
  input (N признаков, rate-coding) -- DeltaSynapse --> hidden (K нейронов AltAILIF)

Запускается ВНУТРИ контейнера KNP (~/knp/run_knp.sh), рядом должен лежать
собранный бинарник oneclass_runner (см. cpp/ и README.md).
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# --- Архитектура сети -------------------------------------------------------

ARCH = {
    "input": {"encoding": "rate", "description": "N признаков -> спайки с частотой v*steps"},
    "hidden": {
        "neuron": "AltAILIF",
        "params": {
            "potential_leak_": 0,
            "is_reset_": True,
            "potential_reset_value_": 0,
            "negative_activation_threshold_": 0,
            # activation_threshold_ задаётся параметром activation_threshold
        },
    },
    "synapse": "DeltaSynapse (delay=1, EXCITATORY/INHIBITORY по знаку веса)",
    "quantization": "веса — целые в диапазоне [-255, 255] (ограничение AltAI-2)",
}

BACKEND_CPU = "/usr/lib/libknp-cpu-single-threaded-backend.so"
BACKEND_ALTAI = "/usr/lib/libknp-altai-backend.so"

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(_HERE, "oneclass_runner")


# --- Низкоуровневый вызов раннера -------------------------------------------

def _run_backend(backend_path, weights_int, samples, steps, activation_threshold, workdir=None):
    """Прогнать samples через SNN на указанном бэкенде, вернуть спайковые скоры."""
    workdir = workdir or tempfile.mkdtemp(prefix="oneclass_")
    proto_csv = os.path.join(workdir, "prototypes.csv")
    data_csv = os.path.join(workdir, "samples.csv")
    np.savetxt(proto_csv, weights_int, fmt="%d", delimiter=",")
    np.savetxt(data_csv, samples, fmt="%.6f", delimiter=",")
    out = subprocess.run(
        [RUNNER, backend_path, proto_csv, data_csv, str(steps), str(activation_threshold)],
        capture_output=True, text=True, check=True,
    )
    # stdout раннера содержит и логи KNP — берём только числовые строки (скоры).
    scores = []
    for line in out.stdout.splitlines():
        line = line.strip()
        try:
            scores.append(float(line))
        except ValueError:
            continue
    return np.array(scores, dtype=float)


# --- Модель ------------------------------------------------------------------

class OneClassSNN:
    """Одноклассовый классификатор на SNN. API в стиле sklearn."""

    def __init__(self, n_prototypes=8, steps=30, activation_threshold=4000,
                 contamination=0.1, random_state=42):
        self.n_prototypes = n_prototypes
        self.steps = steps
        self.activation_threshold = activation_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.architecture = ARCH

    def _to_unit(self, X):
        """Сдвиг в неотрицательную область + L2-нормализация каждого образца.

        После этого потенциал нейрона-прототипа пропорционален косинусной
        близости образца к прототипу — нейрон срабатывает только на «похожих».
        """
        X = np.asarray(X, dtype=float) - self.x_offset_
        X = np.clip(X, 0.0, None)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return X / norm

    # -- обучение (только нормальный класс) --
    def fit(self, X, backend=BACKEND_CPU):
        X = np.asarray(X, dtype=float)
        self.x_offset_ = X.min()
        Xn = self._to_unit(X)

        # Прототипы: простой k-means на numpy (на единичной сфере).
        rng = np.random.default_rng(self.random_state)
        centers = Xn[rng.choice(len(Xn), self.n_prototypes, replace=False)]
        for _ in range(25):
            d = ((Xn[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            for k in range(self.n_prototypes):
                if (labels == k).any():
                    centers[k] = Xn[labels == k].mean(axis=0)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        self.prototypes_ = centers

        # Квантование в целые веса AltAI-2.
        self.weights_int_ = np.clip(np.round(centers * 255), -255, 255).astype(int)

        # Калибровка порога по обучающей выборке.
        train_scores = self._scores(Xn, backend)
        self.threshold_ = float(np.quantile(train_scores, self.contamination))
        return self

    # -- скор (чем выше, тем "нормальнее") --
    def decision_function(self, X, backend=BACKEND_ALTAI):
        return self._scores(self._to_unit(X), backend)

    def predict(self, X, backend=BACKEND_ALTAI):
        """+1 = норма, -1 = аномалия (как в sklearn OneClassSVM)."""
        return np.where(self.decision_function(X, backend) >= self.threshold_, 1, -1)

    def _scores(self, Xn, backend):
        return _run_backend(backend, self.weights_int_, Xn,
                            self.steps, self.activation_threshold, workdir=_HERE)

    # -- сохранение/загрузка --
    def save(self, path):
        np.savez(path, x_offset=self.x_offset_,
                 weights=self.weights_int_, threshold=self.threshold_,
                 steps=self.steps, act=self.activation_threshold)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(n_prototypes=len(d["weights"]), steps=int(d["steps"]),
                activation_threshold=int(d["act"]))
        m.x_offset_ = float(d["x_offset"])
        m.weights_int_, m.threshold_ = d["weights"], float(d["threshold"])
        return m


# --- Демо ---------------------------------------------------------------------

def _roc_auc(scores_pos, scores_neg):
    """AUC вручную: P(score_normal > score_anomaly)."""
    return float((scores_pos[:, None] > scores_neg[None, :]).mean()
                 + 0.5 * (scores_pos[:, None] == scores_neg[None, :]).mean())


def demo():
    rng = np.random.default_rng(0)
    # Нормальный класс — компактное облако в 5-мерном пространстве.
    # Аномалии — разреженные всплески: одна случайная координата зашкаливает.
    mu = np.array([0.35, 0.30, 0.40, 0.25, 0.45])
    X_train = rng.normal(mu, 0.05, size=(300, 5))
    X_test_normal = rng.normal(mu, 0.05, size=(60, 5))
    X_test_anomaly = np.full((60, 5), 0.05) + np.eye(5)[rng.integers(0, 5, 60)] * 0.9

    print("Архитектура SNN:", json.dumps(ARCH, ensure_ascii=False, indent=2))
    model = OneClassSNN(n_prototypes=6, steps=30, activation_threshold=4000,
                        contamination=0.1)
    print("\n[fit] обучение на нормальном классе (калибровка на CPU-бэкенде)...")
    model.fit(X_train)
    print(f"[fit] порог спайкового скора: {model.threshold_:.1f}")

    print("\n[predict] инференс на эмуляторе AltAI-2...")
    s_normal = model.decision_function(X_test_normal)
    s_anomaly = model.decision_function(X_test_anomaly)
    auc = _roc_auc(s_normal, s_anomaly)
    pred = np.concatenate([model.predict(X_test_normal), model.predict(X_test_anomaly)])
    y_true = np.concatenate([np.ones(len(X_test_normal)), -np.ones(len(X_test_anomaly))])
    acc = float((pred == y_true).mean())
    print(f"Эмулятор AltAI-2:  ROC AUC = {auc:.3f},  accuracy = {acc:.3f}")

    # Контроль: эмулятор vs CPU-бэкенд (результаты должны совпадать).
    s_cpu = model.decision_function(np.vstack([X_test_normal, X_test_anomaly]),
                                    backend=BACKEND_CPU)
    s_altai = np.concatenate([s_normal, s_anomaly])
    print(f"Совпадение скоров CPU/AltAI: {np.array_equal(s_cpu, s_altai)} "
          f"(max diff = {np.abs(s_cpu - s_altai).max():.0f})")

    # Опциональный бейзлайн sklearn.
    try:
        from sklearn.svm import OneClassSVM
        from sklearn.metrics import roc_auc_score
        svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(X_train)
        auc_sk = roc_auc_score(
            [1] * len(X_test_normal) + [0] * len(X_test_anomaly),
            np.concatenate([svm.decision_function(X_test_normal),
                            svm.decision_function(X_test_anomaly)]))
        print(f"sklearn OneClassSVM (бейзлайн): ROC AUC = {auc_sk:.3f}")
    except ImportError:
        print("sklearn не установлен — бейзлайн пропущен (pip install scikit-learn)")

    model.save(os.path.join(_HERE, "oneclass_model.npz"))
    print(f"\nМодель сохранена: {os.path.join(_HERE, 'oneclass_model.npz')}")


if __name__ == "__main__":
    sys.exit(demo())
