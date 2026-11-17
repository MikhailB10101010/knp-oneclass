# knp-oneclass — One-Class SNN на эмуляторе AltAI-2

Одноклассовый детектор аномалий (нейроморфный аналог One-Class SVM с RBF-ядром),
исполняемый на эмуляторе нейроморфного процессора **AltAI-2** из состава
Kaspersky Neuromorphic Platform 2.0.0.

> Классический `sklearn.svm.OneClassSVM` на нейроморфном процессоре исполняться не может —
> AltAI-2 выполняет только импульсные нейросети. Поэтому здесь реализована SNN с тем же
> функционалом (обучение только на нормальном классе, детекция аномалий) и API в стиле
> sklearn: `.fit()` / `.decision_function()` / `.predict()`.

## Архитектура

```
input (N признаков, rate-coding)
   │  DeltaSynapse, веса — целые [-255, 255] (квантование AltAI-2)
   ▼
hidden (K нейронов AltAILIF — прототипы нормального класса, k-means)
   │
   ▼
score = суммарное число спайков за T шагов;  score < threshold → аномалия
```

- `oneclass_snn.py` — архитектура, обучение (k-means + квантование + калибровка
  порога), предсказание, демо на синтетических данных.
- `cpp/oneclass_runner.cpp` — исполнитель сети на бэкендах KNP
  (CPU для отладки / `libknp-altai-backend.so` — эмулятор AltAI-2).
- `deploy_knp.sh` — развёртывание платформы KNP + эмулятора на Linux (Docker).

## Развёртывание с нуля

```bash
./deploy_knp.sh          # ставит Docker + SDK-образ KNP (эмулятор AltAI-2 внутри)
```

## Сборка и запуск (внутри контейнера KNP)

```bash
~/knp/run_knp.sh                       # shell в контейнере, ~/knp примонтирован как /work

cd /work/oneclass
cmake -S cpp -B build && cmake --build build -j
cp build/oneclass_runner .

python3 oneclass_snn.py                # демо: fit на CPU, инференс на эмуляторе AltAI-2
```

Ожидаемый вывод: архитектура, порог, ROC AUC / accuracy на эмуляторе,
проверка совпадения скоров CPU vs AltAI, опционально бейзлайн sklearn OneClassSVM.

## Использование как библиотеки

```python
from oneclass_snn import OneClassSNN

model = OneClassSNN(n_prototypes=6, steps=30, contamination=0.1)
model.fit(X_train)                     # только «нормальные» образцы
y = model.predict(X_test)              # +1 норма / -1 аномалия, инференс на эмуляторе
model.save("model.npz")
```

## Ограничения

- На эмуляторе AltAI-2 выполняется только инференс (STDP-обучение на AltAI
  в KNP 2.0.0 не поддерживается); fit выполняется аналитически и калибруется на CPU.
- Признаки нормализуются и кодируются частотой спайков (rate coding).
- Требуется Docker-образ `knp-sdk:cuda` (создаётся скриптом deploy_knp.sh).
