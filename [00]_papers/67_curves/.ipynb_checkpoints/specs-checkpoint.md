# Специфікація ПЗ  

**Проєкт:** Автоматичне формування ієрархічної структури класів для розпізнавання об’єктів
**Версія:** 1.0‑alpha (23.07.2025)
**Мова реалізації:** Python 3.11
**Цільова платформа:** Linux ×86-64 / ARM64, Windows 10/11 ×86-64, CUDA ≥ 12.1, cuDNN ≥ 9

---

## 1. Загальний огляд  

Програмне забезпечення (ПЗ) реалізує повний конвеєр (pipeline) із 8 етапів, описаний у науково‑технічному звіті. 
Конвеєр приймає відеопотік або статичний набір зображень із БПЛА, деталізує об’єкти до багаторівневої ієрархії та експортує структуру (моделі + конфіг) у форматах ONNX / TorchScript / YAML. 
Весь код відповідає PEP 8, використовує статичні типи за PEP 484, модульні тести для кожної суттєвої логіки та CI на GitHub Actions. 

---

## 2. Терміни й скорочення  

* **ROI** — Region of Interest, прямокутник навколо об’єкта. 
* **Back‑bone B** — CNN‑мережа CSPDarknet‑53, попередньо‑натренована на COCO. 
* **Agglo‑кластеризація** — AgglomerativeClustering із linkage = 'ward' (scikit‑learn). 
* **Hydra** — система конфігурацій, що дозволяє комбінувати YAML‑файли. 
* **MLflow** — сервіс для трекінгу експериментів. 

---

## 3. Нефункціональні вимоги  

| Категорія | Вимога |
|-----------|--------|
| **Якість коду** | Дотримання PEP 8 / PEP 257; обов’язкові type hints (PEP 484). |
| **Продуктивність** | ≥ 30 FPS на NVIDIA RTX 3060; ≥ 15 FPS на Jetson AGX Orin. |
| **Пам’ять** | Пік < 6 GB GPU‑RAM у режимі inference. |
| **Відтворюваність** | Фіксований random seed = 2025; MLflow‑run містить git‑hash та параметри Hydra. |
| **Безперервна інтеграція** | Автоматичне тестування, лінт та build ONNX у GitHub Actions. |
| **Ліцензії** | Всі залежності — Apache‑2.0 / MIT. |

---

## 4. Функціональні вимоги  

| ID | Опис |
|----|------|
| FR‑1 | Завантажити датасет у форматі COCO‑JSON або YOLOv5 TXT. |
| FR‑2 | Виконати первинне детектування ROI детектором YOLOv11‑m. |
| FR‑3 | Витягнути ознаки $f\_{ij}=B(r\_{ij})$ та зберегти у HDF5. |
| FR‑4 | Обчислити центроїди $\mu\_l$ та матрицю $\Omega$. |
| FR‑5 | Побудувати матрицю $M$ (формула 3′) та дендрограму $T$. |
| FR‑6 | Згенерувати рівні $L\_k$ за умовою (4′). |
| FR‑7 | Для кожного рівня вибрати $(D\_k,F\_k,C\_k)$ за деревом рішень. |
| FR‑8 | Навчити моделі рівнів 1…K, логуючи в MLflow. |
| FR‑9 | Калібрувати $p^\*_{k}$, виконати FPS/латентність‑тест. |
| FR‑10 | Експортувати моделі в ONNX і TorchScript; створити YAML‑конфіг. |

---

## 5. Архітектура  

### 5.1 Структура каталогів

```

auto\_hierarchy/
├─ auto\_hierarchy/
│   ├─ data/              # завантаження та трансформації
│   ├─ features/          # витяг ознак
│   ├─ clustering/        # кроки 3–4
│   ├─ modeling/          # вибір і навчання D\_k, C\_k
│   ├─ evaluation/        # калібрування, FPS‑тест
│   ├─ export/            # ONNX + YAML
│   ├─ utils/
│   └─ **main**.py
├─ configs/               # Hydra YAML
├─ tests/                 # pytest
├─ requirements.txt
└─ pyproject.toml

````

### 5.2 Основні зовнішні бібліотеки

| Бібліотека | Версія | Причина |
|------------|--------|---------|
| **PyTorch** 2.3 | DL‑фреймворк |
| **PyTorch Lightning** 2.3 | Швидка тренувальна петля |
| **scikit‑learn** 1.7 | AgglomerativeClustering |
| **Ultralytics YOLO** v8.2 | Детектори YOLOv11‑m (форк v8) |
| **rt‑detr** 0.1‑dev | Попередня реалізація RT‑DETR‑Tiny |
| **Hydra‑core** 1.4 | Конфіги |
| **MLflow** 2.11 | Трекінг |

---

## 6. Детальна декомпозиція модулів  

### 6.1 `data/`

| Клас / Функція | Обов’язки |
|----------------|-----------|
| `DatasetLoader` | Читає COCO/YOLO, робить train/val/test split. |
| `ROICropper` | Викликає детектор YOLOv11‑m; обрізає ROI. |
| `HDF5Writer` | Сериалізує $\{r\_{ij},y\_{ij}\}$ → HDF5. |

### 6.2 `features/`

* `FeatureExtractor`: обгортка над back‑bone `B`; кешування на диску. 
* Використовує Torch `torch.compile` для оптимізації.

### 6.3 `clustering/`

* `CentroidCalculator`: обчислює $\mu\_l$ (формула 1). 
* `ConfusionMatrix`: тренує `C_flat`, рахує $\Omega`. 
* `DistanceMatrixBuilder`: нормує метрики і формує $M$ (формула 3′). 
* `AggloBuilder`: створює дендрограму за `sklearn.cluster.AgglomerativeClustering` та пост‑процесингом виконує умову 4′.

### 6.4 `modeling/`

| Компонент | Вміст |
|-----------|-------|
| `ModelSelector` | Реалізує дерево рішень (див. § 1.5.2) в один метод `select(level_stats)`. |
| `LevelTrainer` | Абстракція над тренуванням `(D_k,F_k,C_k)` у Lightning; зберігає чекпоінти. |
| `SuperResolutionWrapper` | ESRGAN ×2 для випадку $\bar{s}_k<16$. |

### 6.5 `evaluation/`

* `LatencyTester`: вимірює T\_k, FPS та GPU / CPU utilization. 
* `ThresholdCalibrator`: шукає $p^\*_{k}$ через ROC‑криву (Youden J). 
* `EnergyProfiler`: читає `nvidia-smi --query-gpu=power.draw`. 

### 6.6 `export/`

* `ONNXExporter`, `TorchScriptExporter`: експортують та валідують моделі. 
* `YamlWriter`: збирає дерева рівнів, пороги, шляхи до ваг.

---

## 7. Алгоритмічні деталі  

### 7.1 Побудова матриці **$M$**

1. **Геометричну** складову нормуємо від 0 до 1. 
2. **Конфузійну** — симетризуємо $(\Omega_{pq}+\Omega_{qp})$ та теж нормуємо. 
3. Комбінуємо за $\lambda$ і зберігаємо у NumPy `float32` матрицю. 

### 7.2 Agglomerative Clustering

```python
agg = AgglomerativeClustering(
        linkage="ward",
        affinity="euclidean",
        distance_threshold=delta_min,
        n_clusters=None,
        compute_full_tree=True
)
labels = agg.fit_predict(M_sym)
````

### 7.3 Умова розрізання (4′)

Рекурсивно обходимо дерево: якщо кластер S задовольняє (4′) — фіксуємо як лист, інакше ділимо на дочірні.

### 7.4 Вибір моделей

Алгоритм 1 (дерево рішень) гарантовано детермінований; усі зміни моделі \$D\_k\$ ведуть до повторної побудови `LatentGraph`, що зберігає відповідність “клас → шлях моделі”.

---

## 8. Інтерфейси та API

### 8.1 CLI‐утиліта `auto-hier`

```
auto-hier \
  --data_path /data/fecl \
  --config configs/base.yaml \
  override.model_selector.lambda=0.65 \
  run
```

*Команди:* `run`, `resume`, `export`, `eval`.
Аргументи обробляє **Hydra**; кожен параметр може бути перезаписаний у CLI.

### 8.2 Python SDK

```python
from auto_hierarchy import Pipeline, PipelineConfig

cfg = PipelineConfig.parse_yaml("configs/base.yaml")
pl = Pipeline(cfg)
pl.fit()
pl.export(output_dir="artifacts/")
```

---

## 9. Документація й стиль коду

* **Док‑строки:** Google‑style, перевіряються `pydocstyle`.
* **Лінт + форматер:** `ruff`, `black` зі 120 sym/рядок.
* **Типізація:** `mypy --strict`.
* **Sphinx** для HTML‑доків, deploy у GitHub Pages.

---

## 10. Тестування

| Рівень      | Інструмент                                    | Покриття           |
| ----------- | --------------------------------------------- | ------------------ |
| Unit        | `pytest` + `hypothesis`                       | ≥ 90 %             |
| Integration | Lightning `Trainer(fast_dev_run)`             | кожен етап         |
| E2E         | `pytest` + `torch.cuda.amp` на 50 зображеннях | вихідні YAML + FPS |

CI workflow:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: {python-version: ["3.11"]}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ matrix.python-version }}}
      - run: pip install -r requirements.txt
      - run: pytest -q
```

---

## 11. План робіт для одного розробника

| Тиждень | Задачі                                            | Артефакти |
| ------- | ------------------------------------------------- | --------- |
| 1       | Проектування репо, `DatasetLoader`, `ROICropper`  | PR #1     |
| 2       | `FeatureExtractor`, кешинг HDF5                   | PR #2     |
| 3       | `CentroidCalculator`, `ConfusionMatrix`, тест‐док | PR #3     |
| 4       | `DistanceMatrixBuilder`, `AggloBuilder`           | PR #4     |
| 5       | `ModelSelector`, підготовка YAML‑конфігів         | PR #5     |
| 6       | `LevelTrainer` (Lightning), MLflow інтеграція     | PR #6     |
| 7       | `LatencyTester`, `ThresholdCalibrator`            | PR #7     |
| 8       | `ONNXExporter`, `YamlWriter`, end‑to‑end скрипт   | PR #8     |
| 9       | Пакування, Sphinx docs, Dockerfile                | реліз 0.9 |
| 10      | Високорівнева оптимізація, профайлинг, реліз 1.0  | реліз 1.0 |

---

## 12. Ризики й пом’якшення

| Ризик                          | Вплив    | Мітiгація                          |
| ------------------------------ | -------- | ---------------------------------- |
| Недоступність RT‑DETR‑Tiny ваг | Середній | Тримати fallback YOLOv11‑m         |
| Перевитрата GPU‑RAM            | Високий  | Mixed‑precision (`torch.cuda.amp`) |
| Нестабільність FPS на Jetson   | Середній | TensorRT INT8 квантизація          |

---

## 13. Ліцензія та відповідність

Увесь вихідний код під Apache 2.0; сторонні моделі — перевірено сумісність (Ultralytics YOLO GPL‑посилань немає).
Документація включає Third‑party licenses notice.

---

## 14. Додатки

* **A.** Приклад HDF5‑схеми.
* **B.** Повний список Hyperparameters і їх діапазонів для Sweep.
* **C.** Приклад YAML‑конфігурації Hydra.

---

