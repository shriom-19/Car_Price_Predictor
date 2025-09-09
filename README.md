# Car_Price_Predictor

A concise, easy-to-use repository for predicting used car prices using a scikit-learn pipeline. This README gives point-wise details, usage examples, and includes ready-to-run scripts for preprocessing, training, evaluating, and serving predictions.

1) Project overview
- Goal: Predict the sale price of used cars based on features such as year, mileage, make, model, fuel type, transmission, engine size, etc.
- Approach: Single scikit-learn Pipeline which handles preprocessing (encoding + scaling) and model training (RandomForestRegressor by default).
- Outputs: Trained pipeline saved to `models/car_price_pipeline.joblib`, training logs printed to console, evaluation metrics.

2) Key features (point-wise)
- End-to-end pipeline (preprocessing + model) so inference is simple.
- Handles categorical and numerical features via ColumnTransformer.
- Provides reproducible scripts:
  - `src/data_preprocessing.py` (helper functions)
  - `src/train_model.py` (train + evaluate and save pipeline)
  - `src/predict.py` (load pipeline and predict from CSV or single JSON/CLI input)
- Easy to extend to other regressors or hyperparameter search.
- Minimal dependencies and clear instructions.

3) Expected dataset (point-wise)
- CSV format (example file name: `data/cars.csv`).
- Required columns (customize if necessary):
  - Numerical: `year`, `mileage`, `engine_size`, `tax`, `mpg` (example numeric fields)
  - Categorical: `make`, `model`, `fuel_type`, `transmission`, `seller_type` (example categorical fields)
  - Target: `price`
- Note: If your dataset uses different column names, update `NUM_FEATURES`, `CAT_FEATURES`, and the `target_col` variable in `src/train_model.py`.

4) File structure suggested (point-wise)
- README.md (this file)
- requirements.txt
- .gitignore
- data/
  - cars.csv (user-provided dataset)
- models/
  - car_price_pipeline.joblib (output after training)
- src/
  - data_preprocessing.py
  - train_model.py
  - predict.py

5) Setup & installation (point-wise)
- Create a virtualenv (recommended):
  - python -m venv .venv
  - source .venv/bin/activate  (macOS / Linux)
  - .venv\Scripts\activate     (Windows)
- Install dependencies:
  - pip install -r requirements.txt

6) Quick start — train (point-wise)
- Place your dataset at `data/cars.csv` or pass path via --data-path.
- Run:
  - python src/train_model.py --data-path data/cars.csv --target-col price --output-model models/car_price_pipeline.joblib
- Output: prints metrics (MAE, RMSE, R2) and saves pipeline.

7) Quick start — predict (point-wise)
- Single prediction via CLI:
  - python src/predict.py --model models/car_price_pipeline.joblib --input '{"year":2016,"mileage":35000,"make":"Ford","model":"Focus","fuel_type":"Petrol","transmission":"Manual","engine_size":1.6}'
- Bulk prediction from CSV:
  - python src/predict.py --model models/car_price_pipeline.joblib --input-csv data/new_cars.csv --output-csv data/new_cars_with_preds.csv

8) Training details (point-wise)
- Model: RandomForestRegressor (default). Change model or hyperparams inside `train_model.py`.
- Cross-validation / hyperparameter tuning: not included by default but simple to add (GridSearchCV).
- Reproducibility: random_state set for train/test split and model.

9) Inference details (point-wise)
- The saved pipeline includes preprocessing (so you do not need to separately transform inputs before calling predict).
- Provide same feature names as during training; missing columns will raise an error (handle by modifying the code to provide defaults).

10) Contribution guidelines (point-wise)
- Fork → branch → PR with tests or a short demo notebook.
- Add issues for feature requests / bug reports.
- Keep code style consistent with PEP8.

11) License & contact (point-wise)
- Add an appropriate LICENSE file (MIT recommended).
- Contact: repo owner (GitHub) — @shriom-19

12) Next steps / customization ideas (point-wise)
- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Add a small FastAPI wrapper for real-time predictions.
- Add unit tests and CI (GitHub Actions).
- Add example Jupyter notebook demonstrating dataset EDA and model explainability (SHAP).

Code examples and helper scripts are included in the `src/` folder. Adjust the feature lists in the scripts to match your dataset's column names.
