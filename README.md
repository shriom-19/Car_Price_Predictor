```markdown
# Car_Price_Predictor

A simple repository and demo app for predicting used car selling prices. This README is written point-wise and includes setup, usage, dataset expectations, and the exact models used in the project.

---

1) Project overview
- Goal: Predict the selling price of used cars from features like year, kilometers driven, brand, fuel, seller, transmission, owner, etc.
- Two main components present in the repo:
  - A training pipeline (scikit-learn) to build, evaluate, and save a pipeline (preprocessing + regressor).
  - A Gradio demo app that loads a pre-trained pickled model (`carmodel.pkl`) and serves a UI for single-record predictions.

2) Files & suggested structure
- README.md (this file)
- requirements.txt
- .gitignore
- data/
  - cars.csv (user-provided dataset)
- models/
  - car_price_pipeline.joblib (output after training — created by `src/train_model.py`)
  - carmodel.pkl (pre-trained pickled model used by Gradio; included in repo or uploaded separately)
- src/
  - data_preprocessing.py
  - train_model.py
  - predict.py
- car_price_predictor.py (Gradio UI — already in repo)

3) Models used in this project (exact)
- RandomForestRegressor (scikit-learn)
  - Where: `src/train_model.py` (the training script builds and uses this).
  - Instantiation: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
  - Purpose: Primary training/regression model in the provided training pipeline combined with preprocessing (ColumnTransformer).
  - Saved format from training script: joblib pipeline (e.g., `models/car_price_pipeline.joblib`).

- Pickled model file: carmodel.pkl (pre-trained model used by Gradio)
  - Where: `car_price_predictor.py` loads `carmodel.pkl` using pickle and calls `model.predict` on preprocessed input.
  - What it is: a pre-trained model object (the repository code does not show the estimator class inside the pickle). In practice it is typically a scikit-learn regressor (could be RandomForestRegressor, GradientBoostingRegressor, LinearRegression, etc.).
  - Notes: The Gradio app applies custom transforms:
    - age = 2025 - year
    - km = log1p(km)
    - inv = expm1 on the model output (indicates target was log-transformed during training or prediction post-processed)
  - How to inspect locally:
    - python - <<'PY'
      import pickle
      with open('carmodel.pkl','rb') as f:
          m = pickle.load(f)
      print('Model type:', type(m))
      print(m)
      PY

4) Expected dataset (CSV)
- Example filename: data/cars.csv
- Example numeric columns (update as needed): year, km_driven (or mileage), engine_size, tax, mpg
- Example categorical columns: brand (or make), model, fuel, transmission, seller, owner
- Target column: price (or use --target-col to specify)
- Important: Ensure feature names match those in the training script (`NUM_FEATURES` and `CAT_FEATURES` in `src/train_model.py`).

5) Quick setup (point-wise)
- Create virtual environment:
  - python -m venv .venv
  - source .venv/bin/activate  (macOS / Linux) or .venv\Scripts\activate (Windows)
- Install dependencies:
  - pip install -r requirements.txt

6) Train (point-wise)
- Place dataset at data/cars.csv (or pass path via --data-path)
- Run:
  - python src/train_model.py --data-path data/cars.csv --target-col price --output-model models/car_price_pipeline.joblib
- Output:
  - Validation metrics printed (MAE, RMSE, R2) and a saved joblib pipeline at models/car_price_pipeline.joblib

7) Predict (point-wise)
- Using the saved joblib pipeline:
  - Single JSON (CLI):
    python src/predict.py --model models/car_price_pipeline.joblib --input '{"year":2016,"km_driven":35000,"brand":"Maruti","fuel":"Petrol","transmission":"Manual","owner":"First Owner"}'
  - Batch CSV:
    python src/predict.py --model models/car_price_pipeline.joblib --input-csv data/new_cars.csv --output-csv data/new_cars_with_preds.csv
- Using Gradio UI:
  - Ensure `carmodel.pkl` is present (or replace it with your trained model saved as a pickle that matches Gradio's expected input schema).
  - Run:
    python car_price_predictor.py
  - The UI expects inputs like model year, kilometers driven, fuel, seller, transmission, owner, brand.

8) Preprocessing notes
- Training pipeline (in `src/train_model.py`) uses:
  - Numeric transformer: SimpleImputer(strategy="median") + StandardScaler()
  - Categorical transformer: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
  - ColumnTransformer to combine numeric and categorical pipelines
  - Pipeline: preprocessor -> RandomForestRegressor
- Gradio app applies specific transforms:
  - age = 2025 - year (converts model year to car age)
  - km = log1p(km_driven)
  - final predicted price = expm1(model_output) (inverse of log transform)
- Important: Use the same preprocessing during training and inference. If using the Gradio model, ensure your pickled model expects these transforms.

9) How to check model saved by train script
- If you trained and saved the joblib pipeline:
  - python - <<'PY'
    import joblib
    p = joblib.load('models/car_price_pipeline.joblib')
    print('Pipeline steps:', list(p.named_steps.keys()))
    print('Model step class:', type(p.named_steps['model']))
    print(p.named_steps['model'])
    PY

10) Recommendations & next steps
- If `brand` or `model` has high cardinality, consider target-encoding or frequency-encoding to reduce dimension explosion from one-hot encoding.
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for better performance.
- Replace pickle-based model in Gradio with the full pipeline saved via joblib so the app uses the same preprocessing as training.
- Add unit tests and CI (GitHub Actions).
- Consider adding a small FastAPI app for production serving.

11) Contribution & license (point-wise)
- Contribute: Fork → branch → PR; include tests or a demo notebook if adding features.
- License: Add a LICENSE file (MIT recommended).

---

If you want, I can:
- generate a small synthetic `data/cars.csv` to test training and prediction,
- update the Gradio app to load the joblib pipeline instead of a raw pickle, or
- add an example command to convert a trained joblib pipeline into the pickled file the Gradio app expects.

```
