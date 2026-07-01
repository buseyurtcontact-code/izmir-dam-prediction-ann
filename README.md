# Dam Fill Level Prediction Using Deep Learning

This project predicts dam fill levels 7, 14, and 30 days into the future using meteorological variables, historical dam fill levels, and production-demand data.

The system performs automatic feature engineering, trains deep learning models (GRU, LSTM, or Dense Neural Networks), evaluates prediction performance, and generates visualization reports.

---

## Features

- Multi-horizon prediction (+7, +14, +30 days)
- Automatic feature engineering
- Weather-based predictors
- Lag feature generation
- Seasonal feature generation
- Rolling statistics
- Walk-Forward Expanding Validation
- Baseline train/test evaluation
- Sample weighting using exponential decay
- Early stopping
- Automatic performance visualization

---

## Deep Learning Models

The framework supports three neural network architectures:

- GRU (default)
- LSTM
- Dense Neural Network (MLP)

---

## Input Data

The project expects a merged dataset containing

- Dam fill percentage
- Temperature
- Rainfall
- Humidity
- Wind speed
- Sunshine duration
- Evaporation
- Production/Demand variables

Main dataset:

```
son_merged.csv
```

Feature configuration:

```
feature_config.json
```

---

## Feature Engineering

The following features are automatically generated.

### Weather Features

- 7-day rainfall accumulation
- 30-day rainfall accumulation
- 90-day rainfall accumulation
- 7-day temperature average

### Lag Features

Historical dam levels:

- Lag 7
- Lag 14
- Lag 30
- Lag 60
- Lag 90

### Trend Features

- Daily change
- Weekly change
- Monthly change
- 30-day moving average
- 30-day moving standard deviation
- 180-day moving average
- 365-day moving average
- Trend direction

### Seasonal Features

- Month (sin/cos encoding)
- Day of year (sin/cos encoding)

---

## Training Strategy

Two evaluation strategies are available.

### Baseline

Training data:

Before 2023

Testing data:

After 2023

### Walk Forward Expanding

The model is retrained every six months using an expanding training window.

---

## Performance Metrics

The following metrics are reported:

- RMSE
- MAE
- R² Score

Prediction plots are automatically generated for every dam and prediction horizon.

---

## Output

The project automatically creates

```
results_test2023/
```

including

- Prediction plots
- Residual plots
- Performance summary
- CSV report

---

## Project Structure

```
project/
│
├── son_merged.csv
├── feature_config.json
├── main.py
├── results_test2023/
│     ├── prediction graphs
│     ├── residual graphs
│     └── dam_results_summary.csv
```

---

## Example Command

Train all dams

```bash
python main.py
```

Train selected dams

```bash
python main.py --dams "Tahtali,Urkmez"
```

Train using data after 2015

```bash
python main.py --startyear 2015
```

---

## Workflow

1. Load dataset
2. Clean missing values
3. Generate engineered features
4. Create lag variables
5. Generate seasonal features
6. Split training and testing datasets
7. Scale features
8. Create time sequences
9. Train deep learning model
10. Predict future dam levels
11. Evaluate performance
12. Generate plots
13. Save summary results

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Author

Buse
Department of Statistics
Ege University
