# GSK Sales by Indication Challenge

## Overview
Hospitals buy a drug from GSK without specifying which disease it is being used for. The same drug can be prescribed for:

- Indication A
- Indication B
- Indication C

If a hospital orders `1,000` units, GSK does not know whether the split was:

- `800 / 100 / 100`
- `500 / 300 / 200`
- or any other combination

This matters because GSK has separate sales teams for each indication. If total hospital sales cannot be allocated back to the underlying indication, it becomes difficult to evaluate which team is actually driving performance.

## The Goal
Build a model that predicts the hospital-level sales split:

- `X%` for Indication A
- `Y%` for Indication B
- `Z%` for Indication C

The model should use hospital sales and commercial activity data to estimate how the total sales volume is distributed across the three indications.

## Available Data
The dataset contains hospital-level information across 6 months for about 100 hospitals.

| Data | What it tells you |
|---|---|
| Monthly sales (units) | How much each hospital bought each month |
| Touchpoints per indication | How many times the sales team contacted the hospital about A, B, or C |
| HCPs per indication | How many doctors at that hospital were contacted for A, B, or C |
| Actual indication split (training sample) | The true A/B/C split used as the label |

## Expected Deliverable
An analytical model that takes:

- sales numbers
- touchpoints
- HCP activity

and outputs the estimated split:

- `X%` Indication A
- `Y%` Indication B
- `Z%` Indication C

for hospitals where the true split is unknown.

## Why This Problem Is Hard
- The labelled sample is small.
- The outputs are compositional, so the predicted percentages must sum to `100%`.
- Indication A dominates on average, so a naive model can look good while still being unhelpful.
- Hospitals vary widely in size, from very low volume to very high volume.

## In One Sentence
Given how much a hospital buys and how the sales team interacts with that hospital’s doctors, predict what share of the drug is being used for each indication.

## Repository Structure
- [sales.xlsx](/Users/sm_aswin21/Desktop/gsk/sales.xlsx): source dataset
- [gsk_model.py](/Users/sm_aswin21/Desktop/gsk/gsk_model.py): data preparation, feature engineering, model training, evaluation, and artifact export
- [streamlit_app.py](/Users/sm_aswin21/Desktop/gsk/streamlit_app.py): Streamlit application for the challenge demo
- [GSK Indication Split Model.ipynb](/Users/sm_aswin21/Desktop/gsk/GSK%20Indication%20Split%20Model.ipynb): notebook for the core modelling workflow
- [Notebook_Detailed.ipynb](/Users/sm_aswin21/Desktop/gsk/Notebook_Detailed.ipynb): additional notebook exploration / detailed analysis
- [requirements.txt](/Users/sm_aswin21/Desktop/gsk/requirements.txt): Python dependencies
- [artifacts/gsk_multinomial_model.joblib](/Users/sm_aswin21/Desktop/gsk/artifacts/gsk_multinomial_model.joblib): exported trained model artifact
- [artifacts/gsk_model_config.json](/Users/sm_aswin21/Desktop/gsk/artifacts/gsk_model_config.json): exported model configuration

## Modelling Approach
This project uses interpretable statistical models instead of black-box tree ensembles.

### Main model
- `Multinomial Logistic Regression`

Why this model:
- It is easier to explain to business users.
- It naturally produces a valid 3-way probability split.
- It respects the compositional nature of the output.

### Benchmark model
- `ALR OLS Benchmark`

This benchmark uses additive log-ratio regression as a simpler statistical comparison model.

## How To Run

### 1. Create and activate an environment
Example with `venv`:

```bash
cd /Users/sm_aswin21/Desktop/gsk
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Run The Notebook Workflow
You can use either notebook depending on how much detail you want.

### Option A: Core modelling notebook
```bash
cd /Users/sm_aswin21/Desktop/gsk
jupyter notebook "GSK Indication Split Model.ipynb"
```

### Option B: Detailed notebook
```bash
cd /Users/sm_aswin21/Desktop/gsk
jupyter notebook "Notebook_Detailed.ipynb"
```

If you prefer JupyterLab:

```bash
cd /Users/sm_aswin21/Desktop/gsk
jupyter lab
```

Then open the notebook from the Jupyter interface.

## Run The Training Script
To generate or refresh the exported model artifacts:

```bash
cd /Users/sm_aswin21/Desktop/gsk
python gsk_model.py
```

This will:
- load `sales.xlsx`
- build the hospital-level modelling table
- train the multinomial model
- compare against the benchmark model
- export the reusable artifacts under `artifacts/`

## Run The Streamlit App
To launch the challenge demo app:

```bash
cd /Users/sm_aswin21/Desktop/gsk
streamlit run streamlit_app.py
```

The app includes:
- an `Executive Summary` view
- a `Calculator` view
- a sidebar with hospital inputs
- a model dropdown to switch between the two supported models
- prediction plots for the submitted hospital scenario

## Deployment Note
If you deploy this app on Streamlit Cloud, make sure the following are present in the repository:

- `requirements.txt`
- `sales.xlsx`
- `artifacts/`
- `streamlit_app.py`
- `gsk_model.py`

If artifacts are missing, regenerate them locally with:

```bash
python gsk_model.py
```

and commit the updated files before deploying.

## Output
The final output of the project is a hospital-level analytical model that estimates how total sales break down across:

- Indication A
- Indication B
- Indication C

This makes it easier to interpret commercial effectiveness by disease area rather than only looking at aggregate hospital sales.
