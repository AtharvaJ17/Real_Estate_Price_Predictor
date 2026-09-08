# Gurgaon House Price Predictor

A simple website with a form where you enter house details, and a Flask backend
predicts the price using your trained model.

## Folder structure

```
gurgaon_price_predictor/
│
├── app.py                 <- Flask backend (run this to start the site)
├── train_model.py         <- Your original training script
├── requirements.txt       <- List of libraries needed
├── gurgaon_10k.csv        <- YOU NEED TO ADD THIS (your dataset)
│
├── templates/
│   └── index.html         <- The web page (form + result)
│
└── static/
    └── style.css           <- The styling
```

## Step 1: Install the libraries

Open a terminal inside the `gurgaon_price_predictor` folder and run:

```
pip install -r requirements.txt
```

## Step 2: Train the model (only needs to be done once)

1. Put your `gurgaon_10k.csv` file in this same folder.
2. Run:

```
python train_model.py
```

This will create a file called `gurgaon_price_model_pipeline.pkl` in the same
folder. That file is your trained model — the website loads it every time it
starts.

## Step 3: Run the website

```
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Fill in the form and click "Predict Price" to see the estimated price.

## ⚠️ Important — please check this before using it

I don't have access to your actual `gurgaon_10k.csv`, so I could not see the
real category values your model was trained on. I filled in the dropdown
options in `templates/index.html` with reasonable guesses:

- **Property Type**: Flat, Independent Floor, Independent House, Villa
- **Ownership Type**: Freehold, Leasehold, Co-operative Society, Power of Attorney
- **Facing Direction**: numbered 1–8 standing in for North/South/East/West etc.

**Before you use the site for real, open your CSV and check the exact,
unique values in the `PROPERTY_TYPE`, `OWNTYPE`, and `FACING` columns**
(in Python: `df['PROPERTY_TYPE'].unique()`, etc.). If they don't match what
I guessed, just edit the `<option value="...">` lines in
`templates/index.html` — that's the only place you'll need to change.

Also double check what number `SECTOR_NUM` and `FACING` were actually stored
as in your data (e.g. is Sector 45 stored as `45` or `"Sector 45"`?), since
the form currently sends plain numbers for both.

## How it works (in plain terms)

1. `train_model.py` reads your CSV, cleans it up, and trains an XGBoost model
   inside a scikit-learn "pipeline" (this pipeline also handles converting
   text categories like Property Type into numbers the model understands).
   It saves the finished pipeline to a `.pkl` file.
2. `app.py` is a small Flask server. When it starts, it loads that `.pkl`
   file into memory once. Every time someone submits the form, it:
   - reads the values from the form,
   - puts them into a table (DataFrame) with the same column names used
     during training,
   - asks the model to predict a price,
   - converts the result back from "log price" to a normal rupee amount,
   - shows it on the page.
3. `index.html` is just a plain HTML form. `style.css` makes it look clean
   and white/professional instead of using default browser styling.

No JavaScript is used — the page reloads on submit, which is the simplest
way to do it for a first project.
