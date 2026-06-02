# Music Genre Classifier

This is a Streamlit app that trains a music genre classifier from `features_30_sec.csv`.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Deploy On Streamlit Community Cloud

1. Push these files to a GitHub repository:
   - `app.py`
   - `requirements.txt`
   - `features_30_sec.csv`
2. Go to https://share.streamlit.io/
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Click **Deploy**.

The app does not need Flask or the `Templates` folder anymore.
