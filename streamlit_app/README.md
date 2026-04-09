# Streamlit App

This folder contains a Streamlit interface for browsing experiment logs and
related rulebase resources.

## Run

From the repository root:

```bash
streamlit run streamlit_app/app.py
```

Or with the project virtual environment:

```bash
./xai/bin/streamlit run streamlit_app/app.py
```

If your environment resolves imports differently, running from inside the app
folder also works:

```bash
cd streamlit_app
streamlit run app.py
```

The app reads:

- experiment logs from `./logs`
- raw traces from `./traces`
- rulebase descriptions and plots from `./rulebases`
