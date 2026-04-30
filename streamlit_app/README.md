# Streamlit App

This folder contains a Streamlit interface for browsing experiment logs,
TCF evaluation results, and related rulebase resources.

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

The left sidebar includes metadata filters and quality filters. Right now the
quality section can filter and sort experiments by TCF so you can quickly find
good, mixed, or bad explanations.

For logs that contain TCF evaluation data, the app shows the newest TCF
score directly under the generated explanation and exposes the full
question-by-question judge output in a collapsed expander. If a log contains
multiple evaluations, the expander includes a selector for older runs.
