# Run and deploy

## Local

Use Python 3.10 or newer. Create and activate a virtual environment, install `requirements.txt`, and run `streamlit run app.py`. The built-in demo dataset is available on first launch. Use `OPENAI_API_KEY` or Streamlit secrets only if you choose to enable cloud answers.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app using this repository, its `main` branch, and `app.py`.
3. Add dependencies from `requirements.txt` (the default dependency file).
4. If cloud answers are wanted, add `OPENAI_API_KEY` under the app's Secrets settings. Do not commit the key to GitHub.
5. Leave the app's Cloud AI checkbox off unless the user has reviewed and approved the sample-data disclosure.

The app needs no credentials for local analysis, demo mode, charts, cleaning, or downloads. A public deployment URL lets others use the app, so review your data-handling and access needs before deploying with real uploaded datasets.
