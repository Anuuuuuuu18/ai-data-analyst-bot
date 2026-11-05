# Data, privacy, and interpretation

## Local analysis

CSV and Excel uploads are consumed in memory by the Streamlit session. The app does not save uploads to the repository or a local output folder. Download buttons create files from the current in-memory result. Formula-like text is prefixed in CSV and Excel exports so spreadsheet software displays it as text. The demo sales dataset is fictional.

## Optional cloud answers

Cloud AI is disabled by default. A user must enable it and check the consent box before a chat question can call the OpenAI Responses API. The request includes the question, column names, dimensions, and at most the first eight rows after selected cleaning steps. It does not include the complete dataset or prior chat transcript. `store=False` is passed to the Responses API. Do not use this option for sensitive or restricted data without appropriate authorization and review of the provider's current terms.

The API key is read from the `OPENAI_API_KEY` environment variable or Streamlit secrets. The app never displays the key in the interface or includes it in an export. `.streamlit/secrets.toml` and `.env` are excluded from Git.

## Analysis limits

- Missing-value filling is an explicit convenience choice, not evidence that a replacement is correct.
- Exact duplicate rows may be valid repeated transactions; confirm the intended business key.
- The IQR screen is a heuristic for review, not an error detector.
- Correlation measures linear association among numeric columns. It does not establish cause.
- Recommendations are generated from simple rules and do not encode company-specific definitions.
- Review the source, units, time period, and definitions before acting on an insight.

## Demo data

`sample_data/sales_messy.csv` is fictional. It intentionally includes blank values, inconsistent whitespace, and an exact repeated row to demonstrate the review workflow. It is not based on a real business or customer.
