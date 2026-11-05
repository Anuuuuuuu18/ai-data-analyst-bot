# AI Data Analyst Bot

An end-to-end Streamlit app that turns a messy CSV or Excel workbook into a reviewable data-cleaning result, interactive charts, practical findings, and downloadable analysis files.

## Get an outcome in three steps

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `\.venv\Scripts\Activate.ps1`; on macOS/Linux, use `source .venv/bin/activate`. The app opens with a built-in messy sales dataset so the first run produces visible results without an API key or file upload. Use **Upload my file** to work with your own `.csv`, `.tsv`, `.json`, `.xlsx`, or `.xls` file.

## What the app produces

- A shape, column, missing-value, and duplicate profile.
- Side-by-side before/after previews and an itemized cleaning log.
- Downloadable cleaned CSV and Excel workbook.
- A configurable dashboard with 1–6 count, bar, line, scatter, histogram, and KPI tiles, plus a shared filter.
- Downloadable dashboard layouts in JSON that can be loaded again with compatible or different datasets.
- A Markdown and JSON analysis report with cautious next-step recommendations.
- SQLite starter queries and a reproducible pandas analysis script.
- A local question-and-answer assistant for common dataset questions.
- Optional OpenAI Responses API answers, disabled by default and gated by an explicit confirmation. Only the question, column names, and up to eight cleaned sample rows are sent; the entire dataset is never sent by this feature.

## Cloud AI setup (optional)

Local analysis needs no API key and makes no network calls. For optional cloud answers, install the requirements as usual, then set `OPENAI_API_KEY` in the environment or in Streamlit secrets. The default model is configurable through `OPENAI_MODEL` (defaults to `gpt-6-luna`). In the sidebar, enable cloud AI and confirm the sample-data disclosure before sending a question. Never send sensitive data unless you have permission to do so.

For a local secrets file, copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and add your own key. `secrets.toml` is ignored by Git.

## Project structure

```text
app.py                         Streamlit user interface and exports
analyst.py                     Cleaning, profiling, recommendation, SQL, and Q&A logic
sample_data/sales_messy.csv    Fictional demo data with intentional quality issues
docs/                          Method, privacy, and deployment guidance
.streamlit/config.toml         App theme and upload limits
requirements.txt               Pinned runtime dependencies
```

## Data handling and limits

CSV, TSV, JSON, and Excel uploads are read in memory for the current Streamlit session and are not written into the repository. Dashboard JSON exports contain chart and filter settings, never uploaded rows. Cloud answers are off by default. If enabled and confirmed, a small sample and the question are sent to the configured OpenAI API. The app does not validate source accuracy, understand an organization's business rules, or prove causation. Treat correlations, unusual-value flags, and generated recommendations as leads for human review. See [data and privacy notes](docs/DATA_AND_PRIVACY.md).

## License

MIT. See [LICENSE](LICENSE).

