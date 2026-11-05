# Analyst workflow

1. Start in demo mode or upload delimited text (CSV/TSV/TXT), JSON/JSONL, Parquet, a PDF, or an Excel/OpenDocument workbook. Select the sheet for a workbook upload. PDFs with tables or selectable text are extracted directly; scanned pages use English OCR.
2. Review the original dataset profile: size, column types, missing values, duplicates, and numeric fields.
3. Select numeric/text missing-value strategies, duplicate removal, and whitespace cleanup. The app runs these choices on an in-memory copy and shows a change log.
4. Inspect the cleaned preview. Export the cleaned CSV or Excel workbook only after reviewing the changes.
5. Open **My Dashboard**, choose 1–6 count, bar, line, scatter, histogram, or metric tiles, and apply a shared filter to the whole dashboard. Download the layout as JSON to reuse it in another session or with another dataset; only settings are saved.
6. Download the analysis report, SQL starter, and pandas starter code. Ask local dataset questions in the chat tab.

The default assistant is deterministic and local. The optional OpenAI setting is separate and needs an API key plus a second consent checkbox. It only provides the first eight cleaned rows and the schema to the API, so treat its answer as sample-based.

