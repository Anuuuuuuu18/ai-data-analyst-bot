# Analyst workflow

1. Start in demo mode or upload a CSV, delimited text file with a detectable separator, or Excel workbook. Select the sheet for an Excel upload.
2. Review the original dataset profile: size, column types, missing values, duplicates, and numeric fields.
3. Select numeric/text missing-value strategies, duplicate removal, and whitespace cleanup. The app runs these choices on an in-memory copy and shows a change log.
4. Inspect the cleaned preview. Export the cleaned CSV or Excel workbook only after reviewing the changes.
5. Use the chart controls to compare groups, distributions, numeric relationships, or a measure across an x-axis.
6. Download the analysis report, SQL starter, and pandas starter code. Ask local dataset questions in the chat tab.

The default assistant is deterministic and local. The optional OpenAI setting is separate and needs an API key plus a second consent checkbox. It only provides the first eight cleaned rows and the schema to the API, so treat its answer as sample-based.
