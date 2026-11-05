"""AI Data Analyst Bot: turn a CSV or workbook into a reviewable analysis."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from analyst import answer_question, clean_data, profile_data, python_starter, recommendations, spreadsheet_safe_copy, sql_starter


APP_DIR = Path(__file__).parent
DEMO_FILE = APP_DIR / "sample_data" / "sales_messy.csv"

st.set_page_config(page_title="AI Data Analyst Bot", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read common CSV encodings and let the Python CSV reader infer delimiters."""
    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="cp1252")


@st.cache_data(show_spinner=False)
def read_excel_bytes(content: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def read_json_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_json(io.BytesIO(content))
    except ValueError:
        return pd.read_json(io.BytesIO(content), lines=True)


@st.cache_data(show_spinner=False)
def read_parquet_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def read_pdf_bytes(content: bytes) -> pd.DataFrame:
    """Extract PDF tables/text, using OCR for scanned pages when available."""
    import pdfplumber

    table_records: list[dict[str, Any]] = []
    page_text: list[dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if len(pdf.pages) > 1_000:
            raise ValueError("This PDF has more than 1,000 pages. Split it into smaller files and upload one part at a time.")
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for table_number, table in enumerate(tables, start=1):
                rows = [[cell.strip() if isinstance(cell, str) else cell for cell in row] for row in table]
                rows = [row for row in rows if any(cell is not None and str(cell).strip() for cell in row)]
                if not rows:
                    continue
                header = [str(cell).strip() if cell is not None and str(cell).strip() else f"Column {index + 1}" for index, cell in enumerate(rows[0])]
                header = [f"{name} ({index + 1})" if header[:index].count(name) else name for index, name in enumerate(header)]
                for row in rows[1:]:
                    values = (row + [None] * len(header))[: len(header)]
                    table_records.append({**dict(zip(header, values)), "source_page": page_number, "source_table": table_number})
            if not tables:
                text = page.extract_text() or ""
                if text.strip():
                    page_text.append({"source_page": page_number, "page_text": text.strip()})
    if table_records:
        return pd.DataFrame(table_records)
    if page_text:
        return pd.DataFrame(page_text)
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ValueError("This PDF is scanned. OCR support is unavailable in this environment; install PyMuPDF, pytesseract, and Tesseract OCR, or upload a searchable PDF.") from exc
    if not shutil.which("tesseract"):
        raise ValueError("This PDF is scanned, but the Tesseract OCR engine is not installed. Upload a searchable PDF or install Tesseract OCR.")
    ocr_records: list[dict[str, Any]] = []
    with fitz.open(stream=content, filetype="pdf") as pdf:
        if len(pdf) > 100:
            raise ValueError("This scanned PDF has more than 100 pages. Split it into smaller files and upload one part at a time.")
        for page_number, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            text = pytesseract.image_to_string(image, lang="eng").strip()
            if text:
                ocr_records.append({"source_page": page_number, "page_text": text})
    if ocr_records:
        return pd.DataFrame(ocr_records)
    raise ValueError("OCR found no readable text. Check the scan quality or upload a searchable PDF.")


def openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def make_chart(
    frame: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None,
    category_column: str | None,
    aggregation: str,
) -> tuple[alt.Chart, str]:
    source_rows = len(frame)
    chart_frame = frame if source_rows <= 4_000 else frame.sample(4_000, random_state=17)
    base = alt.Chart(chart_frame).properties(height=380)
    sample_note = f" · deterministic sample of 4,000 from {source_rows:,} rows" if source_rows > 4_000 else ""
    if chart_type == "Histogram":
        chart = base.mark_bar(color="#34b39a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(field=x_column, type="quantitative", bin=alt.Bin(maxbins=28), title=x_column),
            y=alt.Y(aggregate="count", type="quantitative", title="Number of rows"),
            tooltip=[alt.Tooltip("count():Q", title="Rows")],
        )
        return chart, f"Distribution of {x_column}{sample_note}"
    if chart_type == "Count":
        if not category_column:
            raise ValueError("Choose a column for the count chart.")
        chart = base.mark_bar(color="#34b39a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(field=category_column, type="nominal", sort="-y", title=category_column),
            y=alt.Y(aggregate="count", type="quantitative", title="Number of rows"),
            tooltip=[alt.Tooltip(field=category_column, type="nominal"), alt.Tooltip(aggregate="count", type="quantitative", title="Rows")],
        )
        return chart, f"Row count by {category_column}{sample_note}"
    if chart_type == "Scatter":
        if not y_column:
            raise ValueError("Choose both numeric axes for a scatter chart.")
        correlation = frame[[x_column, y_column]].corr(numeric_only=True).iloc[0, 1]
        association = f"Pearson correlation: {correlation:+.2f}" if pd.notna(correlation) else "Correlation unavailable"
        chart = base.mark_circle(size=78, opacity=.7, color="#3f88c5").encode(
            x=alt.X(field=x_column, type="quantitative", title=x_column), y=alt.Y(field=y_column, type="quantitative", title=y_column),
            tooltip=[x_column, y_column],
        ).interactive()
        return chart, association + " · association does not establish cause" + sample_note
    if chart_type == "Line":
        if not y_column:
            raise ValueError("Choose a numeric measure for a line chart.")
        x_type = "temporal" if pd.api.types.is_datetime64_any_dtype(frame[x_column]) else "quantitative" if pd.api.types.is_numeric_dtype(frame[x_column]) else "nominal"
        chart = base.mark_line(point=True, color="#34b39a").encode(
            x=alt.X(field=x_column, type=x_type, title=x_column),
            y=alt.Y(field=y_column, type="quantitative", aggregate=aggregation.lower(), title=f"{aggregation} {y_column}"),
            tooltip=[alt.Tooltip(field=x_column, type=x_type), alt.Tooltip(field=y_column, type="quantitative", aggregate=aggregation.lower(), format=",.2f")],
        ).interactive()
        return chart, "The chart follows the selected measure over the selected x-axis." + sample_note
    if not y_column or not category_column:
        raise ValueError("Choose a category and a numeric measure for a bar chart.")
    aggregate = {"Mean": "mean", "Sum": "sum", "Median": "median", "Count": "count"}[aggregation]
    y_encoding = alt.Y(field=y_column, type="quantitative", aggregate=aggregate, title=f"{aggregation} {y_column}")
    tooltip = [alt.Tooltip(field=category_column, type="nominal"), alt.Tooltip(field=y_column, type="quantitative", aggregate=aggregate, format=",.2f")]
    chart = base.mark_bar(color="#34b39a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{category_column}:N", sort="-y", title=category_column), y=y_encoding,
        tooltip=tooltip,
    )
    return chart, f"{aggregation} {y_column} by {category_column}{sample_note}"


def build_markdown_report(
    filename: str,
    profile: dict[str, Any],
    clean_profile: dict[str, Any],
    changes: list[dict[str, Any]],
    notes: list[dict[str, str]],
) -> str:
    lines = [
        f"# Analysis report: {filename}",
        "",
        "> Generated from the uploaded data. Recommendations are decision-support prompts, not causal findings.",
        "",
        "## Outcome",
        "",
        f"- Source rows and columns: {profile['rows']:,} × {profile['columns']:,}",
        f"- Cleaned rows and columns: {clean_profile['rows']:,} × {clean_profile['columns']:,}",
        f"- Missing cells: {profile['missing_cells']:,} → {clean_profile['missing_cells']:,}",
        f"- Exact duplicate rows: {profile['duplicate_rows']:,}",
        "",
        "## Cleaning changes",
        "",
    ]
    if changes:
        lines.extend(f"- {item['action']} · {item['column']} · {item['records_changed']:,} records" for item in changes)
    else:
        lines.append("- No changes were needed under the selected cleaning rules.")
    lines.extend(["", "## Suggested next steps", ""])
    lines.extend(f"- **{item['priority']} — {item['title']}:** {item['detail']}" for item in notes)
    lines.extend(["", "## Reproducibility notes", "", "All cleaning choices are configurable in the app. Correlations and outlier flags are exploratory; they do not prove causality. Review the cleaned preview before using an export.", ""])
    return "\n".join(lines)


def default_dashboard(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Choose useful starter tiles based on the uploaded columns."""
    numeric = list(frame.select_dtypes(include="number").columns)
    categories = [column for column in frame.columns if column not in numeric]
    tiles: list[dict[str, Any]] = []
    if numeric and categories:
        tiles.append({"title": f"{numeric[0]} by {categories[0]}", "chart_type": "Bar", "x_column": categories[0], "category_column": categories[0], "y_column": numeric[0], "aggregation": "Sum"})
    elif categories:
        tiles.append({"title": f"Rows by {categories[0]}", "chart_type": "Count", "x_column": categories[0], "category_column": categories[0], "y_column": None, "aggregation": "Count"})
    elif numeric:
        tiles.append({"title": f"{numeric[0]} total", "chart_type": "Metric", "x_column": numeric[0], "category_column": None, "y_column": numeric[0], "aggregation": "Sum"})
    if numeric:
        tiles.append({"title": f"{numeric[0]} distribution", "chart_type": "Histogram", "x_column": numeric[0], "category_column": None, "y_column": None, "aggregation": "Count"})
    if len(numeric) >= 2:
        tiles.append({"title": f"{numeric[0]} vs {numeric[1]}", "chart_type": "Scatter", "x_column": numeric[0], "category_column": None, "y_column": numeric[1], "aggregation": "Mean"})
    while len(tiles) < 3 and (numeric or categories):
        if numeric and categories:
            tiles.append({"title": f"Average {numeric[0]} by {categories[min(len(tiles), len(categories)-1)]}", "chart_type": "Bar", "x_column": categories[min(len(tiles), len(categories)-1)], "category_column": categories[min(len(tiles), len(categories)-1)], "y_column": numeric[0], "aggregation": "Mean"})
        elif categories:
            tiles.append({"title": f"Rows by {categories[min(len(tiles), len(categories)-1)]}", "chart_type": "Count", "x_column": categories[min(len(tiles), len(categories)-1)], "category_column": categories[min(len(tiles), len(categories)-1)], "y_column": None, "aggregation": "Count"})
        else:
            tiles.append({"title": f"Average {numeric[0]}", "chart_type": "Metric", "x_column": numeric[0], "category_column": None, "y_column": numeric[0], "aggregation": "Mean"})
    return tiles[:6]


def normalize_dashboard_tiles(tiles: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate a saved dashboard and fall back safely when columns differ."""
    if not isinstance(tiles, list) or not 1 <= len(tiles) <= 6:
        raise ValueError("A dashboard layout must contain between 1 and 6 charts.")
    defaults = default_dashboard(frame)
    numeric = list(frame.select_dtypes(include="number").columns)
    categories = [column for column in frame.columns if column not in numeric]
    options = {"Count"}
    if numeric:
        options.update({"Bar", "Line", "Scatter", "Histogram", "Metric"})
    valid_aggregations = {"Mean", "Sum", "Median", "Count"}
    normalized = []
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError("Each dashboard chart must be a JSON object.")
        fallback = defaults[index % len(defaults)]
        chart_type = tile.get("chart_type") if tile.get("chart_type") in options else fallback["chart_type"]
        x_column = tile.get("x_column") if tile.get("x_column") in frame.columns else fallback["x_column"]
        category_column = tile.get("category_column") if tile.get("category_column") in frame.columns else (categories[0] if categories else x_column)
        y_column = tile.get("y_column") if tile.get("y_column") in numeric else (numeric[0] if numeric else None)
        aggregation = tile.get("aggregation") if tile.get("aggregation") in valid_aggregations else fallback["aggregation"]
        normalized.append({
            "title": str(tile.get("title") or fallback["title"])[:80],
            "chart_type": chart_type,
            "x_column": x_column,
            "category_column": category_column,
            "y_column": y_column,
            "aggregation": aggregation,
        })
    return normalized


def apply_dashboard_config(config: Any, frame: pd.DataFrame) -> None:
    """Apply a downloaded dashboard layout to the current session and dataset."""
    if not isinstance(config, dict):
        raise ValueError("The layout must be a JSON object.")
    tiles = normalize_dashboard_tiles(config.get("charts"), frame)
    filter_columns = ["All rows", *list(frame.columns)]
    filter_column = config.get("filter_column", "All rows")
    if filter_column not in filter_columns:
        filter_column = "All rows"
    st.session_state.dashboard_tiles = tiles
    st.session_state.dashboard_count = len(tiles)
    st.session_state.dashboard_filter_column = filter_column
    st.session_state.dashboard_name = str(config.get("name") or "My dashboard")[:80]
    if filter_column != "All rows":
        available = frame[filter_column].dropna().unique().tolist()
        requested = config.get("filter_values", [])
        if not isinstance(requested, list):
            requested = []
        st.session_state[f"dashboard_filter_values_{filter_column}"] = [value for value in requested if value in available]
    for index, tile in enumerate(tiles):
        st.session_state[f"dashboard_tile_{index}_title"] = tile["title"]
        st.session_state[f"dashboard_tile_{index}_type"] = tile["chart_type"]
        chart_type = tile["chart_type"].lower()
        if chart_type == "metric":
            st.session_state[f"dashboard_tile_{index}_metric"] = tile["y_column"]
            st.session_state[f"dashboard_tile_{index}_metric_agg"] = tile["aggregation"]
        elif chart_type in {"histogram", "scatter"}:
            st.session_state[f"dashboard_tile_{index}_{chart_type}_x"] = tile["x_column"]
            if chart_type == "scatter":
                st.session_state[f"dashboard_tile_{index}_scatter_y"] = tile["y_column"]
        elif chart_type == "line":
            st.session_state[f"dashboard_tile_{index}_line_x"] = tile["x_column"]
            st.session_state[f"dashboard_tile_{index}_line_y"] = tile["y_column"]
            st.session_state[f"dashboard_tile_{index}_line_agg"] = tile["aggregation"]
        elif chart_type == "bar":
            st.session_state[f"dashboard_tile_{index}_bar_category"] = tile["category_column"]
            st.session_state[f"dashboard_tile_{index}_bar_measure"] = tile["y_column"]
            st.session_state[f"dashboard_tile_{index}_bar_agg"] = tile["aggregation"]
        else:
            st.session_state[f"dashboard_tile_{index}_count_category"] = tile["category_column"]


st.title("📊 AI Data Analyst Bot")
st.caption("From messy file to a clear, downloadable analysis — with every cleaning choice visible.")

with st.sidebar:
    st.header("Your dataset")
    source = st.radio("Start with", ["Try the demo", "Upload my file"], horizontal=False)
    uploaded = None
    sheet_name = ""
    if source == "Upload my file":
        supported_types = ["csv", "tsv", "txt", "json", "jsonl", "parquet", "pdf", "xlsx", "xlsm", "xls", "xlsb", "ods", "xltx", "xltm"]
        uploaded = st.file_uploader("Choose a PDF, spreadsheet, or data file", type=supported_types, help="Supports CSV/TSV/text, JSON/JSONL, Parquet, PDF, and Excel or OpenDocument spreadsheets. Files are processed in this session and are not saved to this repository.")
        filename_lower = uploaded.name.lower() if uploaded is not None else ""
        if uploaded is not None and filename_lower.endswith((".xlsx", ".xlsm", ".xls", ".xlsb", ".ods", ".xltx", ".xltm")):
            try:
                workbook = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
                sheet_name = st.selectbox("Workbook sheet", workbook.sheet_names)
            except Exception as exc:
                st.error(f"Could not read workbook sheets: {exc}")
    st.divider()
    st.subheader("Cleaning choices")
    numeric_strategy = st.selectbox("Fill missing numbers with", ["Median", "Mean", "Zero", "Drop rows"])
    text_strategy = st.selectbox("Fill missing text with", ["Not provided", "Most common", "Drop rows"])
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    trim_text = st.checkbox("Trim spaces around text", value=True)
    st.divider()
    with st.expander("Optional cloud AI chat", expanded=False):
        cloud_ai = st.checkbox("Enable OpenAI answers", value=False, help="Off by default. Local analysis and rule-guided answers do not send data anywhere.")
        cloud_consent = False
        if cloud_ai:
            st.warning("When you ask a question, the app will send it, column names, and up to 8 sample rows to the OpenAI API configured for this app. Do not enable this for sensitive data unless you are authorized to share it.")
            cloud_consent = st.checkbox("I approve sending this limited sample and my questions to the configured API.", value=False)
            st.caption("Configure OPENAI_API_KEY as an environment variable or Streamlit secret. The API is optional; offline analysis still works.")

try:
    if source == "Try the demo":
        raw_df = pd.read_csv(DEMO_FILE)
        filename = DEMO_FILE.name
    elif uploaded is None:
        st.info("Upload a CSV/Excel file, or switch to **Try the demo** to explore a complete sample report immediately.")
        st.stop()
    else:
        content = uploaded.getvalue()
        filename = uploaded.name
        filename_lower = filename.lower()
        if filename_lower.endswith((".xlsx", ".xlsm", ".xls", ".xlsb", ".ods", ".xltx", ".xltm")):
            raw_df = read_excel_bytes(content, sheet_name)
        elif filename_lower.endswith((".json", ".jsonl")):
            raw_df = read_json_bytes(content)
        elif filename_lower.endswith((".parquet", ".pq")):
            raw_df = read_parquet_bytes(content)
        elif filename_lower.endswith(".pdf"):
            raw_df = read_pdf_bytes(content)
        else:
            raw_df = read_csv_bytes(content)
    if raw_df.empty or len(raw_df.columns) == 0:
        st.error("This file has no data rows or columns. Choose another sheet or file.")
        st.stop()
except Exception as exc:
    st.error(f"I couldn't read this file. Check its format and try again. Details: {exc}")
    st.stop()

if filename.lower().endswith(".pdf") and "page_text" in raw_df.columns:
    st.info("No PDF tables were detected. Selectable or OCR-recognized text is organized one row per page; OCR uses English text recognition.")

file_signature = hashlib.sha256((filename + str(raw_df.shape)).encode() + pd.util.hash_pandas_object(raw_df, index=True).values.tobytes()).hexdigest()
if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.messages = []
    saved_dashboards = st.session_state.get("dashboard_saved", {})
    for key in list(st.session_state.keys()):
        if key.startswith("dashboard_") and key != "dashboard_saved":
            del st.session_state[key]
    st.session_state.dashboard_saved = saved_dashboards
    st.session_state.dashboard_tiles = default_dashboard(raw_df)
    st.session_state.dashboard_count = len(st.session_state.dashboard_tiles)
    st.session_state.dashboard_filter_column = "All rows"
    st.session_state.dashboard_filter_values = []

clean_df, cleaning_log = clean_data(raw_df, numeric_strategy=numeric_strategy, text_strategy=text_strategy, remove_duplicates=remove_duplicates, trim_text=trim_text)
source_profile = profile_data(raw_df)
clean_profile = profile_data(clean_df)
actions = recommendations(raw_df)
report_markdown = build_markdown_report(filename, source_profile, clean_profile, cleaning_log, actions)

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{source_profile['rows']:,}", delta=f"{clean_profile['rows'] - source_profile['rows']:+,} after cleaning")
metric_cols[1].metric("Columns", f"{source_profile['columns']:,}")
metric_cols[2].metric("Missing cells", f"{source_profile['missing_cells']:,}", delta=f"{clean_profile['missing_cells'] - source_profile['missing_cells']:+,} after cleaning", delta_color="inverse")
metric_cols[3].metric("Exact duplicates", f"{source_profile['duplicate_rows']:,}")

overview_tab, cleaning_tab, charts_tab, insights_tab, chat_tab = st.tabs(["📋 Dataset", "🧹 Clean", "📊 My Dashboard", "💡 Insights & exports", "💬 Ask the data"])

with overview_tab:
    st.subheader("Your dataset at a glance")
    st.caption(f"Source: `{filename}` · Previewing the uploaded file before any changes.")
    st.dataframe(raw_df.head(100), width="stretch", hide_index=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Column profile")
        profile_rows = []
        for column in raw_df.columns:
            values = raw_df[column]
            profile_rows.append({"Column": str(column), "Type": str(values.dtype), "Missing": int(values.isna().sum()), "Distinct": int(values.nunique(dropna=True))})
        st.dataframe(pd.DataFrame(profile_rows), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Data quality flags")
        if source_profile["missing_by_column"]:
            st.warning("Missing values: " + ", ".join(f"{name} ({count})" for name, count in source_profile["missing_by_column"].items()))
        else:
            st.success("No missing cells found.")
        if source_profile["duplicate_rows"]:
            st.warning(f"{source_profile['duplicate_rows']:,} exact duplicate rows found.")
        else:
            st.success("No exact duplicate rows found.")
        st.caption("A quick profile identifies where to look. It does not validate business meaning or source accuracy.")

with cleaning_tab:
    st.subheader("Review every change before you export")
    before, after = st.columns(2)
    with before:
        st.markdown("#### Before")
        st.caption(f"{len(raw_df):,} rows · {int(raw_df.isna().sum().sum()):,} missing cells")
        st.dataframe(raw_df.head(60), width="stretch", hide_index=True)
    with after:
        st.markdown("#### After")
        st.caption(f"{len(clean_df):,} rows · {int(clean_df.isna().sum().sum()):,} missing cells")
        st.dataframe(clean_df.head(60), width="stretch", hide_index=True)
    st.markdown("#### Change log")
    if cleaning_log:
        st.dataframe(pd.DataFrame(cleaning_log), width="stretch", hide_index=True)
    else:
        st.success("No rows or values changed with the selected options.")
    st.caption("The original uploaded file remains unchanged. Change the cleaning choices in the sidebar to compare outcomes.")
    export_df = spreadsheet_safe_copy(clean_df)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download cleaned CSV", csv_bytes, file_name=f"{Path(filename).stem}_cleaned.csv", mime="text/csv", key="download_clean_csv")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="Cleaned data", index=False)
        pd.DataFrame(cleaning_log or [{"action": "No changes", "column": "—", "records_changed": 0}]).to_excel(writer, sheet_name="Change log", index=False)
    st.download_button("Download cleaned Excel workbook", excel_buffer.getvalue(), file_name=f"{Path(filename).stem}_cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_clean_xlsx")
    st.caption("Spreadsheet formula-like text is prefixed in downloads so spreadsheet applications treat it as text.")

with charts_tab:
    st.subheader("Build your dashboard")
    st.caption("Choose multiple charts and a shared filter for any uploaded dataset. Save the layout as JSON to reuse it later; layout files contain settings, not your data.")

    with st.expander("Load a saved dashboard layout"):
        layout_file = st.file_uploader("Choose a dashboard JSON file", type=["json"], key="dashboard_layout_upload")
        if st.button("Load dashboard layout", disabled=layout_file is None, key="dashboard_load_file"):
            try:
                apply_dashboard_config(json.loads(layout_file.getvalue().decode("utf-8")), clean_df)
                st.rerun()
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                st.error(f"That dashboard layout could not be loaded: {exc}")

    numeric_columns = list(clean_df.select_dtypes(include="number").columns)
    chart_types = ["Count"] + (["Bar", "Line", "Scatter", "Histogram", "Metric"] if numeric_columns else [])
    defaults = default_dashboard(clean_df)
    tiles = list(st.session_state.get("dashboard_tiles", defaults)) or defaults
    chart_count = st.slider("Number of charts", 1, 6, min(6, max(1, len(tiles))), key="dashboard_count")
    while len(tiles) < chart_count:
        tiles.append(defaults[len(tiles) % len(defaults)])
    tiles = tiles[:chart_count]

    filter_columns = ["All rows", *list(clean_df.columns)]
    current_filter = st.session_state.get("dashboard_filter_column", "All rows")
    if current_filter not in filter_columns:
        current_filter = "All rows"
    filter_column = st.selectbox("Filter the whole dashboard by", filter_columns, index=filter_columns.index(current_filter), key="dashboard_filter_column")
    filter_values = []
    filtered_df = clean_df
    if filter_column != "All rows":
        options = sorted(clean_df[filter_column].dropna().unique().tolist(), key=lambda value: str(value))
        has_many_values = len(options) > 150
        if has_many_values:
            options = options[:150]
            st.caption("This field has many unique values; the filter shows the first 150 in sorted order. Leave the selection empty to include all values.")
        filter_key = f"dashboard_filter_values_{filter_column}"
        selected = st.multiselect("Keep values (leave empty for all)", options=options, default=[value for value in st.session_state.get(filter_key, []) if value in options], key=filter_key)
        filter_values = selected
        if selected:
            filtered_df = clean_df[clean_df[filter_column].isin(selected)]
    st.caption(f"Showing {len(filtered_df):,} of {len(clean_df):,} cleaned rows")

    draft_tiles = []
    for index, tile in enumerate(tiles):
        safe_type = tile.get("chart_type") if tile.get("chart_type") in chart_types else chart_types[0]
        title_key = f"dashboard_tile_{index}_title"
        type_key = f"dashboard_tile_{index}_type"
        with st.container(border=True):
            title_cols = st.columns([2, 1])
            title = title_cols[0].text_input("Chart title", value=st.session_state.get(title_key, tile.get("title", f"Chart {index + 1}")), max_chars=80, key=title_key)
            chart_type = title_cols[1].selectbox("Chart type", chart_types, index=chart_types.index(st.session_state.get(type_key, safe_type)) if st.session_state.get(type_key, safe_type) in chart_types else 0, key=type_key)
            x_column = clean_df.columns[0]
            category_column = None
            y_column = None
            aggregation = "Count"
            if chart_type == "Metric":
                cols = st.columns(2)
                prior_y = tile.get("y_column") if tile.get("y_column") in numeric_columns else numeric_columns[0]
                y_column = cols[0].selectbox("Measure", numeric_columns, index=numeric_columns.index(prior_y), key=f"dashboard_tile_{index}_metric")
                aggregations = ["Sum", "Mean", "Median", "Count"]
                prior_agg = tile.get("aggregation") if tile.get("aggregation") in aggregations else "Sum"
                aggregation = cols[1].selectbox("Calculation", aggregations, index=aggregations.index(prior_agg), key=f"dashboard_tile_{index}_metric_agg")
            elif chart_type == "Histogram":
                prior_x = tile.get("x_column") if tile.get("x_column") in numeric_columns else numeric_columns[0]
                x_column = st.selectbox("Measure distribution", numeric_columns, index=numeric_columns.index(prior_x), key=f"dashboard_tile_{index}_histogram_x")
            elif chart_type == "Scatter":
                cols = st.columns(2)
                prior_x = tile.get("x_column") if tile.get("x_column") in numeric_columns else numeric_columns[0]
                prior_y = tile.get("y_column") if tile.get("y_column") in numeric_columns else numeric_columns[min(1, len(numeric_columns)-1)]
                x_column = cols[0].selectbox("Horizontal measure", numeric_columns, index=numeric_columns.index(prior_x), key=f"dashboard_tile_{index}_scatter_x")
                y_column = cols[1].selectbox("Vertical measure", numeric_columns, index=numeric_columns.index(prior_y), key=f"dashboard_tile_{index}_scatter_y")
            elif chart_type == "Line":
                cols = st.columns(3)
                prior_x = tile.get("x_column") if tile.get("x_column") in clean_df.columns else clean_df.columns[0]
                prior_y = tile.get("y_column") if tile.get("y_column") in numeric_columns else numeric_columns[0]
                x_column = cols[0].selectbox("X axis", list(clean_df.columns), index=list(clean_df.columns).index(prior_x), key=f"dashboard_tile_{index}_line_x")
                y_column = cols[1].selectbox("Measure", numeric_columns, index=numeric_columns.index(prior_y), key=f"dashboard_tile_{index}_line_y")
                aggregation = cols[2].selectbox("Calculation", ["Mean", "Sum", "Count"], key=f"dashboard_tile_{index}_line_agg")
            elif chart_type == "Bar":
                cols = st.columns(3)
                category_options = [column for column in clean_df.columns if column not in numeric_columns] or list(clean_df.columns)
                prior_category = tile.get("category_column") if tile.get("category_column") in category_options else category_options[0]
                category_column = cols[0].selectbox("Group by", category_options, index=category_options.index(prior_category), key=f"dashboard_tile_{index}_bar_category")
                y_column = cols[1].selectbox("Measure", numeric_columns, key=f"dashboard_tile_{index}_bar_measure")
                aggregation = cols[2].selectbox("Calculation", ["Mean", "Sum", "Count"], key=f"dashboard_tile_{index}_bar_agg")
            else:
                category_column = st.selectbox("Count rows by", list(clean_df.columns), index=list(clean_df.columns).index(tile.get("category_column")) if tile.get("category_column") in clean_df.columns else 0, key=f"dashboard_tile_{index}_count_category")
            draft_tiles.append({"title": title.strip() or f"Chart {index + 1}", "chart_type": chart_type, "x_column": x_column, "category_column": category_column, "y_column": y_column, "aggregation": aggregation})

    st.session_state.dashboard_tiles = draft_tiles
    st.markdown("#### Dashboard preview")
    if filtered_df.empty:
        st.info("No rows match the selected filter. Choose more values to see the dashboard.")
    else:
        chart_columns = st.columns(2)
        for index, tile in enumerate(draft_tiles):
            with chart_columns[index % 2]:
                st.markdown(f"**{tile['title']}**")
                try:
                    if tile["chart_type"] == "Metric":
                        series = filtered_df[tile["y_column"]]
                        value = {"Sum": series.sum, "Mean": series.mean, "Median": series.median, "Count": series.count}[tile["aggregation"]]()
                        st.metric(label=f"{tile['aggregation']} · {tile['y_column']}", value=f"{value:,.2f}" if pd.notna(value) else "—")
                    else:
                        chart, summary = make_chart(filtered_df, tile["chart_type"], tile["x_column"], tile["y_column"], tile["category_column"], tile["aggregation"])
                        st.altair_chart(chart, width="stretch")
                        st.caption(summary)
                except (ValueError, TypeError, KeyError) as exc:
                    st.warning(f"Adjust this chart's settings: {exc}")

    dashboard_name = st.text_input("Dashboard name", value=st.session_state.get("dashboard_name", "My dashboard"), max_chars=80, key="dashboard_name")
    layout = {"version": 1, "name": dashboard_name.strip() or "My dashboard", "charts": draft_tiles, "filter_column": filter_column, "filter_values": filter_values}
    controls = st.columns(2)
    controls[0].download_button("Save dashboard layout (JSON)", json.dumps(layout, indent=2, default=str), file_name=f"{Path(dashboard_name.strip() or 'dashboard').stem}_dashboard.json", mime="application/json", key="dashboard_download_layout")
    st.caption("The uploaded dataset stays in this app session. Save your layout JSON separately; it contains chart and filter settings, never the uploaded data.")

with insights_tab:
    st.subheader("Decision-ready summary")
    st.markdown(report_markdown)
    st.download_button("Download analysis report (Markdown)", report_markdown, file_name=f"{Path(filename).stem}_analysis.md", mime="text/markdown", key="download_report")
    json_report = {
        "source_file": filename,
        "source_profile": source_profile,
        "cleaned_profile": clean_profile,
        "cleaning_log": cleaning_log,
        "recommendations": actions,
    }
    st.download_button("Download machine-readable report (JSON)", json.dumps(json_report, indent=2, default=str), file_name=f"{Path(filename).stem}_analysis.json", mime="application/json", key="download_report_json")
    st.markdown("#### Suggested next steps")
    for item in actions:
        with st.container(border=True):
            st.caption(f"{item['priority']} PRIORITY")
            st.markdown(f"**{item['title']}**")
            st.write(item["detail"])
    st.markdown("#### SQL starter queries (SQLite)")
    sql_text = sql_starter(clean_df)
    st.code(sql_text, language="sql")
    st.download_button("Download SQL starter", sql_text, file_name=f"{Path(filename).stem}_starter.sql", mime="text/sql", key="download_sql")
    st.markdown("#### Reproducible Python starter")
    python_text = python_starter(filename, numeric_strategy, text_strategy)
    st.code(python_text, language="python")
    st.download_button("Download Python starter", python_text, file_name=f"{Path(filename).stem}_analysis.py", mime="text/x-python", key="download_python")

with chat_tab:
    st.subheader("Ask a question about this dataset")
    st.caption("Local mode is private to this app session and always available. It answers counts, missingness, duplicates, summaries, and correlations.")
    if cloud_ai and cloud_consent and not openai_api_key():
        st.warning("Cloud answers are enabled, but OPENAI_API_KEY is not configured. Local answers are still available.")
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Try: Which columns have missing data?", max_chars=500)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer = None
            if cloud_ai and cloud_consent and openai_api_key():
                try:
                    from openai import OpenAI

                    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
                    compact_data = json.loads(clean_df.head(8).to_json(orient="records", date_format="iso"))
                    context = json.dumps({"columns": list(map(str, clean_df.columns)), "shape": clean_df.shape, "sample_rows": compact_data}, default=str)
                    with st.spinner("Asking the configured OpenAI model about the sample you approved…"):
                        response = OpenAI(api_key=openai_api_key()).responses.create(
                            model=model,
                            store=False,
                            instructions="Answer questions about the supplied dataset sample and schema. State when the sample is insufficient. Never infer causation from correlation. Do not claim to have analyzed rows outside the provided sample unless the provided context explicitly contains aggregate results.",
                            input=f"Dataset context (first 8 cleaned rows only): {context}\n\nQuestion: {prompt}",
                        )
                    answer = response.output_text
                except Exception as exc:
                    st.warning(f"Cloud answer unavailable ({type(exc).__name__}); using the local analyst instead.")
            if answer is None:
                answer = answer_question(prompt, clean_df)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("The original upload is never overwritten. Review cleaning logs and validate domain-specific recommendations before acting.")

