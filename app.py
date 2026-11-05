"""AI Data Analyst Bot: turn a CSV or workbook into a reviewable analysis."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from analyst import answer_question, clean_data, profile_data, python_starter, recommendations, spreadsheet_safe_copy, sql_starter


APP_DIR = Path(__file__).parent
DEMO_FILE = APP_DIR / "sample_data" / "sales_messy.csv"

st.set_page_config(page_title="AI Data Analyst Bot", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


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
    base = alt.Chart(frame).properties(height=380)
    if chart_type == "Histogram":
        chart = base.mark_bar(color="#34b39a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(field=x_column, type="quantitative", bin=alt.Bin(maxbins=28), title=x_column),
            y=alt.Y(aggregate="count", type="quantitative", title="Number of rows"),
            tooltip=[alt.Tooltip("count():Q", title="Rows")],
        )
        return chart, f"Distribution of {x_column}"
    if chart_type == "Scatter":
        if not y_column:
            raise ValueError("Choose both numeric axes for a scatter chart.")
        correlation = frame[[x_column, y_column]].corr(numeric_only=True).iloc[0, 1]
        association = f"Pearson correlation: {correlation:+.2f}" if pd.notna(correlation) else "Correlation unavailable"
        chart = base.mark_circle(size=78, opacity=.7, color="#3f88c5").encode(
            x=alt.X(field=x_column, type="quantitative", title=x_column), y=alt.Y(field=y_column, type="quantitative", title=y_column),
            tooltip=[x_column, y_column],
        ).interactive()
        return chart, association + " · association does not establish cause"
    if chart_type == "Line":
        if not y_column:
            raise ValueError("Choose a numeric measure for a line chart.")
        x_type = "temporal" if pd.api.types.is_datetime64_any_dtype(frame[x_column]) else "quantitative" if pd.api.types.is_numeric_dtype(frame[x_column]) else "nominal"
        chart = base.mark_line(point=True, color="#34b39a").encode(
            x=alt.X(field=x_column, type=x_type, title=x_column),
            y=alt.Y(field=y_column, type="quantitative", aggregate=aggregation.lower(), title=f"{aggregation} {y_column}"),
            tooltip=[alt.Tooltip(field=x_column, type=x_type), alt.Tooltip(field=y_column, type="quantitative", aggregate=aggregation.lower(), format=",.2f")],
        ).interactive()
        return chart, "The chart follows the selected measure over the selected x-axis."
    if not y_column or not category_column:
        raise ValueError("Choose a category and a numeric measure for a bar chart.")
    aggregate = {"Mean": "mean", "Sum": "sum", "Count": "count"}[aggregation]
    y_encoding = alt.Y(field=y_column, type="quantitative", aggregate=aggregate, title=f"{aggregation} {y_column}")
    tooltip = [alt.Tooltip(field=category_column, type="nominal"), alt.Tooltip(field=y_column, type="quantitative", aggregate=aggregate, format=",.2f")]
    chart = base.mark_bar(color="#34b39a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{category_column}:N", sort="-y", title=category_column), y=y_encoding,
        tooltip=tooltip,
    )
    return chart, f"{aggregation} {y_column} by {category_column}"


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


st.title("📊 AI Data Analyst Bot")
st.caption("From messy file to a clear, downloadable analysis — with every cleaning choice visible.")

with st.sidebar:
    st.header("Your dataset")
    source = st.radio("Start with", ["Try the demo", "Upload my file"], horizontal=False)
    uploaded = None
    sheet_name = ""
    if source == "Upload my file":
        uploaded = st.file_uploader("Choose a CSV or Excel workbook", type=["csv", "xlsx", "xls"], help="Your file is processed in this Streamlit session. It is not saved to this repository.")
        if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")):
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
        raw_df = read_excel_bytes(content, sheet_name) if filename.lower().endswith((".xlsx", ".xls")) else read_csv_bytes(content)
    if raw_df.empty or len(raw_df.columns) == 0:
        st.error("This file has no data rows or columns. Choose another sheet or file.")
        st.stop()
except Exception as exc:
    st.error(f"I couldn't read this file. Check its format and try again. Details: {exc}")
    st.stop()

file_signature = hashlib.sha256((filename + str(raw_df.shape)).encode() + pd.util.hash_pandas_object(raw_df, index=True).values.tobytes()).hexdigest()
if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.messages = []

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

overview_tab, cleaning_tab, charts_tab, insights_tab, chat_tab = st.tabs(["📋 Dataset", "🧹 Clean", "📈 Visualize", "💡 Insights & exports", "💬 Ask the data"])

with overview_tab:
    st.subheader("Your dataset at a glance")
    st.caption(f"Source: `{filename}` · Previewing the uploaded file before any changes.")
    st.dataframe(raw_df.head(100), use_container_width=True, hide_index=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Column profile")
        profile_rows = []
        for column in raw_df.columns:
            values = raw_df[column]
            profile_rows.append({"Column": str(column), "Type": str(values.dtype), "Missing": int(values.isna().sum()), "Distinct": int(values.nunique(dropna=True))})
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)
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
        st.dataframe(raw_df.head(60), use_container_width=True, hide_index=True)
    with after:
        st.markdown("#### After")
        st.caption(f"{len(clean_df):,} rows · {int(clean_df.isna().sum().sum()):,} missing cells")
        st.dataframe(clean_df.head(60), use_container_width=True, hide_index=True)
    st.markdown("#### Change log")
    if cleaning_log:
        st.dataframe(pd.DataFrame(cleaning_log), use_container_width=True, hide_index=True)
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
    st.subheader("Build a quick chart")
    numeric_columns = list(clean_df.select_dtypes(include="number").columns)
    if not clean_df.columns.tolist():
        st.info("No columns to chart.")
    else:
        control_cols = st.columns(4)
        chart_type = control_cols[0].selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram"])
        if not numeric_columns:
            st.info("This chart needs at least one numeric column. Review the column types in the Dataset tab.")
        elif chart_type == "Histogram":
            metric = control_cols[1].selectbox("Measure to inspect", numeric_columns)
            chart, summary = make_chart(clean_df, chart_type, metric, None, None, "Count")
            st.altair_chart(chart, use_container_width=True)
            st.caption(summary)
        elif chart_type == "Scatter":
            axis_cols = st.columns(2)
            scatter_x = axis_cols[0].selectbox("Horizontal measure", numeric_columns, key="scatter_x")
            scatter_y = axis_cols[1].selectbox("Vertical measure", numeric_columns, index=min(1, len(numeric_columns)-1), key="scatter_y")
            chart, summary = make_chart(clean_df, chart_type, scatter_x, scatter_y, None, "Mean")
            st.altair_chart(chart, use_container_width=True)
            st.caption(summary)
        elif chart_type == "Line":
            x_col = control_cols[1].selectbox("X axis", list(clean_df.columns))
            y_col = control_cols[2].selectbox("Measure", numeric_columns)
            aggregation = control_cols[3].selectbox("Aggregation", ["Mean", "Sum", "Count"])
            chart, summary = make_chart(clean_df, chart_type, x_col, y_col, None, aggregation)
            st.altair_chart(chart, use_container_width=True)
            st.caption(summary)
        else:
            categorical_columns = [column for column in clean_df.columns if column not in numeric_columns]
            if not categorical_columns:
                st.info("A bar chart needs a category column. Choose Line, Scatter, or Histogram instead.")
            else:
                category = control_cols[1].selectbox("Group bars by", categorical_columns, key="bar_category")
                y_col = control_cols[2].selectbox("Numeric measure", numeric_columns)
                aggregation = control_cols[3].selectbox("Aggregation", ["Mean", "Sum", "Count"])
                chart, summary = make_chart(clean_df, chart_type, category, y_col, category, aggregation)
                st.altair_chart(chart, use_container_width=True)
                st.caption(summary)
    st.caption("Charts use the cleaned data and selected aggregations. Investigate surprising patterns before drawing conclusions.")

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
