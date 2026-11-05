"""Deterministic, inspectable analysis helpers for AI Data Analyst Bot."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd


def profile_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact quality and shape metrics without mutating the input."""
    rows, columns = frame.shape
    missing = frame.isna().sum()
    numeric = frame.select_dtypes(include="number")
    correlation_pairs: list[dict[str, Any]] = []
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        for left_index, left in enumerate(corr.columns):
            for right in corr.columns[left_index + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    correlation_pairs.append({"left": str(left), "right": str(right), "correlation": float(value)})
    correlation_pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_cells": int(missing.sum()),
        "missing_percent": float(missing.sum() / max(rows * columns, 1) * 100),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_columns": [str(column) for column in numeric.columns],
        "categorical_columns": [str(column) for column in frame.select_dtypes(exclude="number").columns],
        "missing_by_column": {str(key): int(value) for key, value in missing.items() if value > 0},
        "correlations": correlation_pairs,
    }


def clean_data(
    frame: pd.DataFrame,
    *,
    numeric_strategy: str = "Median",
    text_strategy: str = "Not provided",
    remove_duplicates: bool = True,
    trim_text: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply explicit cleanup choices to a copy and return a human-readable change log."""
    cleaned = frame.copy(deep=True)
    actions: list[dict[str, Any]] = []

    if trim_text:
        text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            before = cleaned[column]
            trimmed = before.map(lambda value: value.strip() if isinstance(value, str) else value)
            trimmed = trimmed.map(lambda value: pd.NA if isinstance(value, str) and not value else value)
            changed = sum(isinstance(old, str) and (not isinstance(new, str) or old != new) for old, new in zip(before.tolist(), trimmed.tolist()))
            if changed:
                cleaned[column] = trimmed
                actions.append({"action": "Trim surrounding spaces", "column": str(column), "records_changed": changed})

    if remove_duplicates:
        duplicate_count = int(cleaned.duplicated().sum())
        if duplicate_count:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            actions.append({"action": "Remove exact duplicate rows", "column": "All columns", "records_changed": duplicate_count})

    if numeric_strategy == "Drop rows" or text_strategy == "Drop rows":
        before_rows = len(cleaned)
        cleaned = cleaned.dropna().reset_index(drop=True)
        removed = before_rows - len(cleaned)
        if removed:
            actions.append({"action": "Drop rows with missing values", "column": "Any column", "records_changed": removed})
        return cleaned, actions

    for column in cleaned.select_dtypes(include="number").columns:
        missing_count = int(cleaned[column].isna().sum())
        if not missing_count:
            continue
        if numeric_strategy == "Mean" and cleaned[column].notna().any():
            fill_value = cleaned[column].mean()
        elif numeric_strategy == "Median" and cleaned[column].notna().any():
            fill_value = cleaned[column].median()
        else:
            fill_value = 0
        cleaned[column] = cleaned[column].fillna(fill_value)
        actions.append({"action": f"Fill with {numeric_strategy.lower()}", "column": str(column), "records_changed": missing_count})

    for column in cleaned.select_dtypes(include=["object", "string", "category"]).columns:
        missing_count = int(cleaned[column].isna().sum())
        if not missing_count:
            continue
        if text_strategy == "Most common" and cleaned[column].notna().any():
            fill_value = cleaned[column].mode(dropna=True).iloc[0]
        else:
            fill_value = "Not provided"
        cleaned[column] = cleaned[column].astype("object").fillna(fill_value)
        actions.append({"action": "Fill with most common value" if text_strategy == "Most common" else "Fill with Not provided", "column": str(column), "records_changed": missing_count})

    return cleaned.reset_index(drop=True), actions


def recommendations(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Translate measurable data signals into cautious, actionable next steps."""
    profile = profile_data(frame)
    items: list[dict[str, str]] = []
    if profile["missing_cells"]:
        worst = max(profile["missing_by_column"].items(), key=lambda item: item[1])
        percent = worst[1] / max(profile["rows"], 1) * 100
        items.append({"priority": "High" if percent >= 20 else "Medium", "title": "Investigate incomplete fields", "detail": f"'{worst[0]}' is missing in {worst[1]:,} rows ({percent:.1f}%). Check whether blank values have a shared cause before choosing an imputation rule."})
    if profile["duplicate_rows"]:
        items.append({"priority": "High" if profile["duplicate_rows"] / max(profile["rows"], 1) > .02 else "Medium", "title": "Review repeated records", "detail": f"{profile['duplicate_rows']:,} exact duplicate rows were found. Confirm the correct business key before removing records; repeated transactions can sometimes be valid."})
    if profile["rows"] < 30:
        items.append({"priority": "Medium", "title": "Treat small-sample patterns carefully", "detail": f"There are only {profile['rows']:,} rows. Use these summaries as leads, and validate them with more observations before making a high-impact decision."})
    if profile["correlations"]:
        strongest = profile["correlations"][0]
        items.append({"priority": "Explore", "title": "Investigate a relationship, not a cause", "detail": f"The strongest pair is '{strongest['left']}' and '{strongest['right']}' (correlation {strongest['correlation']:+.2f}). Check for outliers, time effects, and confounding factors; correlation alone does not establish causation."})
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if len(values) >= 5 and values.nunique() > 1:
            q1, q3 = values.quantile([.25, .75])
            iqr = q3 - q1
            count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr else 0
            if count:
                items.append({"priority": "Explore", "title": f"Check unusual values in {column}", "detail": f"The interquartile-range screen flags {count:,} values. Verify them against the source; a statistical outlier can be a real and important event."})
                break
    if not items:
        items.append({"priority": "Good start", "title": "No immediate quality flags", "detail": "This quick scan found no missing cells or exact duplicate rows. Confirm definitions, units, and data provenance before using the numbers in a decision."})
    return items


def answer_question(question: str, frame: pd.DataFrame) -> str:
    """Answer common dataset questions locally; never sends data over a network."""
    query = question.casefold().strip()
    profile = profile_data(frame)
    if re.search(r"\b(rows?|records?|observations?)\b", query):
        return f"This dataset has {profile['rows']:,} rows and {profile['columns']:,} columns."
    if re.search(r"\b(columns?|fields?|variables?)\b", query):
        return "The columns are: " + ", ".join(map(str, frame.columns)) + "."
    if re.search(r"\b(duplicate|repeated)\b", query):
        return f"I found {profile['duplicate_rows']:,} exact duplicate rows out of {profile['rows']:,}. Similar-looking values are not counted unless the whole row matches."
    if re.search(r"\b(missing|null|blank|incomplete)\b", query):
        if not profile["missing_by_column"]:
            return "There are no missing cells in the current dataset."
        detail = "; ".join(f"{name}: {count:,}" for name, count in profile["missing_by_column"].items())
        return f"There are {profile['missing_cells']:,} missing cells ({profile['missing_percent']:.1f}% of all cells). By column: {detail}."
    if re.search(r"\b(correlat|relationship|related)\b", query):
        if not profile["correlations"]:
            return "There are not enough numeric columns with variation to calculate a correlation."
        mentioned = [column for column in profile["numeric_columns"] if str(column).casefold() in query]
        if len(mentioned) >= 2:
            value = frame[mentioned[0]].corr(frame[mentioned[1]])
            if pd.notna(value):
                return f"The Pearson correlation between {mentioned[0]} and {mentioned[1]} is {value:+.2f}. This describes a linear association in the current cleaned data; it does not establish cause."
        pairs = profile["correlations"][:3]
        detail = "; ".join(f"{item['left']} and {item['right']}: {item['correlation']:+.2f}" for item in pairs)
        return f"The strongest numeric associations are {detail}. Correlation describes a linear relationship, not a cause."
    for column in frame.columns:
        if str(column).casefold() in query:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                return f"{column}: mean {series.mean():,.2f}, median {series.median():,.2f}, minimum {series.min():,.2f}, maximum {series.max():,.2f}. These summarize the current cleaned data."
            common = series.mode(dropna=True)
            top = common.iloc[0] if not common.empty else "no non-empty values"
            return f"{column} contains {series.nunique(dropna=True):,} distinct non-empty values. The most common value is {top!r}."
    if re.search(r"\b(summary|describe|overview)\b", query):
        return f"The current dataset has {profile['rows']:,} rows, {profile['columns']:,} columns, {profile['missing_cells']:,} missing cells, and {profile['duplicate_rows']:,} exact duplicate rows. Numeric fields: {', '.join(profile['numeric_columns']) or 'none detected'}."
    return "I can answer questions about row counts, columns, missing values, duplicates, numeric summaries, and correlations. Try: 'How many rows are there?' or 'What is the correlation between revenue and quantity?'"


def sql_starter(frame: pd.DataFrame, table_name: str = "uploaded_data") -> str:
    """Build a portable SQLite starter script with safely quoted identifiers."""
    def quote(identifier: str) -> str:
        return '"' + str(identifier).replace('"', '""') + '"'

    definitions = []
    for column, dtype in frame.dtypes.items():
        sql_type = "REAL" if pd.api.types.is_numeric_dtype(dtype) else "TEXT"
        definitions.append(f"    {quote(column)} {sql_type}")
    columns = ", ".join(quote(column) for column in frame.columns)
    return (
        "-- SQLite starter queries for the cleaned dataset.\n"
        f"CREATE TABLE {quote(table_name)} (\n" + ",\n".join(definitions) + "\n);\n\n"
        f"SELECT COUNT(*) AS row_count FROM {quote(table_name)};\n"
        f"SELECT {columns} FROM {quote(table_name)} LIMIT 20;\n"
    )


def spreadsheet_safe_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix formula-like text cells so spreadsheet apps treat them as text."""
    safe = frame.copy(deep=True)
    formula_prefix = re.compile(r"^\s*[=+@\-\t\r]")
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and formula_prefix.match(value) else value
        )
    return safe


def python_starter(filename: str, numeric_strategy: str, text_strategy: str) -> str:
    """Return a compact, reproducible pandas cleaning and analysis script."""
    safe_filename = repr(filename)
    reader = "pd.read_excel" if filename.lower().endswith((".xlsx", ".xls")) else "pd.read_csv"
    lines = [
        "import pandas as pd", "", f"df = {reader}({safe_filename})", "df = df.drop_duplicates()",
        "for column in df.select_dtypes(include=['object', 'string']).columns:",
        "    df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)",
        "for column in df.select_dtypes(include='number').columns:",
    ]
    if numeric_strategy == "Zero":
        lines.append("    df[column] = df[column].fillna(0)")
    else:
        method = "mean" if numeric_strategy == "Mean" else "median"
        lines.extend([f"    fill_value = df[column].{method}() if df[column].notna().any() else 0", "    df[column] = df[column].fillna(fill_value)"])
    lines.append("for column in df.select_dtypes(exclude='number').columns:")
    if text_strategy == "Most common":
        lines.extend(["    modes = df[column].mode(dropna=True)", "    fill_value = modes.iloc[0] if not modes.empty else 'Not provided'", "    df[column] = df[column].fillna(fill_value)"])
    else:
        lines.append("    df[column] = df[column].fillna('Not provided')")
    lines.extend(["", "print(df.shape)", "print(df.describe(include='all').transpose())", "print(df.select_dtypes(include='number').corr())", "df.to_csv('cleaned_output.csv', index=False)", ""])
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    """Convert common numpy/pandas scalar values for JSON report downloads."""
    return json.loads(json.dumps(value, default=lambda item: item.item() if hasattr(item, "item") else str(item)))
