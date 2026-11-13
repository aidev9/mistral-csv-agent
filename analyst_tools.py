"""Fixed analysis tools for the drive-data-analyst agent.

Every tool is a deterministic Python function over an in-memory CSV dataset.
There is no shell, no file system access and no code execution: the model can
only pick one of these tools with fixed arguments.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from mistralai import agents

MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
ALLOWED_FOLDER = "data"
MISSING_VALUES = {"", "na", "n/a", "null", "none", "nan"}
OPERATORS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts_with",
    "ends_with",
    "in",
    "is_null",
    "not_null",
)
AGGREGATIONS = ("count", "sum", "mean", "median", "min", "max", "nunique")

_DRIVE_ID_PATTERN = re.compile(r"/d/([A-Za-z0-9_-]{20,})|[?&]id=([A-Za-z0-9_-]{20,})")
_BARE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{20,}")


@dataclass
class Dataset:
    file_id: str
    file_name: str
    columns: list[str]
    rows: list[dict[str, Any]]


_DATASETS: dict[str, Dataset] = {}


def _dataset_for(session_id: str) -> Dataset:
    dataset = _DATASETS.get(session_id)
    if dataset is None:
        raise agents.ToolError(
            "No CSV is loaded for this session. Call load_csv again with the Google "
            "Drive link or file id of the file discussed earlier in this conversation."
        )
    return dataset


def _dataset_info(dataset: Dataset) -> dict[str, Any]:
    return {
        "file_name": dataset.file_name,
        "file_id": dataset.file_id,
        "row_count": len(dataset.rows),
        "column_count": len(dataset.columns),
        "columns": dataset.columns,
    }


def extract_file_id(value: str) -> str:
    candidate = value.strip()
    match = _DRIVE_ID_PATTERN.search(candidate)
    if match:
        return match.group(1) or match.group(2)
    if _BARE_ID_PATTERN.fullmatch(candidate):
        return candidate
    raise agents.ToolError(
        f"'{value}' does not look like a Google Drive file id or link. Ask the user "
        "for a link of the form https://drive.google.com/file/d/<FILE_ID>/view."
    )


def _parse_cell(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in MISSING_VALUES:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_csv(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not raw_rows:
        raise agents.ToolError("The file contains no data rows.")
    columns: list[str] = []
    for index, header in enumerate(raw_rows[0]):
        name = header.strip() or f"column_{index + 1}"
        base, suffix = name, 2
        while name in columns:
            name = f"{base}_{suffix}"
            suffix += 1
        columns.append(name)
    rows = []
    for raw_row in raw_rows[1:]:
        rows.append(
            {
                column: _parse_cell(raw_row[i]) if i < len(raw_row) else None
                for i, column in enumerate(columns)
            }
        )
    return columns, rows


def _download_csv(file_id: str) -> bytes:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(MAX_DOWNLOAD_BYTES)
    except Exception as exc:
        raise agents.ToolError(
            f"Could not download the file directly from Google Drive ({exc}). Read "
            "the file's content with your Google Drive connector tools, then call "
            "load_csv again passing that content as csv_text."
        ) from exc
    head = data[:512].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise agents.ToolError(
            "Google Drive returned a web page instead of the file, so it is probably "
            "not shared by link. Read the file's content with your Google Drive "
            "connector tools, then call load_csv again passing that content as "
            "csv_text."
        )
    return data


def _column_type(values: list[Any]) -> str:
    if not values:
        return "empty"
    if all(isinstance(value, (int, float)) for value in values):
        return "number"
    return "text"


def _require_column(dataset: Dataset, name: Any) -> str:
    if not isinstance(name, str) or name not in dataset.columns:
        raise agents.ToolError(
            f"Unknown column {name!r}. Available columns: {', '.join(dataset.columns)}."
        )
    return name


def _coerce(value: Any, target: Any) -> Any:
    if isinstance(target, (int, float)) and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _match(row_value: Any, operator: str, value: Any, values: list[Any] | None) -> bool:
    if operator == "is_null":
        return row_value is None
    if operator == "not_null":
        return row_value is not None
    if operator == "in":
        if values is None:
            raise agents.ToolError("The 'in' operator needs a list of values in 'values'.")
        if row_value is None:
            return False
        if isinstance(row_value, str):
            low = row_value.strip().lower()
            return any(
                isinstance(candidate, str) and candidate.strip().lower() == low
                for candidate in values
            )
        return any(_coerce(candidate, row_value) == row_value for candidate in values)
    if operator in ("contains", "starts_with", "ends_with"):
        if row_value is None:
            return False
        haystack = str(row_value).strip().lower()
        needle = str(value).strip().lower()
        if operator == "contains":
            return needle in haystack
        if operator == "starts_with":
            return haystack.startswith(needle)
        return haystack.endswith(needle)
    if value is None:
        if operator == "eq":
            return row_value is None
        if operator == "ne":
            return row_value is not None
        raise agents.ToolError(f"The '{operator}' operator needs a non-null 'value'.")
    if row_value is None:
        return False
    left, right = row_value, _coerce(value, row_value)
    if isinstance(left, str) and isinstance(right, str):
        left_key, right_key = left.strip().lower(), right.strip().lower()
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_key, right_key = left, right
    else:
        raise agents.ToolError(
            f"Cannot compare {row_value!r} with {value!r}: incompatible types. Check "
            "the column's type with describe_columns."
        )
    if operator == "eq":
        return left_key == right_key
    if operator == "ne":
        return left_key != right_key
    if operator == "gt":
        return left_key > right_key
    if operator == "gte":
        return left_key >= right_key
    if operator == "lt":
        return left_key < right_key
    return left_key <= right_key


def _validate_filters(dataset: Dataset, filters: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not filters:
        return []
    validated = []
    for item in filters:
        if not isinstance(item, dict) or "column" not in item or "operator" not in item:
            raise agents.ToolError(
                'Each filter must be a dict with "column" and "operator" keys, and '
                'optionally "value" or "values".'
            )
        _require_column(dataset, item["column"])
        if item["operator"] not in OPERATORS:
            raise agents.ToolError(
                f"Unknown operator {item['operator']!r}. Available operators: "
                f"{', '.join(OPERATORS)}."
            )
        validated.append(item)
    return validated


def _filtered_rows(dataset: Dataset, filters: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    validated = _validate_filters(dataset, filters)
    if not validated:
        return dataset.rows
    return [
        row
        for row in dataset.rows
        if all(
            _match(row.get(item["column"]), item["operator"], item.get("value"), item.get("values"))
            for item in validated
        )
    ]


def _aggregate(values: list[Any], fn: str) -> Any:
    non_null = [value for value in values if value is not None]
    if fn == "count":
        return len(non_null)
    if not non_null:
        return None
    numeric = [value for value in non_null if isinstance(value, (int, float))]
    if fn in ("sum", "mean", "median"):
        if not numeric:
            raise agents.ToolError(f"Cannot compute '{fn}': the column has no numeric values.")
        if fn == "sum":
            return sum(numeric)
        if fn == "mean":
            return sum(numeric) / len(numeric)
        ordered = sorted(numeric)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2
    if fn in ("min", "max"):
        try:
            return min(non_null) if fn == "min" else max(non_null)
        except TypeError:
            raise agents.ToolError(
                f"Cannot compute '{fn}': the column mixes types that cannot be ordered."
            ) from None
    return len({str(value) for value in non_null})


@agents.tool(model_access="direct")
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    file_id_or_url: Annotated[
        str,
        Field(
            description=(
                "Google Drive file id or link, such as "
                "https://drive.google.com/file/d/<FILE_ID>/view."
            )
        ),
    ],
    folder: Annotated[
        str,
        Field(
            description=(
                "Name of the Google Drive folder that contains the file, exactly as "
                "your Drive connector tools report it. Only 'data' is accepted."
            )
        ),
    ],
    file_name: Annotated[
        str | None,
        Field(description="The file's name, when you know it from Google Drive."),
    ] = None,
    csv_text: Annotated[
        str | None,
        Field(
            description=(
                "The CSV's full text content. Provide it only when the direct "
                "download failed and you read the file with the Google Drive "
                "connector."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Load a CSV file from Google Drive into this session for analysis.

    Only files inside the Google Drive folder named 'data' may be loaded: check
    where the file lives with your Google Drive connector tools first, and pass
    that folder as the folder argument. Call it with just the Drive link or file
    id and the folder first: the tool downloads the file directly when it is
    shared by link. If the download fails, read the file's content with your
    Google Drive connector tools and call load_csv again, passing that content
    as csv_text. Loading a new file replaces the previously loaded one. The
    result summarizes the file and its columns.
    """
    if folder.strip().lower() != ALLOWED_FOLDER:
        raise agents.ToolError(
            f"This agent may only analyze CSV files inside the Google Drive folder "
            f"named '{ALLOWED_FOLDER}'. This file is in {folder!r}. Refuse politely "
            "and ask the user to move the file into that folder, or to point at a "
            "file that is already there."
        )
    file_id = extract_file_id(file_id_or_url)
    if csv_text is not None:
        data = csv_text.encode("utf-8")
    else:
        data = _download_csv(file_id)
    columns, rows = parse_csv(data.decode("utf-8-sig"))
    dataset = Dataset(
        file_id=file_id,
        file_name=file_name or file_id,
        columns=columns,
        rows=rows,
    )
    _DATASETS[ctx.session.id] = dataset
    column_report = []
    for column in columns:
        values = [row.get(column) for row in rows]
        non_null = [value for value in values if value is not None]
        column_report.append(
            {
                "name": column,
                "type": _column_type(non_null),
                "nulls": len(values) - len(non_null),
                "example": non_null[0] if non_null else None,
            }
        )
    return {
        "loaded": True,
        "file_name": dataset.file_name,
        "file_id": dataset.file_id,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": column_report,
    }


@agents.tool(model_access="direct")
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Describe every column of the loaded CSV.

    Returns, per column: inferred type, non-null and null counts, distinct
    value count, and min/mean/max for numeric columns or the most frequent
    values for text columns. Use it before the first question about the data
    and whenever column names or value formats are unclear.
    """
    dataset = _dataset_for(ctx.session.id)
    report = []
    for column in dataset.columns:
        values = [row.get(column) for row in dataset.rows]
        non_null = [value for value in values if value is not None]
        info: dict[str, Any] = {
            "name": column,
            "type": _column_type(non_null),
            "non_null": len(non_null),
            "nulls": len(values) - len(non_null),
            "unique": len({str(value) for value in non_null}),
        }
        numeric = [value for value in non_null if isinstance(value, (int, float))]
        if non_null and len(numeric) == len(non_null):
            info["min"] = min(numeric)
            info["max"] = max(numeric)
            info["mean"] = sum(numeric) / len(numeric)
        else:
            info["most_frequent"] = [
                {"value": value, "count": count}
                for value, count in Counter(str(v) for v in non_null).most_common(5)
            ]
        report.append(info)
    return {
        "dataset": _dataset_info(dataset),
        "row_count": len(dataset.rows),
        "columns": report,
    }


@agents.tool(model_access="direct")
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: Annotated[str, Field(description="Exact column name.")],
    operator: Annotated[
        str,
        Field(
            description=(
                "One of: eq, ne, gt, gte, lt, lte, contains, starts_with, ends_with, "
                "in, is_null, not_null. String comparisons are case-insensitive."
            )
        ),
    ],
    value: Annotated[
        Any,
        Field(
            description=(
                "Comparison value. Required by every operator except in, is_null "
                "and not_null."
            )
        ),
    ] = None,
    values: Annotated[
        list[Any] | None,
        Field(description="List of candidate values, only used by the in operator."),
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of example rows to return, 1-50.")
    ] = 20,
) -> dict[str, Any]:
    """Count the rows that match one condition and return example rows.

    Returns the number of matching rows, the total row count, and up to
    `limit` example rows with all their columns.
    """
    dataset = _dataset_for(ctx.session.id)
    _require_column(dataset, column)
    if operator not in OPERATORS:
        raise agents.ToolError(
            f"Unknown operator {operator!r}. Available operators: {', '.join(OPERATORS)}."
        )
    limit = max(1, min(int(limit), 50))
    matching = [
        row for row in dataset.rows if _match(row.get(column), operator, value, values)
    ]
    return {
        "dataset": _dataset_info(dataset),
        "matched": len(matching),
        "total_rows": len(dataset.rows),
        "returned": min(len(matching), limit),
        "rows": matching[:limit],
    }


@agents.tool(model_access="direct")
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[
        list[str], Field(description="One or more column names to group by.")
    ],
    aggregations: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                'Each item is a dict like {"column": "revenue", "fn": "sum"}, with '
                "fn one of: count, sum, mean, median, min, max, nunique. count does "
                'not need a column. An optional "alias" names the result key, '
                'otherwise it is "fn_column", e.g. "sum_revenue".'
            )
        ),
    ],
    filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                'Optional conditions applied before grouping, each like '
                '{"column": "country", "operator": "eq", "value": "France"}, using '
                "the same operators as filter_rows."
            )
        ),
    ] = None,
    order_by: Annotated[
        str | None,
        Field(description="Result key to sort groups by, such as 'sum_revenue'."),
    ] = None,
    descending: Annotated[bool, Field(description="Sort groups descending.")] = True,
    limit: Annotated[
        int, Field(description="Maximum number of groups to return, 1-200.")
    ] = 50,
) -> dict[str, Any]:
    """Group rows and compute aggregates per group, optionally after filtering.

    Returns one entry per group with the group's columns and the requested
    aggregates, plus the number of groups and the rows used.
    """
    dataset = _dataset_for(ctx.session.id)
    if not group_by:
        raise agents.ToolError("group_by needs at least one column name.")
    for column in group_by:
        _require_column(dataset, column)
    if not aggregations:
        raise agents.ToolError("Provide at least one aggregation.")
    rows = _filtered_rows(dataset, filters)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_by)
        groups.setdefault(key, []).append(row)
    results = []
    for key, group_rows in groups.items():
        item: dict[str, Any] = {
            column: value for column, value in zip(group_by, key)
        }
        for aggregation in aggregations:
            if not isinstance(aggregation, dict) or "fn" not in aggregation:
                raise agents.ToolError(
                    'Each aggregation must be a dict with a "fn" key and optionally '
                    'a "column" and an "alias".'
                )
            fn = aggregation["fn"]
            column = aggregation.get("column")
            if fn not in AGGREGATIONS:
                raise agents.ToolError(
                    f"Unknown aggregation {fn!r}. Available aggregations: "
                    f"{', '.join(AGGREGATIONS)}."
                )
            alias = aggregation.get("alias") or (
                "count" if fn == "count" and not column else f"{fn}_{column}"
            )
            if fn == "count":
                item[alias] = len(group_rows)
            else:
                _require_column(dataset, column)
                item[alias] = _aggregate([row.get(column) for row in group_rows], fn)
        results.append(item)
    if order_by is not None:
        try:
            ranked = [item for item in results if item.get(order_by) is not None]
            unranked = [item for item in results if item.get(order_by) is None]
            ranked.sort(key=lambda item: item.get(order_by), reverse=descending)
            results = ranked + unranked
        except TypeError:
            raise agents.ToolError(
                f"Cannot sort by {order_by!r}: its values mix types that cannot be "
                "ordered."
            ) from None
    limit = max(1, min(int(limit), 200))
    return {
        "dataset": _dataset_info(dataset),
        "rows_used": len(rows),
        "group_count": len(results),
        "returned": min(len(results), limit),
        "groups": results[:limit],
    }


@agents.tool(model_access="direct")
async def preview_rows(
    ctx: agents.ToolCallbackContext,
    /,
    offset: Annotated[int, Field(description="Number of rows to skip.")] = 0,
    limit: Annotated[
        int, Field(description="Maximum number of rows to return, 1-50.")
    ] = 5,
) -> dict[str, Any]:
    """Show a slice of the loaded rows to inspect raw values."""
    dataset = _dataset_for(ctx.session.id)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 50))
    return {
        "dataset": _dataset_info(dataset),
        "total_rows": len(dataset.rows),
        "offset": offset,
        "returned": min(len(dataset.rows) - offset, limit),
        "rows": dataset.rows[offset : offset + limit],
    }
