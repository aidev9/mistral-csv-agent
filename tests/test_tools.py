"""Unit tests for the fixed analysis tools. No model, no API key, no network."""

import asyncio

import pytest
from mistralai import agents

from analyst_tools import (
    describe_columns,
    extract_file_id,
    filter_rows,
    group_by_aggregate,
    load_csv,
    parse_csv,
    preview_rows,
    time_trends,
)

SAMPLE = """name,region,sales,notes
Alice,EMEA,120,repeat
Bob,AMER,80,
Carol,EMEA,200,repeat
Dan,APAC,N/A,new
Eve,AMER,150,repeat
Frank,EMEA,,new
"""

CSV_TEXT = SAMPLE
FILE_ID = "1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8"


def ctx(session_id: str) -> agents.ToolCallbackContext:
    return agents.ToolCallbackContext(
        agents.CallbackSession(id=session_id, turn_id="t1"), tool_call_id="c1"
    )


def load(session_id: str = "s-load") -> dict:
    return asyncio.run(
        load_csv(
            ctx(session_id),
            file_id_or_url=FILE_ID,
            file_name="sales.csv",
            folder="data",
            csv_text=CSV_TEXT,
        )
    )


def test_extract_file_id():
    assert (
        extract_file_id("https://drive.google.com/file/d/1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8/view")
        == "1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8"
    )
    assert (
        extract_file_id("https://drive.google.com/open?id=1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8")
        == "1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8"
    )
    assert extract_file_id("1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8") == "1A2b3C4d5E6f7G8h9I0jK1l2M3n4O5p6Q7r8"
    with pytest.raises(agents.ToolError):
        extract_file_id("not a drive link")


def test_parse_csv_types_and_missing():
    columns, rows = parse_csv(SAMPLE)
    assert columns == ["name", "region", "sales", "notes"]
    assert rows[0]["sales"] == 120
    assert rows[3]["sales"] is None
    assert rows[1]["notes"] is None
    assert rows[3]["notes"] == "new"


def test_parse_csv_sniffs_semicolons():
    _, rows = parse_csv("a;b\n1;2\n3;4\n")
    assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_load_csv_only_accepts_data_folder():
    with pytest.raises(agents.ToolError):
        asyncio.run(
            load_csv(
                ctx("s-folder"),
                file_id_or_url=FILE_ID,
                folder="Downloads",
                csv_text=CSV_TEXT,
            )
        )
    result = asyncio.run(
        load_csv(
            ctx("s-folder"),
            file_id_or_url=FILE_ID,
            folder="Data",
            csv_text=CSV_TEXT,
        )
    )
    assert result["loaded"] is True


def test_load_csv_reports_schema():
    result = load()
    assert result["loaded"] is True
    assert result["row_count"] == 6
    assert result["file_name"] == "sales.csv"
    by_name = {column["name"]: column for column in result["columns"]}
    assert by_name["sales"]["type"] == "number"
    assert by_name["sales"]["nulls"] == 2
    assert by_name["region"]["type"] == "text"


def test_describe_columns():
    load()
    result = asyncio.run(describe_columns(ctx("s-load")))
    by_name = {column["name"]: column for column in result["columns"]}
    assert by_name["sales"]["min"] == 80
    assert by_name["sales"]["max"] == 200
    assert by_name["sales"]["mean"] == pytest.approx(550 / 4)
    assert by_name["region"]["unique"] == 3
    assert {"value": "EMEA", "count": 3} in by_name["region"]["most_frequent"]


def test_filter_rows_numeric_and_string():
    load()
    result = asyncio.run(
        filter_rows(ctx("s-load"), column="region", operator="eq", value="emea")
    )
    assert result["matched"] == 3
    assert result["total_rows"] == 6
    assert all(row["region"] == "EMEA" for row in result["rows"])

    result = asyncio.run(
        filter_rows(ctx("s-load"), column="sales", operator="gte", value="120")
    )
    assert result["matched"] == 3

    result = asyncio.run(filter_rows(ctx("s-load"), column="notes", operator="is_null"))
    assert result["matched"] == 1


def test_filter_rows_unknown_column_is_tool_error():
    load()
    with pytest.raises(agents.ToolError):
        asyncio.run(filter_rows(ctx("s-load"), column="wrong", operator="eq", value="x"))


def test_group_by_aggregate():
    load()
    result = asyncio.run(
        group_by_aggregate(
            ctx("s-load"),
            group_by=["region"],
            aggregations=[
                {"fn": "count"},
                {"column": "sales", "fn": "sum", "alias": "total_sales"},
                {"column": "sales", "fn": "mean"},
            ],
            order_by="total_sales",
            descending=True,
        )
    )
    assert result["rows_used"] == 6
    assert result["group_count"] == 3
    groups = {group["region"]: group for group in result["groups"]}
    assert groups["EMEA"]["count"] == 3
    assert groups["EMEA"]["total_sales"] == 320
    assert groups["AMER"]["total_sales"] == 230
    assert groups["APAC"]["total_sales"] is None
    assert result["groups"][0]["region"] == "EMEA"


def test_group_by_aggregate_with_filters():
    load()
    result = asyncio.run(
        group_by_aggregate(
            ctx("s-load"),
            group_by=["region"],
            aggregations=[{"fn": "count"}],
            filters=[{"column": "notes", "operator": "eq", "value": "repeat"}],
        )
    )
    assert result["rows_used"] == 3
    groups = {group["region"]: group for group in result["groups"]}
    assert groups["EMEA"]["count"] == 2
    assert groups["AMER"]["count"] == 1


def test_group_by_aggregate_multiple_group_columns():
    load()
    result = asyncio.run(
        group_by_aggregate(
            ctx("s-load"),
            group_by=["region", "notes"],
            aggregations=[{"column": "sales", "fn": "sum"}],
        )
    )
    keys = {(group["region"], group["notes"]) for group in result["groups"]}
    assert ("EMEA", "repeat") in keys
    assert ("APAC", "new") in keys


def test_group_by_aggregate_rejects_bad_input():
    load()
    with pytest.raises(agents.ToolError):
        asyncio.run(
            group_by_aggregate(
                ctx("s-load"), group_by=["region"], aggregations=[{"column": "sales", "fn": "stdev"}]
            )
        )
    with pytest.raises(agents.ToolError):
        asyncio.run(
            group_by_aggregate(
                ctx("s-load"), group_by=["wrong"], aggregations=[{"fn": "count"}]
            )
        )


def test_filter_rows_reports_share():
    load()
    result = asyncio.run(
        filter_rows(ctx("s-load"), column="region", operator="eq", value="EMEA")
    )
    assert result["matched"] == 3
    assert result["matched_share"] == pytest.approx(0.5)


def test_group_by_aggregate_reports_count_share():
    load()
    result = asyncio.run(
        group_by_aggregate(
            ctx("s-load"), group_by=["region"], aggregations=[{"fn": "count"}]
        )
    )
    groups = {group["region"]: group for group in result["groups"]}
    assert groups["EMEA"]["count_share"] == pytest.approx(0.5)
    assert groups["AMER"]["count_share"] == pytest.approx(round(2 / 6, 4))


def test_time_trends_monthly_counts():
    tickets = """created_date,priority,hours
2024-01-05,P1,10
2024-01-20,P2,20
2024-02-01,P1,5
2024-02-10,P1,15
not-a-date,P1,1
2024-03-02,P2,30
"""
    asyncio.run(
        load_csv(
            ctx("s-time"),
            file_id_or_url=FILE_ID,
            folder="data",
            file_name="tickets.csv",
            csv_text=tickets,
        )
    )
    result = asyncio.run(time_trends(ctx("s-time"), date_column="created_date"))
    assert result["bucket"] == "month"
    assert result["rows_used"] == 6
    assert result["rows_without_date"] == 1
    assert result["series"] == [
        {"bucket": "2024-01", "value": 2},
        {"bucket": "2024-02", "value": 2},
        {"bucket": "2024-03", "value": 1},
    ]
    assert result["peak"] == {"bucket": "2024-01", "value": 2}
    assert result["metric_mean"] == pytest.approx(1.6667)


def test_time_trends_weekly_metric_with_filter():
    tickets = """created_date,priority,hours
2024-01-01,P1,10
2024-01-02,P1,20
2024-01-08,P2,30
2024-01-09,P1,40
"""
    asyncio.run(
        load_csv(
            ctx("s-week"),
            file_id_or_url=FILE_ID,
            folder="data",
            file_name="tickets.csv",
            csv_text=tickets,
        )
    )
    result = asyncio.run(
        time_trends(
            ctx("s-week"),
            date_column="created_date",
            bucket="week",
            metric_column="hours",
            metric_fn="sum",
            filters=[{"column": "priority", "operator": "eq", "value": "P1"}],
        )
    )
    assert result["rows_used"] == 3
    assert result["series"] == [
        {"bucket": "2024-W01", "value": 30},
        {"bucket": "2024-W02", "value": 40},
    ]
    assert result["metric"] == "sum_hours"


def test_time_trends_rejects_bad_input():
    load()
    with pytest.raises(agents.ToolError):
        asyncio.run(time_trends(ctx("s-load"), date_column="wrong_column"))
    with pytest.raises(agents.ToolError):
        asyncio.run(time_trends(ctx("s-load"), date_column="name", bucket="quarter"))


def test_preview_rows():
    load()
    result = asyncio.run(preview_rows(ctx("s-load"), offset=1, limit=2))
    assert result["returned"] == 2
    assert result["rows"][0]["name"] == "Bob"


def test_tools_fail_cleanly_without_dataset():
    with pytest.raises(agents.ToolError):
        asyncio.run(describe_columns(ctx("s-empty")))
    with pytest.raises(agents.ToolError):
        asyncio.run(filter_rows(ctx("s-empty"), column="a", operator="eq", value="b"))
    with pytest.raises(agents.ToolError):
        asyncio.run(group_by_aggregate(ctx("s-empty"), group_by=["a"], aggregations=[{"fn": "count"}]))


def test_load_replaces_previous_dataset():
    load("s-replace")
    asyncio.run(
        load_csv(
            ctx("s-replace"),
            file_id_or_url=FILE_ID,
            folder="data",
            file_name="other.csv",
            csv_text="x\n1\n",
        )
    )
    result = asyncio.run(describe_columns(ctx("s-replace")))
    assert result["dataset"]["file_name"] == "other.csv"
    assert result["row_count"] == 1
