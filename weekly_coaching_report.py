"""Export a weekly coaching report from the pipeline's alarm CSVs.

Run from the repository folder after downloading the cache files with Git LFS:
    python weekly_coaching_report.py

Use --help to supply other alarm CSVs or choose where to save the report.
"""

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from atrisk_pipeline import ALARM_FILES, ALARM_TYPES, CACHE_DIR


SOURCE_COLUMNS = {
    "Driver": "Driver ID",
    "Alarm type": "Alarm type",
    "Alarm start time": "Alarm date",
}
MIN_ALARM_DAYS = 3
DISCUSSION_POINTS = {
    "Harsh Acceleration": "Review pull-away and acceleration habits.",
    "Harsh Braking": "Review anticipation, following distance and smooth braking.",
    "Speeding": "Review speed limits and how speed is managed on the route.",
    "Swaying": "Review steering inputs and lane positioning.",
    "Sharp Cornering": "Review approach speed and steering through turns.",
    "Fatigue Alert": "Discuss rest, breaks and when the alerts occurred.",
    "Yawn": "Discuss rest and whether the alerts coincided with fatigue.",
    "Headway Monitoring Alert": "Review following distance in traffic.",
    "Forward Collision Warning": "Review anticipation and the traffic context of the alerts.",
    "Pedestrian Collision Alert": "Review scanning and approach speed near crossings and stops.",
}


def load_alarm_days(paths):
    """Keep one observation per driver, alarm type and calendar day."""
    daily_frames = []
    for path in paths:
        with path.open("rb") as source:
            if source.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path} is a Git LFS pointer. Run git lfs pull to download it.")

        chunks = pd.read_csv(
            path, usecols=SOURCE_COLUMNS.keys(), dtype="string", chunksize=250_000
        )
        with chunks:
            for chunk in chunks:
                days = chunk.rename(columns=SOURCE_COLUMNS)
                driver = pd.to_numeric(days["Driver ID"], errors="coerce")
                days["Driver ID"] = driver.where(driver.mod(1).eq(0)).astype("Int64")
                days["Alarm type"] = days["Alarm type"].str.strip()
                days["Alarm date"] = pd.to_datetime(
                    days["Alarm date"], format="mixed", errors="coerce"
                ).dt.normalize()
                usable_days = days.loc[days["Alarm type"].isin(ALARM_TYPES)].dropna()
                daily_frames.append(usable_days.drop_duplicates())

    return pd.concat(daily_frames, ignore_index=True).drop_duplicates()


def build_report(alarm_days):
    alarm_days = alarm_days.sort_values("Alarm date").assign(
        **{"Week starting": alarm_days["Alarm date"].dt.to_period("W-SUN").dt.start_time}
    )
    report = (
        alarm_days.groupby(["Week starting", "Driver ID", "Alarm type"], sort=False)
        .agg(alarm_days=("Alarm date", "size"), alarm_dates=("Alarm date", list))
        .reset_index()
    )
    report = report.loc[report["alarm_days"] >= MIN_ALARM_DAYS].copy()
    report["alarm_dates"] = report["alarm_dates"].map(
        lambda dates: ", ".join(day.strftime("%d %b") for day in dates)
    )
    report["Discussion point"] = report["Alarm type"].map(DISCUSSION_POINTS)
    return report.sort_values(
        ["Week starting", "alarm_days", "Driver ID", "Alarm type"],
        ascending=[True, False, True, True],
    )


def export_report(report, alarm_days, path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Weekly coaching"
    sheet.append(["Weekly coaching report"])
    sheet["A1"].font = Font(bold=True, size=16, color="17365D")
    coverage = "No usable alarm records."
    if not alarm_days.empty:
        dates = alarm_days["Alarm date"]
        coverage = f"Usable alarm dates: {dates.min():%d %b %Y} to {dates.max():%d %b %Y}."
    sheet.append([f"{coverage} Weeks run Monday to Sunday. First or last weeks may be partial."])
    sheet.append([
        f"Review a driver and alarm type when it appears on at least {MIN_ALARM_DAYS} distinct days in a week."
    ])
    sheet.append(["Use alerts as discussion prompts. Review the operating context with the driver."])
    if report.empty:
        sheet.append(["No driver and alarm type met the review threshold."])
    for row in range(1, sheet.max_row + 1):
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    sheet.append(["Week starting", "Driver ID", "Alarm type", "Alarm days", "Dates", "Discussion point"])
    header_row = sheet.max_row
    for cell in sheet[header_row]:
        cell.fill = PatternFill("solid", fgColor="17365D")
        cell.font = Font(bold=True, color="FFFFFF")

    for row, entry in enumerate(report.itertuples(index=False, name=None), start=header_row + 1):
        week, driver, alarm, day_count, dates, prompt = entry
        sheet.append([week.date(), int(driver), alarm, int(day_count), dates, prompt])
        sheet.cell(row, 1).number_format = "dd mmm yyyy"
        sheet.cell(row, 6).alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[row].height = 30
    for index, width in enumerate((17, 14, 30, 14, 62, 60), start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = f"C{header_row + 1}"
    sheet.auto_filter.ref = f"A{header_row}:F{sheet.max_row}"
    workbook.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "alarm_csvs", nargs="*", type=Path,
        help="Alarm CSV files. Defaults to the existing March CSVs in cache/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path.home() / "weekly_coaching_report.xlsx",
        help="Destination .xlsx file. Defaults to weekly_coaching_report.xlsx in your home folder.",
    )
    args = parser.parse_args()
    paths = args.alarm_csvs or [CACHE_DIR / Path(name).with_suffix(".csv") for name in ALARM_FILES]
    if args.output.suffix.lower() != ".xlsx":
        parser.error("--output must end in .xlsx")

    try:
        alarm_days = load_alarm_days(paths)
        report = build_report(alarm_days)
        export_report(report, alarm_days, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {len(report):,} weekly coaching entries to {args.output.resolve()}")


if __name__ == "__main__":
    main()
