import datetime
import pathlib

from src.dataset import Dataset


def generate_report_template(month: int, year: int, dataframe: Dataset):
    _from_dt = datetime.date(year, month, 1)
    _to_dt = (
        _from_dt.replace(month=month + 1)
        if month < 12
        else _from_dt.replace(year=year + 1, month=1)
    )
    _from = f"{_from_dt:%d.%m.%Y}"
    _to = f"{_to_dt:%d.%m.%Y}"

    df_month = dataframe[_from:_to]
    df_month_groups = df_month.group_by("day")

    EVENTS_COMMENT = f"total entries: {len(df_month)}\n\n"
    for day, entries in df_month_groups.items():
        EVENTS_COMMENT += f" -- {day:%d.%m.%Y, %a} --\n"
        for e in entries:
            EVENTS_COMMENT += f"@{e.full_date.time():%H:%M}: {e.mood} {', '.join(e.activities)}\n  {e.note}\n"
        EVENTS_COMMENT += "\n"

    month_word = datetime.date(1900, month, 1).strftime("%B")
    ROOT = pathlib.Path(".")
    PERSONAL = pathlib.Path(
        r"C:\Users\Airat\contents\pythoncode\personal\monthly-reports"
    )
    NEW_FILE_NAME = f"{year}-{month:02d}.md"
    SAVE_TO_ROOT = ROOT / NEW_FILE_NAME
    SAVE_TO_PERSONAL = PERSONAL / NEW_FILE_NAME
    SAVE_TO = SAVE_TO_PERSONAL if PERSONAL.exists() else SAVE_TO_ROOT
    if SAVE_TO.exists():
        print(f"file {SAVE_TO} already exists")
        return
    with open(SAVE_TO, "w", encoding="utf-8") as f:
        f.write(f"# {month_word} {year}\n\n<!---\n{EVENTS_COMMENT}\n--->")
    print(f"saved to {SAVE_TO}")
