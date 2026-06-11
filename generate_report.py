import datetime
import pathlib

from src.analytics.report import generate_full_report, generate_monthly_report
from src.dataset import Dataset

path = next(pathlib.Path("data").glob("*.csv"))
df = Dataset(csv_file=path, remove=True)

p1 = generate_full_report(df)
print(f"Full report: {p1} ({p1.stat().st_size:,} bytes)")

last_month = datetime.date.today().replace(day=1) - datetime.timedelta(days=1)
if (
    input(
        f"Generate monthly report for last month? ({last_month:%d.%m.%Y}) (y/n): "
    ).lower()
    != "y"
):
    exit()

p2 = generate_monthly_report(df, last_month)
print(f"Monthly report: {p2} ({p2.stat().st_size:,} bytes)")
