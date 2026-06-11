import pathlib

from src.dataset import Dataset
from src.entry_condition import A, DateIn, MoodIn, NoteHas, Predicate, register

DATA_DIR = pathlib.Path("data")
path = next(DATA_DIR.glob("*.csv"))
print("using file", path.name)

df = Dataset(csv_file=path)
register(set(df.activities()))
