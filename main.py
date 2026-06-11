from src.dataset import Dataset
import pathlib



DATA_DIR = pathlib.Path("data")
path = next(DATA_DIR.glob("*.csv"))
print("using file", path.name)

df = Dataset(csv_file=path)
