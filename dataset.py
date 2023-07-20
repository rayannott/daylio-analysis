import csv
from dataclasses import dataclass
import datetime
import pathlib
from collections import Counter
from typing import Container, Iterator


MOOD_VALUES = {'bad': 1, 'meh': 2, 'ok': 3, 'good': 4, 'great': 5}
DT_FORMAT = r"%Y-%m-%d %H:%M"

MoodCondition = int | Container[int] | None


@dataclass
class Entry:
    full_date: datetime.datetime
    mood: int
    activities: set
    note: str

    def __repr__(self) -> str:
        return f'[{self.full_date.strftime(DT_FORMAT)}] {self.mood} {", ".join(self.activities)}'

    def get_note(self) -> str:
        return self.note
    
    def check_condition(self, incl_act: set[str],
                   excl_act: set[str], 
                   when: datetime.date | None, 
                   mood: MoodCondition) -> bool:
        '''
        Checks if an entry (self) fulfils all of the following conditions:
            has an activity from incl_act
            does not have an activity from excl_act
            is recorded on a particular day
            matches the mood (an exact value or a container of values)
        '''
        if incl_act & excl_act:
            raise ValueError(f'Some activities are included and excluded at the same time:\n{incl_act=}\n{excl_act=}')
        return (True if not incl_act else bool(incl_act & self.activities)) and \
            (not excl_act & self.activities) and \
            (True if when is None else self.full_date.date() == when) and \
            (True if mood is None else (mood == self.mood if isinstance(mood, int) else self.mood in mood))


class Dataset:
    def _from_csv_file(self, csv_file_path: str | pathlib.Path):
        self.entries: list[Entry] = []
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.entries.append(self._get_entry(row))

    def __init__(self, *, csv_file_path: str | pathlib.Path | None = None, entries: list[Entry] | None = None) -> None:
        if entries is not None:
            self.entries = entries
        elif csv_file_path is not None:
            self._from_csv_file(csv_file_path)
            print(self)
        else:
            self.entries = []
    
    def __repr__(self) -> str:
        return f'Dataset({len(self.entries)} entries)'

    def __getitem__(self, idx: int) -> Entry:
        return self.entries[idx]
    
    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def _get_entry(self, row: dict[str, str]) -> Entry:
        datetime_str = row['full_date'] + ' ' + row['time']
        return Entry(
            full_date=datetime.datetime.strptime(datetime_str, DT_FORMAT),
            mood=MOOD_VALUES[row['mood']],
            activities=set(row['activities'].split(' | ')),
            note=row['note']
        )
    
    def sub(self, incl_act: set[str] = set(),
                   excl_act: set[str] = set(), 
                   when: datetime.date | None = None, 
                   mood: MoodCondition = None) -> 'Dataset':
        '''
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments
        '''
        filtered_entries = []
        for e in self:
            if e.check_condition(incl_act, excl_act, when, mood):
                filtered_entries.append(e)
        return Dataset(entries=filtered_entries)
    
    def count(self, incl_act: set[str] = set(),
                excl_act: set[str] = set(), 
                when: datetime.date | None = None, 
                mood: MoodCondition = None) -> int:
        '''
        Counts the number of entries that fulfil the conditions.
        '''
        return sum(1 for e in self if e.check_condition(incl_act, excl_act, when, mood))
    
    def avg_mood(self) -> float:
        return sum(e.mood for e in self)/len(self.entries)
    
    def activities(self) -> Counter[str]:
        '''
        Returns a Counter object for all activities in the dataset.
        Use `self.activities().keys()` to get only the set of all activities.
        '''
        c = Counter()
        for e in self:
            c.update(e.activities)
        return c
    
    def get_datetimes(self) -> list[datetime.datetime]:
        return [e.full_date for e in self]

    def head(self, n: int = 5) -> None:
        print(self)
        for e in self.entries[:n]:
            print(e)
        if len(self.entries) > n:
            print('...')
    
    def analyse(self, activity: str):
        df_with = self.sub(incl_act={activity})
        df_without = self.sub(excl_act={activity})
        mood_with, mood_without = df_with.avg_mood(), df_without.avg_mood()
        #! add more code here