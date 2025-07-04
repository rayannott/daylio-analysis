import datetime
from dataclasses import dataclass, field

from src.tag import Tag
from src.entry_condition import EntryCondition
from src.utils import (
    DT_FORMAT_READ,
    DT_FORMAT_SHOW,
    MOOD_VALUES,
    NoteCondition,
    IncludeExcludeActivities,
    EntryPredicate,
)


@dataclass
class Entry:
    full_date: datetime.datetime
    mood: float
    activities: set[str]
    note: str
    _tags: list[Tag] = field(default_factory=list)

    def __post_init__(self):
        self._tags = list(Tag.pull_tags(self))

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "Entry":
        """Construct an Entry object from a dictionary with the keys as in the CSV file."""
        datetime_str = row["full_date"] + " " + row["time"]
        return cls(
            full_date=datetime.datetime.strptime(datetime_str, DT_FORMAT_READ),
            mood=MOOD_VALUES[row["mood"]],
            activities=set(row["activities"].split(" | "))
            if row["activities"]
            else set(),
            note=row["note"].replace("<br>", "\n").replace("&nbsp;", ""),
        )

    def __repr__(self) -> str:
        return f"[{self.full_date.strftime(DT_FORMAT_SHOW)}] {self.mood} {', '.join(sorted(self.activities))}"

    def verbose(self) -> str:
        return f"{self}\n{'{'}{self.note}{'}'}"

    def check_condition(
        self,
        condition: EntryCondition | None,
        include: IncludeExcludeActivities,
        exclude: IncludeExcludeActivities,
        note_pattern: NoteCondition | None,
        predicate: EntryPredicate | None,
    ) -> bool:
        """
        Checks if an entry (self) fulfils all of the following conditions:
            - satisfies a condition (if provided)
            - has an activity from include (if provided)
            - does not have an activity from exclude (if provided)
            - contains a note pattern (if provided)
            - satisfies a predicate (if provided)

        Parameters:
            - condition: an EntryCondition object
            - include: a string or a set of strings
            - exclude: a string or a set of strings
            - note_contains: a string or a container of strings
            - predicate: a function that takes an Entry object and returns a bool

        Returns: bool: True if all conditions are met, False otherwise
        """
        if condition is not None:
            if include or exclude or note_pattern or predicate:
                raise ValueError(
                    "`condition` must be the only condition to check in the entry"
                )
            return condition.check(self)
        if predicate is not None and not predicate(self):
            return False
        if isinstance(include, str):
            include = {include}
        if isinstance(exclude, str):
            exclude = {exclude}
        if include & exclude:
            raise ValueError(
                f"Some activities are included and excluded at the same time: {include & exclude=}"
            )
        note_condition_result = (
            True
            if note_pattern is None
            else any(
                word in self.note.lower()
                for word in (
                    [note_pattern] if isinstance(note_pattern, str) else note_pattern
                )
            )
        )
        return (
            (True if not include else bool(include & self.activities))
            and (not exclude & self.activities)
            and note_condition_result
            and (True if predicate is None else predicate(self))
        )
