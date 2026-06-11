import datetime

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator

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


class Entry(BaseModel):
    """A single journal entry, validated directly from a CSV row dict."""

    full_date: datetime.datetime
    mood: float
    activities: set[str]
    note: str

    _tags: list[Tag] = PrivateAttr(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _merge_date_and_time(cls, row: dict[str, str]) -> dict[str, str]:
        return {**row, "full_date": f"{row['full_date']} {row['time']}"}

    @field_validator("full_date", mode="before")
    @classmethod
    def _parse_full_date(cls, value: str) -> datetime.datetime:
        return datetime.datetime.strptime(value, DT_FORMAT_READ)

    @field_validator("mood", mode="before")
    @classmethod
    def _lookup_mood(cls, value: str) -> float:
        return MOOD_VALUES[value]

    @field_validator("activities", mode="before")
    @classmethod
    def _split_activities(cls, value: str) -> set[str]:
        return set(value.split(" | ")) if value else set()

    @field_validator("note", mode="before")
    @classmethod
    def _clean_note(cls, value: str) -> str:
        return value.replace("<br>", "\n").replace("&nbsp;", "")

    def model_post_init(self, _context: object) -> None:
        self._tags = list(Tag.pull_tags(self))

    def __repr__(self) -> str:
        return f"[{self.full_date.strftime(DT_FORMAT_SHOW)}] {self.mood} {', '.join(sorted(self.activities))}"

    __str__ = __repr__

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
