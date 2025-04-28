import requests
import random

from src.dataset import Dataset
from src.entry import Entry
from src.tag import Tag


def load_words() -> list[str]:
    URL = (
        "https://gist.githubusercontent.com/"
        + "DevilXD/6ad6cc1fe37872d069a795edd51233b2/"
        + "raw/23d6fb72a489d66ab7a546043f8f88fafa3d640c/"
        + "wordle_words.txt"
    )
    response = requests.get(URL)
    response.raise_for_status()
    words = list(map(str.lower, response.text.split()))
    return words


def generate_mock_dataset_from(df: Dataset) -> Dataset:
    words = load_words()
    activities = list(df.activities().keys())
    random.seed(42)
    activity_to_word_map = dict(zip(activities, random.sample(words, len(activities))))

    def replace_activities(activities: set[str]) -> set[str]:
        return {
            activity_to_word_map[activity].capitalize()
            if activity[0].isupper()
            else activity_to_word_map[activity]
            for activity in activities
        }

    encoded_entries = []
    for entry in df.entries:
        tags = [
            Tag(tag.tag, "*" * len(tag.title), "*" * len(tag.body), entry)
            for tag in entry._tags
        ]
        encoded_entries.append(
            Entry(
                full_date=entry.full_date,
                mood=entry.mood + random.uniform(-2, 2),
                activities=replace_activities(entry.activities),
                note="*" * len(entry.note),
                _tags=tags,
            )
        )
    return Dataset(_entries=encoded_entries)
