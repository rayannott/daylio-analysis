import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc


class LazyLanguageTools:
    def __init__(self) -> None:
        self.initialized = False

    def init(self):
        if self.initialized:
            return
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.russian_stopwords = set(stopwords.words("russian"))

        self.english_lemmatizer = WordNetLemmatizer()
        self.english_stopwords = set(stopwords.words("english"))
        self.initialized = True


lang_tools = LazyLanguageTools()


def clean_russian_text(text: str, additional_stopwords: set[str] = set()) -> list[str]:
    lang_tools.init()
    text = re.sub(r"[^а-яё\s]", " ", text.lower())
    doc = Doc(text)
    doc.segment(lang_tools.segmenter)
    doc.tag_morph(lang_tools.morph_tagger)
    if doc.tokens is None:
        return []
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(lang_tools.morph_vocab)
        lemma = token.lemma
        if (
            lemma not in lang_tools.russian_stopwords
            and lemma not in additional_stopwords
            and len(lemma) > 2
        ):
            lemmas.append(lemma)
    return lemmas


def clean_english_text(text: str, additional_stopwords: set[str] = set()) -> list[str]:
    lang_tools.init()

    lang_tools.english_lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmas = [
        lang_tools.english_lemmatizer.lemmatize(token)
        for token in tokens
        if token not in lang_tools.english_stopwords
        and token not in additional_stopwords
        and len(token) > 2
    ]
    return lemmas
