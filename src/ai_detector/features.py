import re
import nltk
import numpy as np
from lexical_diversity import lex_div as ld
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


FEATURE_NAMES = [
    'word_count',
    'sentence_count',
    'avg_sentence_length',
    'sentence_length_variance',
    'avg_word_length',
    'type_token_ratio',
    'mtld',
    'punctuation_rate',
    'comma_rate',
    'em_dash_rate',
    'uppercase_ratio',
    'digit_ratio',
    'stopword_ratio'
]


def ensure_nltk_resources() -> None:
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


ensure_nltk_resources()
ENGLISH_STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', str(text)).lower()


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0


def extract_stylometry(text: str) -> list[float]:
    text = str(text)

    sentences = split_sentences(text)
    cleaned_text = clean_text(text)
    words = word_tokenize(cleaned_text)

    sentence_lengths = [
        len(word_tokenize(clean_text(sentence)))
        for sentence in sentences
    ]

    character_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)

    punctuation_count = len(re.findall(r'[^\w\s]', text))
    comma_count = text.count(',')
    em_dash_count = text.count('—')
    uppercase_count = sum(character.isupper() for character in text)
    digit_count = sum(character.isdigit() for character in text)
    stopword_count = sum(word in ENGLISH_STOPWORDS for word in words)

    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'sentence_length_variance': np.var(sentence_lengths) if sentence_lengths else 0,
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'type_token_ratio': safe_divide(len(set(words)), word_count),
        'mtld': ld.mtld(words) if len(words) >= 10 else 0,
        'punctuation_rate': safe_divide(punctuation_count, character_count),
        'comma_rate': safe_divide(comma_count, character_count),
        'em_dash_rate': safe_divide(em_dash_count, character_count),
        'uppercase_ratio': safe_divide(uppercase_count, character_count),
        'digit_ratio': safe_divide(digit_count, character_count),
        'stopword_ratio': safe_divide(stopword_count, word_count)
    }

    return [
        float(np.nan_to_num(features[name], nan=0, posinf=0, neginf=0))
        for name in FEATURE_NAMES
    ]


def extract_stylometry_batch(texts: list[str]) -> np.ndarray:
    features = [extract_stylometry(text) for text in texts]
    return np.array(features, dtype=np.float32)