"""Normalize HellaSwag text before selecting fixed capability panels."""

import re


def preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub("\\[.*?\\]", "", text).replace("  ", " ")
