"""News category classifier — Sports, Politics, Business, Technology."""

from .categories import CATEGORY_GUIDE, Category, Classification
from .classifier import NewsClassifier
from .cli import main
from .examples import FEW_SHOT_EXAMPLES

__all__ = [
    "CATEGORY_GUIDE",
    "Category",
    "Classification",
    "FEW_SHOT_EXAMPLES",
    "NewsClassifier",
    "main",
]
