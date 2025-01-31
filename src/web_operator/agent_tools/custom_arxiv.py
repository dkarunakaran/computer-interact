"""Util that calls Arxiv."""

import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

from langchain_community.utilities.arxiv import ArxivAPIWrapper


#File got modified from this link:https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/arxiv.py


class CustomArxivAPIWrapper(ArxivAPIWrapper):
    """Wrapper around the Arxiv API."""
    
    top_k_results: int = 10
   