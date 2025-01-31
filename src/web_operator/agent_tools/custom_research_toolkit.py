from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from web_operator.agent_tools.custom_arxiv_query import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun


class CustomResearchToolkit(BaseToolkit):

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
           ArxivQueryRun(),
           PubmedQueryRun()
        ]