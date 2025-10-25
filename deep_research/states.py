# states.py
import operator
from dataclasses import dataclass, field
from typing import List, Optional
from typing_extensions import Annotated

# Define the states 
@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = ""                         # Report topic
    search_query: str = ""                           # Search query
    rationale: str = ""                              # Rationale for the search query
    web_research_results: Annotated[List[str], operator.add] = field(default_factory=list)
    sources_gathered:  Annotated[List[str], operator.add] = field(default_factory=list)
    images:            Annotated[List[str], operator.add] = field(default_factory=list)  # ✅ add this
    research_loop_count: int = 0                     # Research loop count
    running_summary: str = ""                        # Rolling/final summary
    knowledge_gap: str = ""                          # Knowledge gap
    websocket_id: Optional[str] = None               # (only if you still use websockets)
    thoughts: str = ""                               # Model thoughts (if you surface them)

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = ""
    websocket_id: Optional[str] = None

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = ""
