"""Tests for structured JSON outputs and Pydantic schema guarantees."""

import json
import pytest
from backend.app.core.llm_provider import repair_and_extract_json
from backend.app.schemas.chat import (
    AnalysisReport,
    MeetingNotesExtraction,
    StructuredOutputRequest,
)


def test_repair_and_extract_json_clean():
    """Verify parser parses clean JSON."""
    raw = '{"summary": "Test", "sentiment": "positive", "key_insights": [], "recommended_actions": [], "confidence_score": 0.9}'
    parsed = repair_and_extract_json(raw)
    assert parsed["summary"] == "Test"
    assert parsed["sentiment"] == "positive"


def test_repair_and_extract_json_with_markdown_fences():
    """Verify parser extracts JSON wrapped in Markdown code blocks."""
    raw = """
    Here is your requested JSON output:
    ```json
    {
      "summary": "Extracted with markdown fences",
      "sentiment": "neutral",
      "key_insights": [
        {"title": "Insight 1", "description": "Details", "impact_level": "medium"}
      ],
      "recommended_actions": ["Deploy"],
      "confidence_score": 0.95
    }
    ```
    Thank you for using the AI Assistant!
    """
    parsed = repair_and_extract_json(raw)
    report = AnalysisReport.model_validate(parsed)
    assert report.summary == "Extracted with markdown fences"
    assert report.sentiment == "neutral"
    assert len(report.key_insights) == 1
    assert report.confidence_score == 0.95


def test_meeting_notes_schema_validation():
    """Verify validation of meeting notes structured extraction."""
    data = {
        "topic": "Architecture Review",
        "attendees": ["Alice", "Bob"],
        "key_decisions": ["Use vLLM for serving", "Adopt ChromaDB"],
        "action_items": [{"task": "Write Dockerfile", "assignee": "Alice", "priority": "high"}],
    }
    validated = MeetingNotesExtraction.model_validate(data)
    assert validated.topic == "Architecture Review"
    assert len(validated.action_items) == 1
    assert validated.action_items[0].priority == "high"
