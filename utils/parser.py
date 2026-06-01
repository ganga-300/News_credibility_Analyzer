import json
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def parse_json_response(response_text: str, default_fallback: dict = None) -> dict:
    """Safely parse JSON from an LLM response, handling potential formatting issues."""
    if default_fallback is None:
        default_fallback = {}
        
    try:
        # First attempt: direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Second attempt: try to extract JSON from markdown block
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Find first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    json_str = response_text[start:end+1]
                else:
                    raise ValueError("No JSON object found")
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {response_text}. Error: {str(e)}")
            return default_fallback
