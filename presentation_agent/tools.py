"""Presentation Tools.

This module provides tools for generating professional PowerPoint and PDF presentations
from structured slide data using Quarto.
"""

from .quarto_tool import generate_presentation_quarto

# Export the primary tool
generate_presentation = generate_presentation_quarto


__all__ = ["generate_presentation", "generate_presentation_quarto"]
