"""Quarto-based presentation generation tool.

This module provides tools for generating PowerPoint presentations using Quarto.
Quarto is more flexible than python-pptx and handles complex formatting better.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union
from langchain_core.tools import tool

PROJECT_DIR = Path(__file__).parent.parent
PRESENTATIONS_DIR = PROJECT_DIR / "presentations"
PRESENTATIONS_DIR.mkdir(exist_ok=True)


@tool(parse_docstring=True)
def generate_presentation_quarto(
    slides_data: Union[str, list, dict], 
    title: str = "Presentation",
    output_format: str = "pptx"
) -> str:
    """Generate PowerPoint presentation using Quarto from slide specifications.

    Creates a professional PowerPoint presentation by generating a Quarto (.qmd) file
    and rendering it to the specified format (PowerPoint by default).
    
    Args:
        slides_data: Either a JSON string or Python list of slide objects
        title: Title for the presentation (used for output filename)
        output_format: Output format: "pptx" (default), "pdf", or "html"

    Returns:
        Success message with file path to generated presentation or error details
    """
    try:
        # Parse slides_data if it's a string
        if isinstance(slides_data, str):
            # Handle double-escaped JSON
            cleaned = slides_data.replace('\\"', '"') if '\\"' in slides_data else slides_data
            slides = json.loads(cleaned)
        elif isinstance(slides_data, (list, dict)):
            slides = slides_data
        else:
            return "❌ Error: slides_data must be a JSON string, list, or dictionary"

        # Validate slides structure
        if not isinstance(slides, list) or len(slides) == 0:
            return "❌ Error: slides must be a non-empty array of slide objects"

        for i, slide in enumerate(slides):
            if not isinstance(slide, dict):
                return f"❌ Error: Slide {i} is not an object"
            if "title" not in slide or "content" not in slide:
                return f"❌ Error: Slide {i} missing 'title' or 'content' fields"
            if not isinstance(slide["title"], str):
                return f"❌ Error: Slide {i} title must be a string"
            if not isinstance(slide["content"], list):
                return f"❌ Error: Slide {i} content must be an array"

        # Generate Quarto markdown
        qmd_content = _generate_quarto_markdown(slides, title)

        # Create temporary directory for Quarto file
        output_name = title.replace(" ", "_").lower()
        qmd_path = PRESENTATIONS_DIR / f"{output_name}.qmd"

        # Write the .qmd file
        qmd_path.write_text(qmd_content, encoding="utf-8")

        # Determine output extension
        ext_map = {"pptx": "pptx", "pdf": "pdf", "html": "html"}
        output_ext = ext_map.get(output_format.lower(), "pptx")

        # Run Quarto render
        try:
            result = subprocess.run(
                ["quarto", "render", str(qmd_path)],
                cwd=str(PRESENTATIONS_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return f"❌ Quarto rendering failed: {result.stderr}"

            # Verify output file was created
            output_path = PRESENTATIONS_DIR / f"{output_name}.{output_ext}"
            if output_path.exists():
                # Use relative path for display to avoid confusing agent with absolute paths
                relative_path = Path("presentations") / f"{output_name}.{output_ext}"
                return (
                    f"✅ Presentation generated successfully!\n"
                    f"📊 File: {relative_path}\n"
                    f"📄 Format: {output_format.upper()}\n"
                    f"📊 Slides: {len(slides)}\n"
                    f"The presentation is ready to download from the Streamlit interface."
                )
            else:
                return f"⚠️ Rendering completed but output file not found. Expected at presentations/{output_name}.{output_ext}"

        except subprocess.TimeoutExpired:
            return "❌ Error: Quarto rendering timed out (exceeded 60 seconds)"
        except FileNotFoundError:
            return "❌ Error: Quarto not found. Please install Quarto: https://quarto.org/docs/get-started/"

    except json.JSONDecodeError as e:
        return f"❌ JSON parse error: {str(e)}"
    except Exception as e:
        return f"❌ Error generating presentation: {str(e)}"


def _generate_quarto_markdown(slides: list, title: str) -> str:
    """Generate Quarto markdown (.qmd) content from slide specifications.
    
    Args:
        slides: List of slide dictionaries with title and content
        title: Presentation title
        
    Returns:
        String containing Quarto markdown
    """
    # Quarto YAML header for PowerPoint presentation
    qmd = f"""---
title: "{title}"
format: pptx
---

"""

    # Add slides
    for slide in slides:
        slide_title = slide.get("title", "Untitled")
        content = slide.get("content", [])

        # Add slide title (level 2 heading creates a new slide in Quarto)
        qmd += f"## {slide_title}\n\n"

        # Add bullet points
        if isinstance(content, list):
            for item in content:
                # Escape special characters
                item_text = str(item).replace('"', '\\"')
                qmd += f"- {item_text}\n"
        else:
            qmd += f"- {content}\n"

        qmd += "\n"

    return qmd