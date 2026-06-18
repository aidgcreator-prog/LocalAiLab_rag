#!/usr/bin/env python
"""Demo script for creating presentations using presentation_tools."""

from presentation_tools import create_pptx_presentation, create_pdf_presentation
from pathlib import Path

# Define presentation content
ai_trends_slides = [
    {
        "title": "AI Trends in 2024",
        "content": ["Exploring the Latest Developments", "Machine Learning to Agent Systems"]
    },
    {
        "title": "Machine Learning Fundamentals",
        "content": ["Supervised and unsupervised learning", "Deep learning breakthroughs", "Transfer learning applications"]
    },
    {
        "title": "Transformers Revolution",
        "content": ["Attention mechanisms", "Self-supervised learning", "Foundation models at scale"]
    },
    {
        "title": "Generative AI",
        "content": ["Text generation models", "Image synthesis and creation", "Multimodal capabilities"]
    },
    {
        "title": "Large Language Models (LLMs)",
        "content": ["GPT and similar architectures", "Few-shot learning capabilities", "Prompt engineering techniques"]
    },
    {
        "title": "Multimodal AI Systems",
        "content": ["Vision and language integration", "Audio-visual processing", "Cross-modal understanding"]
    },
    {
        "title": "AI Agent Systems",
        "content": ["Agent orchestration patterns", "Tool integration and planning", "Multi-agent workflows"]
    },
    {
        "title": "Safety and Ethics",
        "content": ["Alignment and safety research", "Bias detection and mitigation", "Responsible AI deployment"]
    },
    {
        "title": "Future Outlook",
        "content": ["Emerging applications", "Scalability challenges", "Regulatory landscape"]
    },
    {
        "title": "Thank You",
        "content": ["Questions and Discussion"]
    },
]

# Create output directory
output_dir = Path("presentations")
output_dir.mkdir(exist_ok=True)

# Generate presentations
print("Creating PPTX presentation...")
pptx_path = create_pptx_presentation(ai_trends_slides, str(output_dir / "AI_Trends.pptx"))
print(f"[OK] Created: {pptx_path}")

print("\nCreating PDF presentation...")
pdf_path = create_pdf_presentation(ai_trends_slides, str(output_dir / "AI_Trends.pdf"))
print(f"[OK] Created: {pdf_path}")

print(f"\n[OK] Presentation created successfully!")
print(f"  - PowerPoint: {pptx_path}")
print(f"  - PDF: {pdf_path}")
print(f"  - Total slides: {len(ai_trends_slides)}")
