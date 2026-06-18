"""Data Science Tools.

This module provides Python code execution tools for data analysis using conda base environment.
Supports data manipulation, visualization, and statistical analysis.

The conda base environment contains pre-installed data science packages (pandas, numpy,
matplotlib, seaborn, scikit-learn, scipy, etc.) so the agent doesn't need to reinstall them.
"""

import os
import subprocess
import json
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache
from langchain_core.tools import tool

# Project directory
PROJECT_DIR = Path(__file__).parent.parent

# Get data scientist model preference (if set)
DATA_SCIENTIST_MODEL = os.getenv("DATA_SCIENTIST_MODEL", "").strip()


@lru_cache(maxsize=1)
def _get_conda_python() -> str:
    """Find conda base Python executable. Falls back to sys.executable."""
    # 1. Check CONDA_BASE_PREFIX (set when any conda env is active)
    conda_base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_BASE_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_base:
        candidate = Path(conda_base) / ("python.exe" if os.name == "nt" else "bin/python")
        if candidate.exists():
            return str(candidate)

    # 2. Run 'conda info --base' to find it
    conda_cmd = shutil.which("conda")
    if conda_cmd:
        try:
            result = subprocess.run(
                [conda_cmd, "info", "--base"],
                capture_output=True, text=True, timeout=10,
            )
            base_dir = result.stdout.strip()
            if base_dir:
                candidate = Path(base_dir) / ("python.exe" if os.name == "nt" else "bin/python")
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass

    # 3. Common default locations
    for default_path in [
        Path.home() / "anaconda3" / ("python.exe" if os.name == "nt" else "bin/python"),
        Path.home() / "miniconda3" / ("python.exe" if os.name == "nt" else "bin/python"),
        Path("C:/Users") / os.getenv("USERNAME", "User") / "anaconda3" / "python.exe",
    ]:
        if default_path.exists():
            return str(default_path)

    # Fallback to current interpreter
    return sys.executable


@lru_cache(maxsize=1)
def _get_conda_pip() -> str:
    """Find pip executable in conda base environment."""
    conda_python = Path(_get_conda_python())
    conda_dir = conda_python.parent
    if os.name == "nt":
        pip_path = conda_dir / "Scripts" / "pip.exe"
    else:
        pip_path = conda_dir / "pip"
    if pip_path.exists():
        return str(pip_path)
    # Fallback: run pip via python -m pip
    return ""


@tool
def execute_python_code(code: str, timeout: int = 30) -> str:
    """Execute Python code in conda base environment.

    Pre-installed libraries include: pandas, numpy, matplotlib, seaborn,
    scikit-learn, scipy, and other common data science packages.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
    """
    try:
        # Create plots directory if it doesn't exist
        plots_dir = PROJECT_DIR / "generated_plots"
        plots_dir.mkdir(exist_ok=True)

        # Add matplotlib backend configuration and auto-save to injected code
        injected_code = """
import os
import time as _time_mod
_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as _mpl_err:
    print(f"[WARN] matplotlib unavailable: {{_mpl_err}}")

# Store current figures count
_initial_figs = set(plt.get_fignums()) if _HAS_MPL else set()
_plots_dir = r'{plots_dir}'
_run_ts = str(int(_time_mod.time() * 1000))

""".format(plots_dir=str(plots_dir).replace('\\', '\\\\'))

        # Combine injected setup code with user code
        full_code = injected_code + code

        # Add code to save all generated figures
        cleanup_code = """

# Save all newly created figures (use timestamp to avoid overwriting previous plots)
_new_figs = sorted(set(plt.get_fignums()) - _initial_figs) if _HAS_MPL else []
# If no new figs detected but figures exist and initial was empty, save all
if not _new_figs and _HAS_MPL and plt.get_fignums() and not _initial_figs:
    _new_figs = sorted(plt.get_fignums())
_plot_files = []
if _HAS_MPL and _new_figs:
    for i, fig_num in enumerate(_new_figs, 1):
        _fig = plt.figure(fig_num)
        _plot_path = os.path.join(_plots_dir, f'plot_{_run_ts}_{i}.png')
        _fig.savefig(_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        _plot_files.append(_plot_path)
    plt.close('all')
    if _plot_files:
        print(f"[PLOT] Saved {len(_plot_files)} plot(s)")
        for pf in _plot_files:
            print(f"  - {os.path.basename(pf)}")
"""

        full_code = full_code + cleanup_code

        # Create temporary Python script
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=PROJECT_DIR,
            encoding='utf-8'
        ) as f:
            f.write(full_code)
            temp_file = f.name

        try:
            # Execute using Python in conda base environment
            conda_python = _get_conda_python()
            result = subprocess.run(
                [conda_python, temp_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                cwd=PROJECT_DIR
            )

            output = result.stdout

            # Capture any errors
            if result.returncode != 0:
                output += f"\n\n[ERROR]\n{result.stderr}"

            # Add model info if custom data scientist model is configured
            if DATA_SCIENTIST_MODEL:
                output += f"\n\n[MODEL] Data Scientist Model: `{DATA_SCIENTIST_MODEL}`"

            if result.returncode == 0:
                output += "\n\n[OK] Code executed successfully"

            return output

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    except subprocess.TimeoutExpired:
        return f"[ERROR] Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"[ERROR] Error executing code: {str(e)}"


@tool
def install_package(package_name: str, version: Optional[str] = None) -> str:
    """Install a Python package in conda base environment.

    The conda base env already has: pandas, numpy, matplotlib, seaborn, scikit-learn,
    scipy, statsmodels, plotly, and many more. Only install if truly needed.

    Args:
        package_name: Name of the package to install (e.g., 'plotly', 'xgboost')
        version: Optional specific version (e.g., '1.0.0'). If not specified, installs latest
    """
    try:
        # First check if the package is already available in conda base
        conda_python = _get_conda_python()
        check = subprocess.run(
            [conda_python, "-c", f"import {package_name.replace('-', '_')}; print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        if check.returncode == 0 and "OK" in check.stdout:
            msg = f"[OK] Package '{package_name}' is already installed in conda base environment."
            if DATA_SCIENTIST_MODEL:
                msg += f"\n[MODEL] Data Scientist Model: `{DATA_SCIENTIST_MODEL}`"
            return msg

        # Not available — install via pip into conda base
        full_package = f"{package_name}=={version}" if version else package_name
        conda_pip = _get_conda_pip()
        if conda_pip:
            cmd = [conda_pip, 'install', full_package, '-q']
        else:
            cmd = [conda_python, '-m', 'pip', 'install', full_package, '-q']

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_DIR
        )

        if result.returncode == 0:
            msg = f"[OK] Successfully installed {full_package} in conda base environment"
            if DATA_SCIENTIST_MODEL:
                msg += f"\n[MODEL] Data Scientist Model: `{DATA_SCIENTIST_MODEL}`"
            return msg
        else:
            return f"[ERROR] Failed to install {full_package}:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return f"[ERROR] Installation timed out"
    except Exception as e:
        return f"[ERROR] Error installing package: {str(e)}"


@tool
def think_tool(thought: str) -> str:
    """Reflect on your data analysis progress and next steps.

    Use this to pause and think about:
    - What have you learned from the data so far?
    - What patterns or insights have you found?
    - What additional analysis is needed?
    - How to best present the findings?

    Args:
        thought: Your reflection on the analysis so far
    """
    return f"[THINK] Reflection recorded: {thought}\n\nContinuing with next analysis steps..."


@tool(parse_docstring=True)
def render_quarto_report(
    report_spec: str,
    output_filename: str = "",
) -> str:
    """Render a structured data analysis report as a .docx file using Quarto.

    Generates a Quarto (.qmd) document from the provided report specification and
    renders it to a Word (.docx) file.  All section text supports Markdown formatting.
    Plot file references (from generated_plots/) are embedded as figures.

    Args:
        report_spec: JSON string with the full report structure.  Required top-level keys:

            title (str): Report title.
            toc (list[str] | null): Explicit Table of Contents entries provided by the user.
                If null or empty the agent must supply an auto-generated list.
            abstract (str): Concise summary paragraph.
            introduction (str): Overview of methods, scope, and dataset context.
            eda_sections (list): Each item is a dict with:
                - heading (str): Subsection title.
                - body (str): Statistical explanation (Markdown).
                - plot_files (list[str]): Basenames of saved plots from generated_plots/.
                - table_md (str | null): Optional Markdown table.
            eda_conclusion (str): Synthesis of EDA patterns and trends.
            advanced_sections (list): Same schema as eda_sections.
                heading should name the method (e.g. "Multiple Regression").
            advanced_conclusion (str): Synthesis of model insights.
            conclusion (str): Comprehensive integrative summary.
            file_references (list[str]): All input datasets, scripts, and supplementary
                files used during the analysis (full or relative paths).

        output_filename: Output .docx filename (no path, no extension).
            If empty, derived from the report title.

    Returns:
        Path to the rendered .docx file, or an error message with details.
    """
    import json as _json
    import re as _re
    import shutil as _shutil

    REPORTS_DIR = PROJECT_DIR / "reports"
    REPORTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR = PROJECT_DIR / "generated_plots"

    # ── Parse spec ────────────────────────────────────────────────────────────
    try:
        if isinstance(report_spec, str):
            spec = _json.loads(report_spec)
        else:
            spec = dict(report_spec)
    except Exception as e:
        return f"[ERROR] Could not parse report_spec JSON: {e}"

    title = spec.get("title", "Data Analysis Report")
    abstract = spec.get("abstract", "")
    introduction = spec.get("introduction", "")
    eda_sections = spec.get("eda_sections") or []
    eda_conclusion = spec.get("eda_conclusion", "")
    advanced_sections = spec.get("advanced_sections") or []
    advanced_conclusion = spec.get("advanced_conclusion", "")
    conclusion = spec.get("conclusion", "")
    file_references = spec.get("file_references") or []
    toc_entries = spec.get("toc") or []

    # ── Determine output filename ─────────────────────────────────────────────
    if not output_filename:
        safe = _re.sub(r"[^a-zA-Z0-9 _-]", "", title)
        output_filename = safe.strip().replace(" ", "_")[:60] or "report"

    qmd_path = REPORTS_DIR / f"{output_filename}.qmd"
    docx_path = REPORTS_DIR / f"{output_filename}.docx"

    # Copy referenced plot files into the reports dir so Quarto can find them
    # (Quarto resolves image paths relative to the .qmd file location)
    plot_copy_map: dict[str, str] = {}
    all_plot_refs = []
    for section in list(eda_sections) + list(advanced_sections):
        all_plot_refs.extend(section.get("plot_files") or [])
    for plot_name in all_plot_refs:
        src = PLOTS_DIR / Path(plot_name).name
        if src.exists():
            dest = REPORTS_DIR / src.name
            if not dest.exists():
                _shutil.copy2(src, dest)
            plot_copy_map[Path(plot_name).name] = src.name

    # ── Build ToC comment block ───────────────────────────────────────────────
    toc_comment = ""
    if toc_entries:
        toc_lines = "\n".join(f"- {entry}" for entry in toc_entries)
        toc_comment = f"<!-- Table of Contents\n{toc_lines}\n-->\n\n"

    # ── Helper: render a list of subsections ─────────────────────────────────
    def _render_sections(sections: list) -> str:
        out = ""
        for sec in sections:
            heading = sec.get("heading", "Subsection")
            body = sec.get("body", "")
            plots = sec.get("plot_files") or []
            table = sec.get("table_md") or ""
            out += f"### {heading}\n\n{body}\n\n"
            if table:
                out += f"{table}\n\n"
            for pf in plots:
                pname = Path(pf).name
                if pname in plot_copy_map:
                    out += f"![{heading}]({pname}){{fig-align=\"center\" width=90%}}\n\n"
        return out

    # ── Assemble .qmd ─────────────────────────────────────────────────────────
    qmd = f"""---
title: "{title}"
date: "{__import__('datetime').date.today().isoformat()}"
format:
  docx:
    toc: true
    toc-depth: 3
    number-sections: true
    highlight-style: github
    fig-cap-location: bottom
execute:
  echo: false
  warning: false
---

{toc_comment}## Abstract

{abstract}

## 1. Introduction

{introduction}

## 2. Exploratory Data Analysis (EDA)

{_render_sections(eda_sections)}
### EDA Conclusion

{eda_conclusion}

## 3. Advanced Analysis

{_render_sections(advanced_sections)}
### Advanced Analysis Conclusion

{advanced_conclusion}

## 4. Conclusion

{conclusion}

## 5. File References

The following datasets, scripts, and supplementary files were used in this analysis:

"""
    if file_references:
        for ref in file_references:
            qmd += f"- `{ref}`\n"
    else:
        qmd += "_No file references recorded._\n"

    # ── Write .qmd ────────────────────────────────────────────────────────────
    qmd_path.write_text(qmd, encoding="utf-8")

    # ── Render via Quarto ─────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["quarto", "render", str(qmd_path), "--to", "docx"],
            cwd=str(REPORTS_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return (
                f"[ERROR] Quarto rendering failed.\n"
                f"stderr: {result.stderr[-2000:]}\n"
                f"stdout: {result.stdout[-1000:]}\n"
                f"The .qmd source is at: {qmd_path}"
            )
    except FileNotFoundError:
        return (
            "[ERROR] Quarto not found. Install from https://quarto.org/docs/get-started/\n"
            f"The .qmd source was saved to: {qmd_path}"
        )
    except subprocess.TimeoutExpired:
        return f"[ERROR] Quarto render timed out (>120 s). .qmd saved at: {qmd_path}"

    if docx_path.exists():
        rel = Path("reports") / docx_path.name
        return (
            f"[OK] Report rendered successfully.\n"
            f"File: {rel}\n"
            f"Size: {docx_path.stat().st_size:,} bytes\n"
            f"The .docx report is ready to download from the Streamlit interface."
        )
    return f"[WARN] Quarto finished but output not found at expected path: {docx_path}"


# Common data science imports for quick reference
COMMON_IMPORTS = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
"""
