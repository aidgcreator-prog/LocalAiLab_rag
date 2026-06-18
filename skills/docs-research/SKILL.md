---
name: docs-research
description: Research technical questions using MCP documentation tools. Use when the user asks about LangChain, LangGraph, DeepAgents APIs, or needs documentation-grounded answers.
---

# Documentation Research Skill

Docs-first technical research for LangChain, LangGraph, and DeepAgents.

## When to Use This Skill

Use this skill when:
- User asks about LangChain/LangGraph/DeepAgents APIs
- User needs documentation-grounded answers
- User asks how to configure or deploy agents
- User asks about middleware, tools, or MCP integration

## Core Behavior

- Prefer MCP documentation tools for factual questions
- Search first, then read the most relevant page, then answer
- Base answers on documented behavior when possible
- If documentation is incomplete or ambiguous, say so explicitly
- Distinguish clearly between documented facts and inference

## Answer Format

When answering a docs question:

1. Start with the direct answer
2. Include a short explanation grounded in the docs
3. Cite the relevant page title or URL when useful
4. If there are multiple valid approaches, compare them briefly
5. If an API or behavior is not documented, say "I couldn't verify that in the docs."

## Tooling Workflow

1. Use the docs MCP search tool to find relevant pages
2. Use the docs MCP page-reading tool on the best match
3. Synthesize the answer from the documentation
4. Avoid guessing when the docs do not support a claim

## Boundaries

- Do not invent undocumented flags, APIs, or configuration
- Do not claim certainty when the docs do not show it
- If the user asks for code, provide a minimal example consistent with the documentation
- If the user asks a non-docs question, you can still help, but note when stepping beyond documentation
