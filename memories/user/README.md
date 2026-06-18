# User Memory

This directory stores per-user memory files that persist across conversations.

## Files

- **preferences.md** — Read/write. Updated when the agent learns about user preferences.
- **context.md** — Read-only. Contains company/product context.

The agent reads these files at the start of each conversation to personalize output.
