# DeepAgents Documentation Review - Improvement Guide

**Date**: April 17, 2026  
**Source**: https://docs.langchain.com/oss/python/deepagents/ + https://www.langchain.com/deep-agents

## Executive Summary

Your app implements DeepAgents correctly, but is missing **6 key features** that would significantly improve production readiness:

1. ⭐ **LangSmith Integration** - Missing observability (HIGHEST PRIORITY)
2. ⭐ **Built-in `write_todos` Tool** - Should replace custom TODO tracking
3. ⭐ **Auto-summarization Middleware** - Better context management
4. ⭐ **LangGraph Memory Store** - Better than custom state_manager
5. **Prompt Caching** - Cost and latency reduction
6. **Filesystem Permissions** - Security enforcement

---

## 1. LangSmith Integration ⭐⭐⭐ (HIGHEST PRIORITY)

### Current State
❌ **Not implemented** - No tracing, debugging, or observability

### What DeepAgents Docs Say
> Use [LangSmith](https://docs.langchain.com/langsmith/home) to trace requests, debug agent behavior, and evaluate outputs. Set `LANGSMITH_TRACING=true` and your API key to get started.

### Benefits
- ✅ Trace every agent decision and tool call
- ✅ Debug agent reasoning in detail
- ✅ Evaluate agent outputs programmatically
- ✅ See what agent is actually doing
- ✅ Deployment tracking

### Implementation

**Step 1: Install LangSmith**
```bash
pip install langsmith
```

**Step 2: Add to `.env`**
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-api-key>  # Get from https://smith.langchain.com/
LANGSMITH_PROJECT=<project-name>   # e.g., "my-agents"
```

**Step 3: Enable in code**

Update `agent.py`:
```python
import os
from langsmith import Client

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# The rest of your code remains unchanged
# DeepAgents will automatically trace all tool calls
```

**Step 4: Use Dashboard**
- Go to https://smith.langchain.com/
- View every agent call, tool invocation, and decision tree
- Debug failures instantly
- Evaluate performance metrics

### What You'll See
```
Agent Trace:
├─ [REASON] User wants data analysis
├─ [DELEGATE] -> data_scientist agent
│  ├─ [LOAD_FILE] -> file: analysis.csv
│  ├─ [EXECUTE_PYTHON] -> generate statistics
│  └─ [OBSERVE] -> 1000 rows analyzed
└─ [SYNTHESIZE] -> Final response
```

---

## 2. Built-in `write_todos` Tool ⭐⭐⭐

### Current State
✅ Manual TODO tracking in prompts (via your custom prompts)  
❌ Not using DeepAgents' built-in `write_todos` tool

### What DeepAgents Docs Say
> Deep Agents include a built-in [write_todos](https://docs.langchain.com/oss/python/langchain/middleware/built-in#to-do-list) tool that enables agents to break down complex tasks into discrete steps, track progress, and adapt plans as new information emerges.

### Why Switch
- **Better**: Native DeepAgents tool, not custom
- **Integrated**: Works with context management
- **Persistent**: Tracked across agent executions
- **Smart**: Agent adapts plan based on observations

### Current vs. Recommended

**Current (Custom):**
```python
# In prompts.py - manual TODO format
# 📋 ANALYSIS PLAN:
# 1. Load dataset - PENDING
# 2. Check quality - PENDING
```

**Recommended (DeepAgents Built-in):**
```python
# Agent automatically gets access to write_todos tool
agent = create_deep_agent(
    model=model,
    # ... other params ...
    tools=[
        write_todos,  # <- Built-in tool automatically available
        execute_python_code,
        tavily_search,
    ],
)

# Agent can then naturally say:
# "Let me plan this analysis:"
# - Write a TODO for loading the file
# - Write a TODO for validation
# - Write a TODO for visualization
# And track progress automatically!
```

### Implementation
No additional code needed! `write_todos` is available automatically in DeepAgents agents. Update system prompts to encourage usage:

**In `agent.py` system_prompt:**
```python
system_prompt=(
    "You are a combined orchestrator agent...\n\n"
    "## Task Planning\n"
    "Use the write_todos tool to break down complex tasks:\n"
    "1. Call write_todos() with a list of steps\n"
    "2. Track progress with write_todos() as you complete steps\n"
    "3. Adapt the plan based on new information\n\n"
    # ... rest of prompt ...
)
```

**In `data_scientist_agent/prompts.py`:**
```python
DATA_SCIENTIST_INSTRUCTIONS = f"""...\n
<Task Planning with write_todos>
Use the built-in write_todos tool to create your analysis plan:
- write_todos(["Load dataset", "Check data quality", "Create plots", "Generate report"])
- After each step, call write_todos() again to mark items complete
- Adapt the plan if you discover new insights
</Task Planning with write_todos>
"""
```

### Migration Path
- **Keep**: Your ReAct [REASON]/[ACT]/[OBSERVE] markers (good for transparency)
- **Add**: Encourage use of `write_todos` tool for better tracking
- **Result**: Agent uses both ReAct + built-in TODOs

---

## 3. Auto-summarization Middleware ⭐⭐

### Current State
✅ Manual state_manager for persistence  
❌ No automatic context compression

### What DeepAgents Docs Say
> Auto-summarization compacts older conversation messages when the context window grows long, keeping the agent effective across extended sessions.

### Why Important
- **Problem**: Long conversations fill up context window
- **Solution**: DeepAgents automatically compresses old messages
- **Benefit**: Agents stay effective for hours/days of conversation

### Implementation

**Add to `agent.py` after creating agent:**
```python
from deepagents.middleware import AutoSummarizeMiddleware

agent = create_deep_agent(
    model=model,
    # ... other params ...
)

# Wrap agent with auto-summarization
agent_with_summarization = AutoSummarizeMiddleware(
    agent=agent,
    summary_window=10,  # Summarize after 10 messages
    max_context_tokens=8000,  # Keep context under 8k tokens
)

# Use this agent going forward
agent = agent_with_summarization
```

**In `streamlit_app.py`:**
```python
def load_agent():
    """Lazy load agent with auto-summarization."""
    from agent import agent as _agent
    
    # Agent already has auto-summarization if we added it
    return _agent
```

### What Happens
```
Session 1: 5 messages (no compression needed)
Session 2: 15 messages -> [auto-compress older 5] + [keep recent 10]
Session 3: 25 messages -> [auto-compress to summary] + [keep recent 10]
Result: Agent stays effective without losing context
```

---

## 4. LangGraph Memory Store ⭐⭐

### Current State
✅ Custom `state_manager.py` for session state  
❌ Not using LangGraph's built-in Memory Store

### What DeepAgents Docs Say
> Extend agents with persistent memory across threads using LangGraph's [Memory Store](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store). Agents can save and retrieve information from previous conversations.

### Why Switch
| Aspect | Your state_manager | LangGraph Memory Store |
|--------|------------------|----------------------|
| Persistence | Files (manual) | LangGraph (built-in) |
| Cross-session | Per session | Across all sessions |
| Query | Manual load | Automatic retrieval |
| Integration | External | Native to DeepAgents |

### Implementation

**Replace custom state_manager with LangGraph Memory Store:**

```python
from langgraph.store import InMemoryStore

# Create memory store
memory_store = InMemoryStore()

# Use with agent
agent = create_deep_agent(
    model=model,
    # ... other params ...
)

# When invoking agent
result = await agent.ainvoke(
    {"messages": messages},
    config={
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,  # For memory keying
        }
    },
)

# Agent automatically saves to memory_store
# On next invocation with same user_id, memory is retrieved
```

### Benefits
```python
# Agent can naturally save knowledge:
# "I remember that user X prefers matplotlib over plotly"
# "In previous analysis, this user had 1000 data points"

# Next time user X interacts:
# Agent retrieves this memory automatically
# No manual loading needed!
```

---

## 5. Prompt Caching ⭐

### Current State
❌ Not implemented

### What DeepAgents Docs Say
> Use prompt caching to reduce latency and cost.

### How It Works
- Cache system prompts (which don't change)
- Cache large context blocks (like your skills/)
- Reduce tokens sent to model by 50%+
- Reduce latency by caching model attention

### Implementation

**In `.env`:**
```env
LANGCHAIN_ENABLE_CACHE=true
```

**In `agent.py`:**
```python
from langchain_core.caching import InMemoryCache
from langchain_core.globals import set_llm_cache

# Enable caching for all models
set_llm_cache(InMemoryCache())

# Now all LLM calls cache their responses automatically
agent = create_deep_agent(...)
```

### Cost Savings
```
Typical conversation (no cache):
- Message 1: 2000 tokens to model = $0.02
- Message 2: 2000 tokens to model = $0.02
- Message 3: 2000 tokens to model = $0.02
Total: $0.06, 6000 tokens

With prompt caching:
- Message 1: 2000 tokens to model = $0.02
- Message 2: 300 new tokens (1700 cached) = $0.003
- Message 3: 300 new tokens (1700 cached) = $0.003
Total: $0.026, 3600 tokens effective
Savings: ~55% cost reduction!
```

---

## 6. Filesystem Permissions ⭐

### Current State
✅ Using FilesystemBackend  
❌ No permission rules

### What DeepAgents Docs Say
> Declare [permission rules](https://docs.langchain.com/oss/python/deepagents/permissions) that control which files and directories agents can read or write.

### Why Important
- **Security**: Prevent agents from accessing sensitive files
- **Control**: Define what each subagent can access
- **Audit**: Track file operations

### Implementation

**Add to `agent.py`:**

```python
from deepagents.permissions import PermissionRule

agent = create_deep_agent(
    model=model,
    # ... other params ...
    permissions=[
        # Researcher can only read skills/ and memory/
        PermissionRule(
            subagent="researcher",
            allow_read=["/skills/", "/AGENTS.md"],
            deny_write=True,  # Read-only
        ),
        # Data scientist can read/write to data/ directory
        PermissionRule(
            subagent="data_scientist",
            allow_read=["/", "/data/"],
            allow_write=["/data/", "/generated_plots/"],
            deny_write_to=["/AGENTS.md", "/agent.py"],  # Protect core files
        ),
        # Coder can edit code but not system files
        PermissionRule(
            subagent="coder",
            allow_write=["/src/", "/tests/"],
            deny_write_to=["/agent.py", "/.env"],
        ),
    ],
)
```

### Security Benefits
```
Before permissions:
- Any subagent could read .env file (leaked API keys!)
- Any subagent could modify AGENTS.md (corrupted system)
- Data scientist could delete agent.py

After permissions:
- Only specified dirs can be read/written
- Core system files protected
- Clear audit trail of who accessed what
```

---

## Optional Features (Lower Priority)

### Shell Execution with Sandbox
**Current**: Data scientist uses `execute_python_code` (safe)  
**Available**: Use sandbox backend for full shell access (Modal, Daytona, Deno)  
**Recommendation**: Keep current approach (safer for user data)

### Human-in-the-Loop
**Current**: Not implemented  
**Available**: LangGraph interrupts for approval workflows  
**Recommendation**: Add for sensitive operations (file deletion, code deployment)

---

## Implementation Roadmap

### Week 1 (CRITICAL)
1. ✅ **LangSmith Integration** - Add tracing immediately
   - `pip install langsmith`
   - Set env vars
   - Start debugging agent behavior
   - Estimate effort: 30 min

### Week 2 (IMPORTANT)
2. ✅ **Built-in `write_todos`** - Update prompts
   - Encourage tool usage in system prompts
   - No code changes needed
   - Estimate effort: 15 min

3. ✅ **Auto-summarization** - Wrap agent
   - Add AutoSummarizeMiddleware
   - Test with long conversations
   - Estimate effort: 30 min

### Week 3 (RECOMMENDED)
4. ✅ **LangGraph Memory Store** - Replace state_manager
   - Remove custom `state_manager.py`
   - Use LangGraph's built-in Memory Store
   - Simpler, more reliable
   - Estimate effort: 1 hour

5. ✅ **Prompt Caching** - Enable cache
   - Set LANGCHAIN_ENABLE_CACHE=true
   - Measure latency improvements
   - Estimate effort: 15 min

### Week 4 (NICE-TO-HAVE)
6. ✅ **Filesystem Permissions** - Add security layer
   - Define permission rules per subagent
   - Protect sensitive files
   - Estimate effort: 1 hour

---

## What Your App Already Does Well ✅

| Feature | Status | Notes |
|---------|--------|-------|
| Core DeepAgents pattern | ✅ | Excellent foundation |
| Subagent delegation | ✅ | All 7 subagents working |
| Tool integration | ✅ | tavily_search, code execution, etc. |
| Virtual filesystem | ✅ | FilesystemBackend working |
| Skills directory | ✅ | Organized structure |
| ReAct loop | ✅ | Great addition for transparency |
| Agent registry | ✅ | Helpful for UI routing |

---

## Migration Checklist

### Pre-Migration
- [ ] Review each feature in docs
- [ ] Estimate effort for each
- [ ] Prioritize by impact
- [ ] Create feature branch

### During Migration
- [ ] Update `.env` with new keys
- [ ] Add new dependencies
- [ ] Update `agent.py` with features
- [ ] Update system prompts
- [ ] Test each feature in isolation
- [ ] Test full end-to-end flow
- [ ] Monitor LangSmith dashboard

### Post-Migration
- [ ] Remove old code (state_manager, etc.)
- [ ] Update documentation
- [ ] Train team on new tools
- [ ] Set up monitoring in LangSmith
- [ ] Establish baseline metrics

---

## Summary

**Your app is architecturally sound**, but lacks production-grade observability and optimization. The biggest wins are:

1. **LangSmith**: See what agents are doing (debugging/visibility)
2. **write_todos**: Native task tracking (simplification)
3. **Auto-summarization**: Long-session support (scalability)
4. **Memory Store**: Cross-session knowledge (capability)
5. **Caching**: Cost reduction (efficiency)
6. **Permissions**: Security layer (governance)

**Estimated total effort**: 4-5 hours spread over 4 weeks  
**Expected benefit**: Production-ready agent system with full visibility and optimization

---

## References

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/)
- [LangSmith Platform](https://www.langchain.com/langsmith-platform)
- [Context Management Guide](https://blog.langchain.com/context-management-for-deepagents/)
- [LangGraph Memory Store](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store)
- [DeepAgents Comparison](https://docs.langchain.com/oss/python/deepagents/comparison)
