# ReAct Loop (Reason-Act) Implementation Guide

## What is ReAct?

**ReAct** stands for **Reason-Act** - a foundational pattern in AI agent development that makes agent reasoning transparent and debuggable:

1. **REASON** - Agent analyzes the problem and plans next steps
2. **ACT** - Agent executes an action (calls a tool, runs code, etc.)
3. **OBSERVE** - Agent receives and processes the result
4. **LOOP** - Repeat until goal is achieved

## How It Works in Your App

The ReAct loop is implemented using **XML-style markers** in agent responses:

```
[REASON]Analyze what needs to happen[/REASON]
[THOUGHT]Intermediate thinking (optional)[/THOUGHT]
[ACT]Execute the action[/ACT]
[OBSERVE]Report what happened[/OBSERVE]
```

## UI Features

### 1. **Collapsible ReAct Panel**
When an agent response contains ReAct markers, the chat shows:
- 🧠 **Reasoning Process** expander with all thinking steps
- Each step displays in order: Reasoning → Thought → Action → Observation
- Clean final response with markers stripped

### 2. **Visual Step Display**
Each ReAct step shows:
- 🧠 **Reasoning:** Why this step is needed
- 💭 **Thought:** Intermediate thinking
- ⚡ **Action:** Code or tool call to execute
- 👁️ **Observation:** Result/output of the action

Example in UI:
```
Step 1:
🧠 Reasoning: Need to understand the dataset
⚡ Action: Load CSV and check structure
👁️ Observation: 1000 rows, 15 columns, 2% missing values
---
Step 2:
🧠 Reasoning: Check data distributions
⚡ Action: Generate descriptive statistics
👁️ Observation: Age ranges 18-85, mostly normal distribution
```

## Agent Integration

### Main Orchestrator Agent
Uses ReAct to show delegation decisions:

```
[REASON]User wants to analyze data. Should delegate to data_scientist.[/REASON]
[ACT]Ask data_scientist agent to analyze dataset[/ACT]
[OBSERVE]Received comprehensive analysis with insights[/OBSERVE]
```

### Data Scientist Agent
Uses ReAct for analysis workflow:

```
[REASON]Need to load and explore the dataset[/REASON]
[ACT]Execute: df = pd.read_csv(file); df.info()[/ACT]
[OBSERVE]Dataset loaded: 1000 rows × 15 columns, 2% missing[/OBSERVE]

[REASON]Now perform statistical analysis[/REASON]
[ACT]Execute: df.describe(), correlation matrix[/ACT]
[OBSERVE]Mean age 42, median 40, strong correlation between income and education[/OBSERVE]
```

## Benefits

✅ **Transparency** - See agent's reasoning process  
✅ **Debuggability** - Trace where decisions were made  
✅ **Validation** - Verify logic at each step  
✅ **Learning** - Understand agent decision patterns  
✅ **Iteration** - Easier to spot and fix reasoning errors  

## How to Use

### For Agents
When responding to user requests, include ReAct markers:

1. **Break down complex tasks** into reasoning → action → observation cycles
2. **Use markers for clarity** - they help the UI parse your thinking
3. **Show intermediate steps** - especially in analysis workflows
4. **Report observations** - always state what the action produced

### For Users
When seeing agent responses:

1. **Expand the ReAct Loop panel** to see agent reasoning
2. **Check each step** - verify the logic makes sense
3. **Review observations** - ensure actions produced expected results
4. **Ask questions** - if reasoning doesn't match your expectations

## Example Workflow

### User Request
"Analyze this dataset and find patterns"

### Agent Response with ReAct

```
[REASON]User wants pattern discovery in data. I need to:
1. Load and explore the dataset
2. Check distributions and correlations
3. Identify anomalies and trends
[/REASON]

[ACT]
import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes)
print(df.describe())
[/ACT]

[OBSERVE]
Dataset: 5000 rows × 8 columns
- 4 numeric (age, income, score, years)
- 4 categorical (region, category, status, grade)
- No missing values
- Age: 18-75 (mean 45)
- Income: 20k-250k (mean 95k)
[/OBSERVE]

[REASON]Now I'll check for patterns and correlations[/REASON]

[ACT]
correlation_matrix = df[['age', 'income', 'score', 'years']].corr()
print(correlation_matrix)
[/ACT]

[OBSERVE]
Strong patterns found:
- Income vs Education Years: r=0.82 (strong positive)
- Score vs Experience: r=0.71 (strong positive)
- Age vs Income: r=0.45 (moderate positive)
[/OBSERVE]
```

### UI Display
User sees:
- Collapsible "🔄 ReAct Loop - Reasoning Process" section
- Each step showing reasoning, action, observation
- Clean summary below the loop

## Customization

### File: `react_utils.py`
Core ReAct utilities:
- `format_react_step()` - Format individual steps
- `extract_react_steps()` - Parse ReAct markers from text
- `should_show_react_details()` - Detect ReAct content
- `strip_react_markers()` - Clean response text
- `format_react_display()` - UI-ready formatting

### File: `agent.py`
Main orchestrator ReAct instructions in system prompt

### File: `data_scientist_agent/prompts.py`
Data scientist ReAct guidance for analysis workflows

### File: `streamlit_app.py`
UI rendering:
- `render_message_with_react()` - Display messages with ReAct parsing
- Automatic ReAct panel in chat history

## Tips for Best Results

1. **Be Specific in Reasoning** - "Need to check data types" is better than "Analyzing"
2. **Include Action Details** - Show the code/tool being executed
3. **Report Key Observations** - Don't skip important findings
4. **Use Multiple Steps** - Complex tasks get multiple ReAct cycles
5. **Update TODOs** - Especially in data science workflows (✅ DONE, IN_PROGRESS, PENDING)

## Next Steps

- Add ReAct loop to Researcher agent for web search transparency
- Add ReAct loop to Writer agent for drafting workflow
- Create ReAct step templates for each specialized agent
- Build ReAct analytics dashboard to track reasoning patterns
