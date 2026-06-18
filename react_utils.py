"""ReAct Loop Utilities - Reason-Act Pattern Implementation

Provides utilities for formatting, parsing, and displaying the ReAct (Reason-Act) loop:
- REASON: Agent analyzes the problem and plans action
- ACT: Agent executes a tool or action
- OBSERVE: Agent receives and processes the result
- LOOP: Repeat until goal achieved

Usage:
    # Format a ReAct response
    response = format_react_step(
        reason="Need to search for recent AI trends",
        action="tavily_search('latest generative AI 2026')",
        observation="Found 5 recent papers on LLMs"
    )
    
    # Parse agent output to extract ReAct components
    components = parse_react_output(agent_response)
    if components:
        st.write(components['reason'])
        st.code(components['action'])
        st.info(components['observation'])
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ReactStep:
    """A single step in the ReAct loop."""
    reason: str
    action: Optional[str] = None
    observation: Optional[str] = None
    thought: Optional[str] = None  # Intermediate thinking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reason": self.reason,
            "action": self.action,
            "observation": self.observation,
            "thought": self.thought,
        }


def format_react_step(
    reason: str,
    action: Optional[str] = None,
    observation: Optional[str] = None,
    thought: Optional[str] = None,
) -> str:
    """Format a ReAct loop step as markdown.
    
    Args:
        reason: The reasoning/thought process
        action: The action to take (tool call or code execution)
        observation: The result of the action
        thought: Additional intermediate thinking
        
    Returns:
        Formatted markdown string
    """
    parts = []
    
    if reason:
        parts.append(f"**[REASON] Reasoning:**\n{reason}")
    
    if thought:
        parts.append(f"**[THINK] Thought:**\n{thought}")
    
    if action:
        parts.append(f"**[ACTION] Action:**\n```\n{action}\n```")
    
    if observation:
        parts.append(f"**[OBSERVE] Observation:**\n{observation}")
    
    return "\n\n".join(parts)


def parse_react_markers(text: str) -> Optional[ReactStep]:
    """Parse text for ReAct markers (REASON:, ACT:, OBSERVE:).
    
    Looks for patterns like:
    - [REASON] ... [/REASON]
    - [ACT] ... [/ACT]
    - [OBSERVE] ... [/OBSERVE]
    
    Args:
        text: The text to parse
        
    Returns:
        ReactStep if markers found, None otherwise
    """
    reason_match = re.search(r'\[REASON\](.*?)\[/REASON\]', text, re.DOTALL)
    action_match = re.search(r'\[ACT\](.*?)\[/ACT\]', text, re.DOTALL)
    observe_match = re.search(r'\[OBSERVE\](.*?)\[/OBSERVE\]', text, re.DOTALL)
    thought_match = re.search(r'\[THOUGHT\](.*?)\[/THOUGHT\]', text, re.DOTALL)
    
    if reason_match or action_match or observe_match:
        return ReactStep(
            reason=reason_match.group(1).strip() if reason_match else "",
            action=action_match.group(1).strip() if action_match else None,
            observation=observe_match.group(1).strip() if observe_match else None,
            thought=thought_match.group(1).strip() if thought_match else None,
        )
    
    return None


def extract_react_steps(text: str) -> list[ReactStep]:
    """Extract all ReAct loop steps from text.
    
    Args:
        text: The text containing ReAct markers
        
    Returns:
        List of ReactStep objects
    """
    steps = []
    
    # Find all [REASON]...[/REASON] blocks
    reason_blocks = re.finditer(r'\[REASON\](.*?)\[/REASON\]', text, re.DOTALL)
    
    for reason_match in reason_blocks:
        reason = reason_match.group(1).strip()
        start_pos = reason_match.end()
        
        # Look for corresponding action and observation after this reason
        action_match = re.search(r'\[ACT\](.*?)\[/ACT\]', text[start_pos:], re.DOTALL)
        observe_match = re.search(r'\[OBSERVE\](.*?)\[/OBSERVE\]', text[start_pos:], re.DOTALL)
        thought_match = re.search(r'\[THOUGHT\](.*?)\[/THOUGHT\]', text[start_pos:], re.DOTALL)
        
        step = ReactStep(
            reason=reason,
            action=action_match.group(1).strip() if action_match else None,
            observation=observe_match.group(1).strip() if observe_match else None,
            thought=thought_match.group(1).strip() if thought_match else None,
        )
        steps.append(step)
    
    return steps


def format_react_display(step: ReactStep) -> str:
    """Format a ReactStep for display in UI.
    
    Args:
        step: The ReactStep to format
        
    Returns:
        HTML/markdown formatted string
    """
    return format_react_step(
        reason=step.reason,
        action=step.action,
        observation=step.observation,
        thought=step.thought,
    )


def should_show_react_details(message_content: str) -> bool:
    """Check if message contains ReAct markers worth displaying.
    
    Args:
        message_content: The message text
        
    Returns:
        True if ReAct markers detected
    """
    react_markers = ['[REASON]', '[ACT]', '[OBSERVE]', '[THOUGHT]']
    return any(marker in message_content for marker in react_markers)


def strip_react_markers(text: str) -> str:
    """Remove ReAct markers from text while keeping content.
    
    Args:
        text: Text with ReAct markers
        
    Returns:
        Text with markers removed but content preserved
    """
    # Remove opening and closing tags but keep content
    text = re.sub(r'\[REASON\](.*?)\[/REASON\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[THOUGHT\](.*?)\[/THOUGHT\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[ACT\](.*?)\[/ACT\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[OBSERVE\](.*?)\[/OBSERVE\]', r'\1', text, flags=re.DOTALL)
    return text


def create_react_system_prompt_section() -> str:
    """Generate the ReAct section for system prompts.
    
    Returns:
        System prompt text encouraging ReAct loop format
    """
    return """
## ReAct Loop - Reason-Act Pattern

For each step in solving a problem, follow this pattern:

1. **REASON** [REASON]Analyze the current state and determine what needs to happen next[/REASON]
2. **THOUGHT** [THOUGHT]Any intermediate thinking or sub-reasoning[/THOUGHT] (optional)
3. **ACTION** [ACT]Describe the action you will take or tool you will call[/ACT]
4. **OBSERVATION** [OBSERVE]Report what happened after the action (result, output, error)[/OBSERVE]
5. **LOOP** If goal not achieved, go back to REASON with new information

Example:
[REASON]The user wants to know recent AI trends. I should search for latest information.[/REASON]
[ACT]Call tavily_search("latest generative AI 2026")[/ACT]
[OBSERVE]Found 5 papers from 2026 on transformer improvements and multimodal models[/OBSERVE]

[REASON]Now I have information. Let me synthesize a comprehensive answer for the user.[/REASON]
[ACT]Compile information into response[/ACT]
[OBSERVE]Response ready with cited sources[/OBSERVE]

Benefits of this format:
- Clear decision making visible to user and in logs
- Easy to debug agent reasoning
- Shows intermediate steps for transparency
- Supports iterative problem solving
"""
