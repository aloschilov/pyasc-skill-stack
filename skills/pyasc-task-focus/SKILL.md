---
name: pyasc-task-focus
description: Task focus and attention management skill for pyasc kernel development. Avoids the "lost in the middle" problem in long tasks by maintaining todo.md files. Trigger — task requires more than 5 steps, complex multi-step kernel development, long context conversations, or user explicitly requests a plan.
---

# Task Focus and Attention Management

## Overview

This skill maintains focus during long pyasc kernel development tasks by creating and updating `todo.md` files, keeping the global goal visible at the end of the context.

## Core Principles

> **Problem**: In long tasks (>10 steps), the model tends to lose focus on earlier goals.
>
> **Solution**: Constantly update and re-read the todo list, pushing the global plan to the end of the context.

## When to use

| Scenario | Description |
|----------|-------------|
| **Highly recommended** | Task > 5 steps, multiple sub-goals, long context, kernel development workflow |
| **Not needed** | Simple single-step tasks, quick questions |

## Core workflow

```
Start task -> Create todo.md -> Execute steps -> Update todo.md ->
    |                                                |
    +---- Every 3-5 steps, print todo.md to end of context <----+
```

### Step 1: Create todo.md

Create immediately when task starts:

```markdown
# Task: [name]

## Goal
[1-2 sentences describing the goal]

## To-do items
- [ ] Step 1
- [ ] Step 2
- [ ] ...

## Progress
0/N
```

### Step 2: Keep updating

After each step:
1. Mark completed: `- [ ]` -> `- [x]`
2. Update progress
3. Write todo.md content to end of context

### Step 3: Stay visible

Every 3-5 steps, reprint current todo.md state.

## Templates

### Kernel development template

See [assets/template_kerneldev.md](assets/template_kerneldev.md)

### Debug template

See [assets/template_debug.md](assets/template_debug.md)

### Simple template

See [assets/template_simple.md](assets/template_simple.md)

## References

- [Best Practices](references/best-practices.md)
