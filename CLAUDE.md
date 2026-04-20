## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - DO NOT KEEP PUSHING
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behaviour between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - DO NOT OVER ENGINEER
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Do not ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/todo.md` after corrections

## Important Instructions

- **Clear Code**: All functions are to be fitted with well documented docstrings(description, args and returns) and type hinting.
- **ENV Confidential**: Do not ever view `.env` or `.env.local` files, `.env.example` and `.env.local.example` are fine though.
- **Copy Pastable**: All terminal commands inside documentation are to be inside a code snippet of its own, with its label stated before the snippet.
- **Flow Diagrams**: All documentation is required to have flow diagrams to provide a visual example of how the flow of the project works, preferably in the form of mermaid code snippets but drawio is fine as well.
- **Professionalism**: All documentation is to be strictly professional while at the same time, easy to understand. Do not use Emojis at all.
- **Auto .venv**: All virtual environments are to be named `.venv` and set up in such a way that they are auto activated upon entering the project root via WSL.
- **Setup Ready**: Setup script Eg. `setup.sh` should be up to date in such a way that when pulled from a fresh system, it is as if it has everything needed for deployment.
- **CICD DataOps**: CI/CD Pipelines are important to ensure that code quality is not compromised, so we need to run checks locally until they all clear before pushing.
- **MLOps MLFlow**: All performances of tested available models are to be accessed via MLFlow.

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what is necessary. Avoid introducing bugs.