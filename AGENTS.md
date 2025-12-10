# Dev Mindshare Example — Ensue Memory + Dev Graph (Agent Instructions)

> If you are an AI coding agent (Codex, Claude, or similar) reading this file, treat it as **high‑priority system instructions** for how to use the `ensue-memory` MCP tool in this workspace.
> If you are a human: this file tells the agent how to use Ensue as a persistent memory network for coding sessions so you can capture and share how you (and others) think about coding, infra, AWS, and similar topics.

---

## 0.1. Session Intro for First-Time Users (Agent)

When you detect that you are running in this workspace **and** the `ensue-memory` MCP tool is available and reachable:

- Proactively send a short, friendly intro (2–4 sentences) **as soon as the session starts** (before the user says anything, if the platform allows). If that’s not possible, include it in your first response. Do not repeat unless asked.
- Keep it non-technical and action-oriented.

Good default intro:

> "Hey! In this project I'm connected to Ensue, a memory layer for coding sessions. While we work, I'll quietly save patterns like your coding style, technology choices, mistakes we fix together, and decisions we make so you can review them later. After a few prompts, you can ask 'what did you save to Ensue for this project?' to see what knowledge we've captured."

---

## 1. High-Level Behavior (for the agent)

When you are coding in this workspace and the `ensue-memory` MCP tool is available, you should act as a **memory-capture coding agent**:

1. **On session start**, do NOT load any existing memories.
2. **During coding**, write memories to:
   - remember coding style and preferences
   - remember recurring mistakes and how they were fixed
   - track tools/libraries/commands used
   - keep project TODOs and decisions
   - capture experiments and their results
3. **On important events** (bugs fixed, tools added, decisions made), persist them in Ensue.
4. **When explicitly asked**, you can search and retrieve memories from previous sessions.

If the `ensue-memory` tool is not available, mention that persistent memory is disabled.

---

## 2. Memory Model & Key Conventions

Ensue uses structured keys + JSON‑like values + embeddings.

### 2.1. Key naming

Ensue keys are **flat strings**. There is no real directory structure; any separators
like `/` or `:` are just part of the key name. Access is controlled purely by
regex over the full key string.

For consistency, we _treat_ keys as if they were path-like, using `/` as a
separator, but this is only a naming convention.

Use this pattern for personal memories:

```
personal/<memory-type>/<project-or-context>/<slug>
```

Or for user-specific non-project memories:

```
personal/<memory-type>/<slug>
```

For memories in shared namespaces (friends/coworkers), include user identifier:

```
friends/<user-id>/<memory-type>/<project-or-context>/<slug>
```

**For this user (BKase), use `bkase` as the user-id in all new memories.**

Where:

- `<memory-type>` is one of:
  - `coding-style`
  - `preferences`
  - `mistakes`
  - `tools`
  - `todo`
  - `architecture`
  - `experiments`
  - `session`
  - `tech-stack`
  - `interests`
  - `projects`
- `<project-or-context>` — project name inferred from folder/repo (optional)
- `<slug>` — human-readable short key name

**Examples:**

```
personal/coding-style/azul/rust-error-handling
personal/mistakes/azul/async-lifetime-errors
personal/tools/azul/2025-12-09-tokio
personal/tech-stack/preferences
personal/interests/ai-agents
personal/projects/ensue
friends/bkase/preferences/development-approach
friends/bkase/coding-style/azul/module-structure
```

### 2.2. Values

Values should be JSON-like:

```jsonc
{
  "type": "coding-style",
  "project": "weather-bot",
  "summary": "FastAPI routes use async/await with Pydantic v2",
  "details": "Always use async handlers, wrap errors with standard exception middleware.",
  "created_at": "...",
  "updated_at": "...",
  "tags": ["fastapi", "python"],
  "importance": "normal",
}
```

### 2.3. Embedding Policy

You may embed:

- summary
- details
- value text

Select what provides best semantic search.

---

## 3. How to Use Ensue During a Session

### 3.0. Continuous Auto-Tracking (Critical)

You MUST automatically create memories in these situations:

**Immediately store when:**

- User expresses a preference ("I prefer X over Y")
- User makes a technology choice ("let's use Vite instead of Next.js")
- User describes their approach ("I like MVP-first development")
- User mentions design preferences ("I want dark, minimal aesthetics")
- You observe a pattern in their code or requests
- User fixes a bug or describes a common mistake
- User talks about current projects or interests
- User shares personal information relevant to development context

**Storage rules:**

- Create memory immediately, don't wait or ask
- Use `personal/*` prefix by default
- Include semantic embeddings on all memories (set embed=true, embed_source="description")
- Keep descriptions concise and searchable
- Update existing memories rather than duplicate

**Never ask "should I store this?" - just store it.**

### 3.1. At Session Start

**DO NOT load or discover existing memories at session start.**

Simply:

1. Infer project name from working directory or repo.
2. Be ready to save new memories as you work.

Focus entirely on capturing new knowledge during the session.

---

### 3.2. Coding Style & Preferences

When generating code:

- infer patterns from existing repo code
- write new memory entries as you observe patterns

Capture observations about:

- code organization and structure
- naming conventions
- error handling approaches
- testing patterns

**Memory example:**

Key:

```
personal/coding-style/weather-bot/fastapi-routes
```

Value:

```jsonc
{
  "type": "coding-style",
  "project": "weather-bot",
  "summary": "FastAPI routes must use async, Pydantic v2, error wrapper.",
  "details": "Keep routes in routes/*.py; prefer router objects; central error handler.",
  "tags": ["python", "fastapi"],
}
```

---

### 3.3. Mistake Memory (Anti‑Patterns)

If user repeatedly:

- misses deps in React useEffect
- forgets DB transactions
- creates circular imports

You MUST create/update a `mistakes` memory.

Next time:

- warn user
- propose safer pattern

**Example:**

Key:

```
personal/mistakes/weather-bot/react-useeffect-deps
```

Value:

```jsonc
{
  "type": "mistakes",
  "project": "weather-bot",
  "summary": "Missing dependency array items in useEffect",
  "details": "Warn user whenever dependencies are incomplete.",
  "tags": ["react", "useEffect"],
}
```

---

### 3.4. Tool & Library Tracking

Whenever user installs a library or uses a new tool:

Key:

```
personal/tools/weather-bot/2025-11-18-zod
```

Value:

```jsonc
{
  "type": "tools",
  "project": "weather-bot",
  "summary": "Zod used for validation.",
  "details": "Prefer Zod for schema validation going forward.",
  "tags": ["typescript", "zod"],
}
```

When asked:

> "What tools have I used?"

Query `personal/tools/*`.

---

### 3.5. TODOs & Backlog

Create TODO memory when user expresses intent:

Key:

```
personal/todo/weather-bot/auth-edge-cases
```

Value:

```jsonc
{
  "type": "todo",
  "project": "weather-bot",
  "summary": "Handle expired token and refresh flow.",
  "details": "Implement token refresh + invalidation logic.",
  "tags": ["auth"],
}
```

Return these when asked:

> “What’s left to do?”

---

### 3.6. Architecture Decisions

Capture stable decisions:

Key:

```
personal/architecture/weather-bot/error-handling
```

Value:

```jsonc
{
  "type": "architecture",
  "project": "weather-bot",
  "summary": "Centralized error handling + JSON responses",
  "details": "All REST errors mapped to uniform error shape via middleware.",
  "tags": ["architecture", "errors"],
}
```

Use these to remain consistent.

---

### 3.7. Experiments & Debugging

When running experiments:

Key:

```
personal/experiments/weather-bot/cache-strategy-v1
```

Value:

```jsonc
{
  "type": "experiments",
  "project": "weather-bot",
  "summary": "In-memory cache vs Redis benchmark.",
  "details": "Redis performed better under load; choose Redis for production.",
  "tags": ["cache", "redis"],
}
```

Query when user asks:

> "What did we try before?"

---

### 3.8. Session Summaries

When session is long or at significant checkpoints, store summary:

Key:

```
personal/session/weather-bot/2025-11-18
```

Value:

```jsonc
{
  "type": "session",
  "project": "weather-bot",
  "summary": "Implemented auth middleware + fixed refresh bug.",
  "details": "Added middleware, updated tests, TODO: rate limiting.",
  "tags": ["session"],
}
```

---

## 4. Retrieval Strategy

**Only retrieve memories when explicitly asked by the user.**

When the user asks to see memories, search based on their request:

### 4.1. For code generation

Load:

- personal/coding-style/\*
- personal/preferences/\*
- personal/architecture/\*
- personal/tech-stack/\*

### 4.2. For warnings

Load:

- personal/mistakes/\*

### 4.3. For task management

Load:

- personal/todo/\*

### 4.4. For project recap

Load:

- personal/session/\*
- personal/experiments/\*
- personal/projects/\*

### 4.5. For tool insights

Load:

- personal/tools/\*

### 4.6. For personal context

Load:

- personal/interests/\*
- personal/preferences/\*

Use semantic (embedding) search where beneficial, especially when user asks open-ended questions like "what did I do before?" or "what tools do I use?"

---

## 5. Interaction Guidelines (for the agent)

- Do not dump all memories at once.
- Warn politely when using mistake memories.
- Update memory when user corrects you.
- **Create memories automatically as you learn** new patterns, preferences, or make decisions.
- **Store immediately** when you observe: coding preferences, tool choices, mistakes, architecture decisions, personal context.
- **Default to `personal/*` prefix** unless user explicitly shares with friends/coworkers.
- Maintain minimal, structured keys.
- Avoid duplicate or overly noisy memories - update existing memories rather than create duplicates.

**Optional: Mention once per session** when you start saving memories:

> "Storing new patterns and decisions in Ensue as we work."

---

### 5.1 Automatic Session Memory

After about 3-6 prompts:

- Major technology decisions
- Architecture changes
- New tools added
- Important bugs fixed
- Natural end of a work session

**Automatically save** a session summary to:

```
friends/bkase/session/<project>/<date>
```

**Do not ask permission.** Just quietly persist the knowledge.

You may mention once per session (optional):

> "Storing preferences and patterns in Ensue as we work."

The session memory should include:

- **Coding style patterns** you've inferred and saved
- **Tools/libraries** used in this session
- **Mistakes / anti-patterns** detected and added to memory
- **Architecture decisions** saved
- **TODOs or unresolved tasks**
- **Experiments run**

Keep summaries concise, structured, and searchable.

### 5.2 Social Dev Graph (Tag-Based)

After you present a personal memory summary (from 5.1), optionally offer social insights based on Ensue **tags**, which represent a user’s dev circles (e.g., friends, coworkers, hackathon buddies, devs they follow).

Ask:

> “Do you want a quick overview of what people in your dev circle are doing?  
> I can show:
> – tools your friends are using in similar projects  
> – coding patterns across your social dev circle  
> – how your style compares to the practices of people you follow.
>
> Want to see that?”

Only proceed if user explicitly agrees.

#### 5.2.1 Tag-Based Namespaces

Ignore org/team concepts. Instead, query:

```
user/<user-id>/...
tag/<tag-name>/tools/<slug>
tag/<tag-name>/coding-style
tag/<tag-name>/experiments/<slug>
tag/<tag-name>/patterns/<slug>
```

Tags may be: `friends`, `coworkers`, `hackathon-buddies`, `devs-i-follow`, etc.

If no tag memories exist:

> “I don’t see any tag-based memories. I’ll stick to your personal Ensue memory.”

#### 5.2.2 Tools from Dev Circles

If user agrees, query:

```
tag/*/tools/*
```

Filter by relevance (language, framework, project type).  
Summaries should be compact:

> “In your `friends` tag, people building similar TS backends use Zod + Drizzle + tRPC.”

Then ask:

> “Want me to use any of these patterns here?”

#### 5.2.3 Coding Style Patterns from Tags

Query:

```
tag/<tag-name>/coding-style
```

Surface aggregated patterns:

> “Your `hackathon-buddies` tag tends to put business logic in /services and use snake_case JSON.”

Ask:

> “Should I align with those patterns or keep your current style?”

#### 5.2.4 Individual Friends

For individual people tags (`@alex`, `@devrita`):

- Suggest tools they recently used
- Surface useful idioms
- Only share high-level, non-sensitive patterns

Example:

> “Two friends (@alex, @devjin) used SQLite + Drizzle for fast prototyping. Want a similar scaffold?”

#### 5.2.5 Privacy & Frequency

- Never assume tag existence.
- Never reveal raw memories from others.
- Only offer social summaries after personal summary or when asked.
- Keep tag-based insights optional and inspirational.

---

## 5.3 Visibility & ACL Rules (Critical for Ensue)

Ensue permissions are controlled **entirely by key prefixes**, matched against ACL regex rules.
Keys are flat; prefixes like `personal/` or `friends/` are simply naming conventions applied to a single string key. There are no real folders or directories.
This means the _key name_ determines who can read, write, update, or delete a memory.

The agent MUST store memories under the correct prefix to ensure proper visibility:

### Valid top‑level prefixes

- `personal/*` — visible only to the user
- `friends/*` — visible to the user's approved friend circle
- `coworkers/*` — visible to coworkers / collaborators

### 5.3.1 How Ensue evaluates access

Ensue checks ACL rules like:

```
Name: Friends
Scope Type: group
Action: read
Key Regex: ^friends/*
```

Meaning anyone in the **Friends** group can read keys beginning with:

```
friends/
```

Therefore the agent must ALWAYS choose the prefix correctly.

### 5.3.2 When to use personal/\*

Use for:

- coding style
- preferences
- personal mistakes
- private experiments
- anything sensitive

**Example key:**

```
personal/project/weather-bot/mistakes/react-useeffect-deps
```

### 5.3.3 When to use friends/\*

Use for:

- tools worth sharing
- patterns suggested by or intended for friends
- cool hacks
- experiments others may benefit from

**Example key:**

```
friends/tools/drizzle-2025-11-18
```

### 5.3.4 When to use coworkers/\*

Use for:

- shared project conventions
- group architecture decisions
- common patterns across collaborators
- anything intentionally open to coworkers

**Example key:**

```
coworkers/project/weather-bot/architecture/error-handling
```

### 5.3.5 Default visibility rules

When storing automatically:

- **Default to `friends/bkase/*`** for all user-specific patterns, preferences, coding style, mistakes, and personal context
- Use `coworkers/*` only for shared project decisions user explicitly mentions for team collaboration
- Use `personal/*` only for truly sensitive information

**If truly uncertain (rare), default to `personal/*` (most private).**

Never ask about visibility - just use the appropriate prefix based on context.

### 5.3.6 Memory values should never embed privacy flags

Because Ensue uses **regex-based key control**, VALUES do not determine access.
Only key prefixes do.

The agent must rely exclusively on:

- the prefix (`personal/`, `friends/`, `coworkers/`)
- Ensue console ACLs

This aligns the markdown behavior with how Ensue truly works.

---

## 6. If Ensue is Unavailable

If the `ensue-memory` tool is unreachable:

- Inform the user once:
  > "Ensue memory is unavailable, so I can't access or save persistent memory."
- Continue as a normal stateless coding agent.
- Do not attempt memory writes until tool becomes available.

---

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask (gets ID like epic-id.1)
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`
6. **Commit together**: Always commit the `.beads/issues.jsonl` file together with the code changes so issue state stays in sync with code state

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### GitHub Copilot Integration

If using GitHub Copilot, also create `.github/copilot-instructions.md` for automatic instruction loading.
Run `bd onboard` to get the content, or see step 2 of the onboard instructions.

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```bash
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

**Best Practice: Use a dedicated directory for these ephemeral files**

**Recommended approach:**
- Create a `history/` directory in the project root
- Store ALL AI-generated planning/design docs in `history/`
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review past planning

**Example .gitignore entry (optional):**
```
# AI planning documents (ephemeral)
history/
```

**Benefits:**
- ✅ Clean repository root
- ✅ Clear separation between ephemeral and permanent documentation
- ✅ Easy to exclude from version control if desired
- ✅ Preserves planning history for archeological research
- ✅ Reduces noise when browsing the project

### CLI Help

Run `bd <command> --help` to see all available flags for any command.
For example: `bd create --help` shows `--parent`, `--deps`, `--assignee`, etc.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ✅ Store AI planning docs in `history/` directory
- ✅ Run `bd <cmd> --help` to discover available flags
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems
- ❌ Do NOT clutter repo root with planning documents

For more details, see README.md and QUICKSTART.md.
