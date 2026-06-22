# Agent skills installed for building Maestro

Installed via the skills.sh CLI (`npx skills add <repo> --skill <name> --copy`) into `.claude/skills/` (mirrored in `.agents/skills/`). Curated to **"write better code" only** — no design/launch/doc-gen fluff. Review skill contents before trusting; they run with full agent permissions.

## Code-quality core (obra/superpowers)
| Skill | When to use |
|---|---|
| `test-driven-development` | Before writing any feature/bugfix code — tests first (router, OpenAI-compat, eval). |
| `systematic-debugging` | Any bug / test failure / streaming or proxy issue — investigate before patching. |
| `verification-before-completion` | Hard gate before claiming anything "works" — run the command, show output, then assert. |
| `requesting-code-review` | After a feature pass / before merge — verify work meets requirements. |
| `receiving-code-review` | When acting on review feedback — technical rigor, not blind agreement. |

## Architecture (mattpocock/skills)
| Skill | When to use |
|---|---|
| `improve-codebase-architecture` | Review/shape the router + server + adapter module boundaries. |

## Discovery (vercel-labs/skills)
| Skill | When to use |
|---|---|
| `find-skills` | Discover/install more skills on demand (`npx skills find <query>`). |

## Spec-driven workflow (from OpenFugu's `.claude/skills` — OpenSpec)
| Skill | When to use |
|---|---|
| `openspec-propose` | Turn a change request into proposal.md + design.md + tasks.md in one step. |
| `openspec-apply-change` | Implement a proposed change from its tasks. |
| `openspec-explore` | Explore current specs/changes. |
| `openspec-archive-change` | Archive a completed change. |
| `openspec-sync-specs` | Keep specs in sync. |

> **OpenSpec needs the `openspec` CLI** (`npm i -g openspec` or `npx openspec`). It's a spec-driven dev workflow (propose → design → tasks → apply → archive). Optional — use it to drive the build cleanly, or skip and implement straight from `MAESTRO-PLAN.md`. Included because Walid asked to use OpenFugu's bundled skills.

## Not installed (deliberately, per "nothing else")
`frontend-design`, `web-design-guidelines`, `webapp-testing`, `slack-gif-creator`, `paper-context-resolver`, React/Vercel-deploy skills — relevant only if/when we add a dashboard or landing page (post-v0).
