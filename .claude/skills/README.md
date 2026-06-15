# BING project skills

Project-specific Claude Code skills for BING development. Each subfolder contains a `SKILL.md` with YAML frontmatter (`name`, `description`) that Claude Code uses to decide when to load the skill.

## Skill inventory

| Skill | Trigger | Purpose |
|---|---|---|
| [add-anw-model](add-anw-model/SKILL.md) | "add a new a_nw model" | Scaffold a new absorption model class + register + test |
| [add-bbnw-model](add-bbnw-model/SKILL.md) | "add a backscattering model" | Scaffold a new bb_nw model class + register + test |
| [run-bing-fit](run-bing-fit/SKILL.md) | "fit this Rrs spectrum" | Canonical end-to-end MCMC fitting workflow |
| [diagnose-mcmc](diagnose-mcmc/SKILL.md) | "is this fit converged?" | Chain-health diagnostics (autocorr, acceptance, pile-up) |
| [fit-l23-spectrum](fit-l23-spectrum/SKILL.md) | "validate on Loisel 2023" | L23 synthetic-truth fitting and comparison |
| [inelastic-rrs](inelastic-rrs/SKILL.md) | "include Raman / fluorescence" | Add Raman and chlorophyll fluorescence to the forward model |
| [satellite-band-prep](satellite-band-prep/SKILL.md) | "simulate a PACE spectrum" | Wavelength interpolation + per-satellite noise variance |
| [batch-fit-argo](batch-fit-argo/SKILL.md) | "fit all Argo profiles" | Parallel batch MCMC with checkpointing |
| [plot-bing-fit](plot-bing-fit/SKILL.md) | "plot this fit" | Three-panel + decomposition + corner plots |
| [debug-priors](debug-priors/SKILL.md) | "log_prob is -inf" | Triage stuck/empty chains and prior misconfigurations |
| [add-paper-analysis](add-paper-analysis/SKILL.md) | "start a new analysis" | Layout for a new `papers/<topic>/` directory |

## How skills are loaded

Claude Code auto-discovers skills placed in `.claude/skills/<slug>/SKILL.md`. The `description` field in each YAML frontmatter is what Claude matches against the user's prompt to decide whether to load the skill body. Keep descriptions specific and trigger-focused.

The Anthropic spec is at <https://github.com/anthropics/skills/tree/main/spec>.

## Conventions used here

- Each skill is a single `SKILL.md` under ~30 KB so it fits cleanly into context when loaded.
- Skills link to one another with relative paths so Claude can follow them.
- Skills link to source files via `../../../bing/...` paths from inside their folder.
- Code examples are minimal but **runnable** — they reflect the actual public API, including the recent `rt_dict` and `bing.rt.*` refactors.

## Adding a new skill

1. Create `.claude/skills/<slug>/SKILL.md` with YAML frontmatter:
   ```markdown
   ---
   name: my-skill
   description: One specific sentence that names the trigger phrase and the outcome.
   ---

   # Title

   Body...
   ```
2. Add a row to the inventory table above.
3. Cross-link from related skills.
4. Keep it under ~30 KB; offload long reference material to `references/` subfolder if needed.

## See also

- [prompts/claude.md](../../prompts/claude.md) — discussion of which skills BING needs and community examples
- [CLAUDE.md](../../CLAUDE.md) — top-level BING development guide
- [anthropics/skills](https://github.com/anthropics/skills) — official skill examples
- [ianhi/scientific-python-skills](https://github.com/ianhi/scientific-python-skills) — `xarray.md`, `zarr.md` (useful for PACE/Argo/L23 data handling)
