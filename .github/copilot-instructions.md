# GitHub Copilot — Implementation Spec Tools

These instructions bind Copilot Chat to three deterministic commands that operate **from the repository root** and use **absolute paths** for all file I/O:

- `/spec "<feature description>"` – create or update a feature specification  
- `/plan "<implementation details or arguments>"` – execute the plan workflow to generate design artifacts  
- `/tasks "<context or hints>"` – generate an actionable, dependency‑ordered `tasks.md`

> Always detect the repo root via: `ROOT=$(git rev-parse --show-toplevel)` and use absolute paths like `$ROOT/path/to/file`.

---

## /plan — Execute the implementation planning workflow

**Goal:** Run the plan template to generate design artifacts (research, data‑model, contracts, quickstart) under the feature's specs directory.

**Steps (Copilot MUST execute exactly):**
1. **Run:**  
   ```bash
   ROOT=$(git rev-parse --show-toplevel)
   cd "$ROOT" && .specify/scripts/bash/setup-plan.sh --json
   ```

2. Parse JSON for: `FEATURE_SPEC`, `IMPL_PLAN`, `SPECS_DIR`, `BRANCH`.
   Convert each to absolute paths: `FEATURE_SPEC="$ROOT/${FEATURE_SPEC#./}"`, etc.

3. Read & analyze the feature spec at `FEATURE_SPEC` and extract:
   - Feature requirements & user stories
   - Functional & non‑functional requirements
   - Success/acceptance criteria
   - Constraints/dependencies

4. Read the constitution:
   `CONSTITUTION="$ROOT/.specify/memory/constitution.md"` and honor all constitutional requirements.

5. Execute the plan template:
   - Load template: `PLAN_TPL="$ROOT/.specify/templates/plan-template.md"` (already copied to `IMPL_PLAN`)
   - Set Input path to `FEATURE_SPEC`
   - Run Execution Flow (main) steps 1–9 inside the template
   - Follow template error handling and gates verbatim

6. Generate artifacts in `SPECS_DIR`:
   - Phase 0 ⟶ `research.md`
   - Phase 1 ⟶ `data-model.md`, `contracts/`, `quickstart.md`
   - Phase 2 ⟶ `tasks.md` (describe approach only; do not create)

7. Incorporate user arguments into Technical Context as `$ARGUMENTS`

8. Update Progress Tracking in the plan document as phases complete.

9. Verify completion:
   - Progress Tracking shows Phase 0 and Phase 1 complete
   - Required artifacts exist in `SPECS_DIR`
   - No ERROR states remain

10. Report (to chat):
    - branch: `$BRANCH`
    - impl_plan: absolute path
    - specs_dir: absolute path
    - generated: list of absolute artifact paths

**Failure handling:**
- If `setup-plan.sh` fails or JSON is missing keys → STOP and print the stderr and working directory.
- If `plan-template.md` gates fail → STOP and summarize unmet gates.

---

## /spec — Create or update the feature specification

**Goal:** Turn a natural‑language feature description into a structured spec.

**Steps:**

1. **Run:**
   ```bash
   ROOT=$(git rev-parse --show-toplevel)
   cd "$ROOT" && .specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS"
   ```

2. Parse JSON for `BRANCH_NAME` and `SPEC_FILE`. Convert to absolute: `SPEC_FILE="$ROOT/${SPEC_FILE#./}"`.

3. Load template: `$ROOT/.specify/templates/spec-template.md` and follow its required sections exactly.

4. Write the specification to `SPEC_FILE`:
   - Preserve section order and headings
   - Replace placeholders with concrete details from `$ARGUMENTS`
   - Mark unknowns with `[NEEDS CLARIFICATION: …]` rather than guessing
   - Keep WHAT/WHY, avoid implementation HOW

5. Report (to chat):
   - branch: `$BRANCH_NAME`
   - spec_file: absolute path
   - status: "Spec ready for /plan"

**Failure handling:** If script doesn't output JSON or `SPEC_FILE` is not created, STOP with stderr.

---

## /tasks — Generate dependency‑ordered tasks

**Goal:** Produce an executable `tasks.md` based on available design artifacts.

**Steps:**

1. **Run:**
   ```bash
   ROOT=$(git rev-parse --show-toplevel)
   cd "$ROOT" && .specify/scripts/bash/check-task-prerequisites.sh --json
   ```

2. Parse JSON for `FEATURE_DIR` (abs path) and `AVAILABLE_DOCS` list (abs paths).

3. Analyze documents (when present):
   - `plan.md` (always) – tech stack, libraries, structure
   - `data-model.md` – entities → model tasks
   - `contracts/` – each file → contract test + implementation tasks
   - `research.md` – decisions → setup tasks
   - `quickstart.md` – scenarios → integration tests

4. Generate tasks using `$ROOT/.specify/templates/tasks-template.md`:
   - Setup tasks
   - Test‑first tasks `[P]` (one per contract, one per integration scenario)
   - Core tasks: one per entity/service/CLI/endpoint
   - Integration tasks
   - Polish tasks `[P]`
   - Mark `[P]` only when files are independent

5. Ordering rules:
   - Setup → Tests → Models → Services → Endpoints → Integration → Polish
   - Tests before implementation (TDD)
   - Models before services
   - Different files = parallel `[P]`; same file = sequential

6. Write: `FEATURE_DIR/tasks.md`
   - Numbered T001, T002, …
   - Exact file paths
   - Dependency notes
   - Parallel execution examples
   - Feature name pulled from `plan.md`

7. Report (to chat): absolute path to `tasks.md` and task count.

**Failure handling:** If `plan.md` not found in `FEATURE_DIR` → ERROR "No implementation plan found" (do not fabricate).

---

## Global Rules (ALL commands)

- Always resolve repo root: `git rev-parse --show-toplevel`
- Convert all paths to absolute
- Print the current branch: `git rev-parse --abbrev-ref HEAD`
- On any gate/validation failure: STOP and report the specific gate + file + line if known
- Do not mutate templates; only write feature artifacts under `SPECS_DIR`/`FEATURE_DIR`

---

**Policy:** Copilot must refuse to proceed if any step would use a relative path or if a gate check fails. Copilot must echo every shell command it runs and capture JSON for audit in the chat.