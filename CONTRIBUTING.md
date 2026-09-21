# Contributing

This repository accompanies a manuscript whose numerical results are **frozen**. The
single most important rule for any change is therefore:

> **Do not perturb a frozen result.** If a change would move a reported number, it is a
> manuscript change, not a code change, and it must be agreed separately.

## Development setup

```bash
python -m venv .venv
pip install -r requirements/base.txt
pip install -e .
python -m pytest tests -q
```

## Ground rules

1. **Stage files explicitly.** Use `git add <path>` for the specific files you changed.
   Avoid `git add -A` / `git add .` / `git commit -a` — sweeping stages are how unrelated
   work gets committed together.
2. **Commit in small steps.** One logical change per commit, with a message in
   `type(scope): summary` form (e.g. `fix(sgrpn): preserve calibration CSV float round trips`).
3. **Check before you touch.** Run `git status` first; confirm there are no unstaged
   changes on the files you are about to edit.
4. **Do not rewrite history.** No `git rebase` / `git reset --hard` on shared branches.
5. **Never silently overwrite.** If a file's contents do not match what you expected,
   stop and investigate rather than overwriting it.
6. **Verify numeric claims against the source.** When a change touches numbers, compare
   against the frozen table character by character before writing. It is very easy to
   invert which method is better.
7. **Frozen protocols are immutable.** The YAML files in `configs/` and the corrective
   protocol in `docs/paper/` carry SHA-256 fingerprints. Changing them invalidates the
   provenance chain; add a new protocol instead of editing a frozen one.

## Known pitfalls

These have all caused real, silent bugs in this codebase.

| # | Pitfall | What to do |
|---|---|---|
| 1 | **matplotlib mathtext degrades silently on line breaks** — it renders LaTeX source as literal text with no error and no warning | Collapse whitespace before rendering: `" ".join(latex.split())`. `scripts/render_math_png.py` already contains a validation guard — keep it |
| 2 | `\mathcal L` raises `ParseFatalException`; mathtext only accepts `\mathcal{L}` | The `preprocess_latex` helper inserts the braces automatically; do not strip it |
| 3 | **Passing a string containing backslashes through `python -c` inside a shell mangles it**, producing completely wrong debugging conclusions | For anything containing `\` (LaTeX especially), write a `.py` file and execute that instead |
| 4 | On Windows, Git Bash maps `/tmp` to a drive path that native executables (`python.exe`, `curl.exe`) do not understand, and the write **fails silently** | Always use absolute paths with forward slashes, e.g. `E:/...` |
| 5 | A silently dropped reference (e.g. a mis-resolved markdown anchor) makes output look like it is missing content, when the content exists in the source | Check the source markup before concluding that content is absent |
| 6 | **A git ref write can report success while the ref file never lands** (sandboxed/restricted environments intercepting the lock-and-rename step). Symptom: `git log` reports "does not have any commits yet" and `git status` shows every file as newly added | Confirm with `git rev-parse HEAD` and `git fsck --lost-found` **before touching the working tree**. The objects are safe; only the ref needs rebuilding |
