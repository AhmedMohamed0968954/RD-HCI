# GitHub workflow

Remote repository: https://github.com/AhmedMohamed0968954/RD-HCI

The repository already has a main branch. Use its clone rather than creating an unrelated Git history. The starter lives in `epic_starter/`, preserving the existing root README.

## First upload from VSC

The prepared clone is at:

```bash
cd /data/leuven/394/vsc39484/epic-kitchens/code/RD-HCI
git status
git diff --cached --stat
```

Set your own identity for this repository, replacing both placeholders. Your GitHub noreply email is also suitable. This is commit attribution, not authentication.

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR COMMIT EMAIL"
git commit -m "Add runnable AR and STA preprocessing starter"
git push -u origin starter/ar-sta-foundation
```

Authenticate through your own Git credential setup when GitHub requests it; do not paste passwords or tokens into chat, source files, remote URLs, or shell command history. Write access requires being an owner or collaborator. After pushing, open a pull request from `starter/ar-sta-foundation` to `main` for the team to review. Do not force-push over another member's work.

## What belongs in Git?

Keep source code, tests, dependency lists, documentation and reusable job scripts. Do not add the enclosing VSC project directory: it contains data, authentication files, environments and generated outputs. `.gitignore` excludes these, but always inspect staged filenames before committing.

## Download scripts and cleanup

- The completed download scripts can stay outside the repository, in the existing parent `code/` directory. They are not needed for training and are not included in this starter.
- Keep a copy of the verified downloader for recovery or missing-file downloads. No files have been deleted or moved automatically.
- Download logs and manifests are useful verification records; keep them locally for now.
- `__pycache__` and `.pytest_cache` can be regenerated and do not need version control.
- `check_on_vsc.slurm` is reusable: each run creates fresh outputs. Keep it as a CPU smoke test. Adjust its account and paths for other team members.

## Small next milestone

Run tests and inspect one AR/STA sample before adding a model. The current pipeline completes annotation reading, split creation and frame sampling; it does not yet train or evaluate a learned model.

Reference: [GitHub guidance on existing local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
