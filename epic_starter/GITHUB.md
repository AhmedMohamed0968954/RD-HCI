# Git workflow for this integration

Repository: https://github.com/AhmedMohamed0968954/RD-HCI

The data integration builds on the existing main branch in a separate branch,
`data/integrate-ar-sta`. It preserves the original Dataset/exporter entry points.
Review and merge through a pull request instead of replacing main directly.

```bash
cd /data/leuven/394/vsc39484/epic-kitchens/code/RD-HCI
git status
git diff --stat
# After reviewing and committing changes:
git push -u origin data/integrate-ar-sta
```

Authentication is separate from Git author name/email. Enter credentials only
through your own Git credential setup, not in code, chat or remote URLs. Access to
GitHub does not grant access to another team member's VSC files.

Keep Python modules, scripts, tests and documentation in Git. Videos, environments,
credentials, generated splits, previews, clip caches and models stay outside the
repository. The data/ Python source modules are explicitly allowed by .gitignore.
The original committed CSV files were not changed by this integration.

The old standalone /code/epic_starter directory is a historical copy. Continue
working in /code/RD-HCI; do not manually copy between two active code folders.
