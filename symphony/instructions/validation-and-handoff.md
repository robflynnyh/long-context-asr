# Validation And Handoff

Run the most targeted command or test that demonstrates the task is complete.

For documentation-only changes, run `git diff --check` and inspect the diff.

For code changes, run the narrowest relevant script or test. Prefer small smoke
configs before launching full experiments.

If validation cannot run, document the exact blocker and the command that should
be run later.

Commit completed changes on the issue branch.

Push the branch to `origin`.

Open a GitHub pull request against `dev`, unless the issue explicitly provides a
different `Branch/ref` base.

Include the PR URL in the Linear completion comment.

If pushing or PR creation fails, do not move the issue to `Human Review`; post a
blocker comment with the exact failing command and error.

Use the `linear_graphql` tool for Linear updates.

Post one completion comment summarizing files changed, validation, output paths
if any, GitHub PR URL, and residual risk.

Move the issue to `Human Review` only when the requested work is complete and
the GitHub handoff has succeeded.

Do not move the issue to `Human Review` if the requested work is incomplete,
blocked, not pushed, or missing a PR. In that case, post a blocker comment
explaining exactly what is missing or failing.

Before ending a completed issue, verify with `linear_graphql` that the expected
completion comment exists and that the issue state is `Human Review`.
