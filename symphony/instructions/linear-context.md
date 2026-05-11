# Linear Context

Before planning or editing, use the `linear_graphql` tool to fetch this issue's
recent comments, newest last.

Use this query shape with the current issue id:

```graphql
query IssueComments($id: String!) {
  issue(id: $id) {
    comments(first: 20) {
      nodes {
        body
        createdAt
        user {
          name
        }
      }
    }
  }
}
```

Treat recent human comments as current task context, especially comments made
after the latest completion, queue, or blocker comment.

If a recent human comment asks a question or requests clarification rather than
implementation, answer it in Linear first and do not move the issue to
`In Review`.

If recent comments request rework on an existing PR or branch, inspect that PR
or branch before editing.

In your plan, explicitly state which recent comments changed or constrained the
task.
