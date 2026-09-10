# REFRESH MATERIALIZED VIEW

Runs the view's query again and replaces the stored rows with the result. Until it completes, readers see the previous result rather than a half-written one.

## Syntax

```sql
REFRESH MATERIALIZED VIEW name
```

## Examples

```sql
REFRESH MATERIALIZED VIEW totals
```

The stored rows are replaced by the query's current result.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CREATE MATERIALIZED VIEW](create-materialized-view.md)
