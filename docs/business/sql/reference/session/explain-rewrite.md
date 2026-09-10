# EXPLAIN REWRITE

Asks what a pending upgrade would do to one object a user wrote, such as a view definition or a stored statement whose syntax an upgrade changes. It reports the rewrite rather than performing it, so an operator can read what will change before acknowledging it. It is not a query plan, which is why it branches away from the plan options EXPLAIN otherwise reads.

## Syntax

```sql
EXPLAIN REWRITE FOR OBJECT name
```

## Examples

```sql
EXPLAIN REWRITE FOR OBJECT paid
```

What an upgrade would change in that object definition.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [EXPLAIN](explain.md)
- [ACKNOWLEDGE UPGRADE REWRITES](acknowledge-upgrade-rewrites.md)
