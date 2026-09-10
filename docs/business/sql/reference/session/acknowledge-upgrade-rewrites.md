# ACKNOWLEDGE UPGRADE REWRITES

Accepts the rewrites an upgrade would apply to user-authored objects, such as a view definition the new version parses differently. An upgrade does not rewrite authored text without acknowledgement. Categories are acknowledged separately, so an unambiguous rewrite can be accepted without accepting an ambiguous one.

## Syntax

```sql
ACKNOWLEDGE UPGRADE REWRITES category
```

## Examples

```sql
ACKNOWLEDGE UPGRADE REWRITES AMBIGUOUS
```

The upgrade proceeds with the rewrites in that category.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [EXPLAIN REWRITE](explain-rewrite.md)
- [TRIGGER UPGRADE](trigger-upgrade.md)
- [SHOW UPGRADE](show-upgrade.md)
