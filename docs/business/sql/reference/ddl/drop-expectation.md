# DROP EXPECTATION

Removes an expectation, so writes are no longer checked against it. The violations it recorded stay, because they are history rather than state.

## Syntax

```sql
ALTER TABLE name DROP EXPECTATION name
```

## Examples

```sql
ALTER TABLE orders DROP EXPECTATION pos
```

Writes are no longer checked against it.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ADD EXPECTATION](add-expectation.md)
