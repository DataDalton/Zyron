# DISABLE FEATURE

Turns a capability off, so the table stops maintaining what it needed. What the capability had already recorded may be discarded, which is why turning one off is not always reversible without a rebuild.

## Syntax

```sql
ALTER TABLE name DISABLE feature
```

## Examples

```sql
ALTER TABLE orders DISABLE soft_delete
```

The table stops maintaining it and what it recorded may be discarded.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ENABLE FEATURE](enable-feature.md)
