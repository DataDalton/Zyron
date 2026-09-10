# SHOW

Reads what a setting is currently at for this session, which is the value a statement would run under now rather than the node's configured default. SHOW ALL lists every setting with its value.

## Syntax

```sql
SHOW name | SHOW ALL
```

## Examples

```sql
SHOW search_path
```

The schemas a bare name currently resolves through.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SET](set.md)
