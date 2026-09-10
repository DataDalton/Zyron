# ALTER ANALYZER

Changes an analyzer. Indexes already built hold the words the old rule produced, so a change takes effect for them only once they are rebuilt.

## Syntax

```sql
ALTER ANALYZER name SET (option = value, ...)
```

## Examples

```sql
ALTER ANALYZER simple SET (tokenizer = 'whitespace')
```

New indexing uses the changed rule, and existing indexes keep the old words until rebuilt.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ANALYZER](create-analyzer.md)
- [REINDEX](reindex.md)
