# CREATE SYNONYM DICTIONARY

Groups words a search treats as equivalent, so a query for one matches rows holding another. Synonyms are applied when a query is analyzed, not when the index is built, so a change takes effect without reindexing.

## Syntax

```sql
CREATE SYNONYM DICTIONARY name (('word', 'word', ...), ...)
```

## Examples

```sql
CREATE SYNONYM DICTIONARY vehicles (('car', 'automobile'))
```

A search for either word finds rows holding the other.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER SYNONYM DICTIONARY](alter-synonym-dictionary.md)
- [DROP SYNONYM DICTIONARY](drop-synonym-dictionary.md)
- [CREATE ANALYZER](create-analyzer.md)
