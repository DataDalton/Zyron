# DROP ANALYZER

Removes the analyzer. An index that named it keeps the words it already holds and cannot be rebuilt until another is named.

## Syntax

```sql
DROP ANALYZER [IF EXISTS] name
```

## Examples

```sql
DROP ANALYZER simple
```

Indexes that named it keep their words and cannot be rebuilt under it.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ANALYZER](create-analyzer.md)
