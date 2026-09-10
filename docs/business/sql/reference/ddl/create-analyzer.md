# CREATE ANALYZER

Says how text becomes words: what splits it, and what is done to each piece afterwards such as lowercasing or stemming. Search and indexing must use the same analyzer, because a query analyzed differently from the index produces words the index never stored.

## Syntax

```sql
CREATE ANALYZER name AS (tokenizer = 'name' [, filters = ...])
```

## Examples

```sql
CREATE ANALYZER simple AS (tokenizer = 'standard')
```

A named rule a full-text index and its queries share.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER ANALYZER](alter-analyzer.md)
- [DROP ANALYZER](drop-analyzer.md)
- [CREATE FULLTEXT INDEX](create-fulltext-index.md)
