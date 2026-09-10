# CREATE FULLTEXT INDEX

Indexes text so a search finds rows by the words they contain rather than by the whole value matching. The analyzer decides what counts as a word, which is why the same text indexed under two analyzers answers differently.

## Syntax

```sql
CREATE FULLTEXT INDEX name ON table (col, ...) [WITH (option = value, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH (analyzer = 'name')` | Names the analyzer that decides what a word is. | The default analyzer applies. |

## Examples

```sql
CREATE FULLTEXT INDEX body_ft ON articles (body)
```

An index that finds rows by the words in that column.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ANALYZER](create-analyzer.md)
- [CREATE HYBRID INDEX](create-hybrid-index.md)
- [CREATE INDEX](create-index.md)
