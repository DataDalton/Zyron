# CREATE HYBRID INDEX

Indexes a text column and a vector column together, so one query ranks rows by word match and by vector nearness at once. Searching the two separately and combining the results afterwards loses the ranking, because neither list knows what the other scored.

## Syntax

```sql
CREATE HYBRID INDEX name ON table (text_col, vector_col) [WITH (option = value, ...)]
```

## Examples

```sql
CREATE HYBRID INDEX search_idx ON articles (body, embedding)
```

An index ranking one search by both word match and vector nearness.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE FULLTEXT INDEX](create-fulltext-index.md)
- [CREATE VECTOR INDEX](create-vector-index.md)
