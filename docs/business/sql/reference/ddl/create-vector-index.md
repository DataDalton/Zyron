# CREATE VECTOR INDEX

Indexes a vector column so a query can ask for the rows nearest a given vector without comparing against every row. The metric says what near means, and it has to match what the vectors were produced for: a cosine index over vectors compared by distance answers confidently and wrongly.

## Syntax

```sql
CREATE VECTOR INDEX name ON table (col) [WITH (metric = 'cosine' | ..., ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH (metric = 'cosine')` | Says what distance the index measures by. | The default metric applies. |

## Examples

```sql
CREATE VECTOR INDEX emb_idx ON articles (embedding) WITH (metric = 'cosine')
```

An index answering nearest-neighbour queries by cosine distance.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE HYBRID INDEX](create-hybrid-index.md)
- [CREATE INDEX](create-index.md)
