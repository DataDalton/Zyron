# CREATE FEATURE GROUP

Declares a set of model inputs and how each is computed. The entity key identifies what one row of features describes, and serving looks a row up by it. Training and serving read the same definition, so a feature cannot be computed one way for training and another for serving. The refresh interval sets how current the stored values are kept.

## Syntax

```sql
CREATE FEATURE GROUP [IF NOT EXISTS] name (ENTITY KEY col, FEATURES (name [type] AS (expr), ...) [, SOURCE AS (query)] [, REFRESH EVERY 'interval'] [, WITH (...)])
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ENTITY KEY col` | The column identifying what one row of features is about, which serving looks a row up by. | Not applicable. |
| `FEATURES (name AS (expr), ...)` | Each feature and the expression that computes it. | Not applicable. |
| `SOURCE AS (query)` | The relation the feature expressions are computed over. | The values are written rather than computed. |
| `REFRESH EVERY 'interval'` | How often the stored values are recomputed. | The values are recomputed only when asked for. |

## Examples

```sql
CREATE FEATURE GROUP user_features (ENTITY KEY id, FEATURES (visits INT AS (count_visits)))
```

A declared set of inputs, keyed by entity, that training and serving both read.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP FEATURE GROUP](drop-feature-group.md)
- [CREATE MODEL](create-model.md)
