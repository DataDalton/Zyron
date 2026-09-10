# DROP MODEL

Removes a model, so a query calling it fails. The data it was trained on is untouched, and a model of the same name trained again is a different model rather than this one restored.

## Syntax

```sql
DROP MODEL [IF EXISTS] name
```

## Examples

```sql
DROP MODEL churn
```

Queries calling it fail and the training data is untouched.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE MODEL](create-model.md)
