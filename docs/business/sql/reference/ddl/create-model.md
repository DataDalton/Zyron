# CREATE MODEL

Trains a model on rows the query produces and registers it so a query can call it for predictions. No data leaves the database, and the grants on the training data apply to the training. TARGET names the column the model learns to predict. Without it the algorithm learns structure rather than a labelled outcome.

## Syntax

```sql
CREATE MODEL [IF NOT EXISTS] name TYPE algorithm FEATURES (col, ...) [TARGET col] USING (query) [WITH (option = value, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FEATURES (col, ...)` | The columns of the query the model learns from. | Not applicable. |
| `TARGET col` | The column the model learns to predict. | The model learns structure rather than predicting a column, as an unsupervised algorithm does. |
| `USING (query)` | The query producing the rows it trains on. | Not applicable. |
| `WITH (option = value, ...)` | Training options the algorithm reads. | The algorithm's own defaults apply. |

## Examples

```sql
CREATE MODEL churn TYPE logistic_regression FEATURES (visits) TARGET churned USING (SELECT visits, churned FROM customers)
```

A trained model a query can call for predictions.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP MODEL](drop-model.md)
- [CREATE FEATURE GROUP](create-feature-group.md)
