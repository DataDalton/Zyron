# row_diff_ordinal

Compares the two value arrays position by position and names the columns that differ, taking the names from the first argument. Matching by position rather than by name suits a changeset, where the column order is fixed by the schema.

## Syntax

```sql
row_diff_ordinal(columns, old_values, new_values)
```

## Returns

ARRAY of change entries as JSON text. NULL when any argument is NULL or does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `columns` | JSON array of column names, in the order the values are given. | Not applicable. |
| `old_values` | JSON array of the values before the change. | Not applicable. |

## Examples

```sql
SELECT row_diff_ordinal('["a","b"]', '[1,2]', '[1,3]')
```

One entry naming b, with 2 before and 3 after.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [row_diff](row-diff.md)
- [json_diff_table](json-diff-table.md)
- [change_log](change-log.md)
