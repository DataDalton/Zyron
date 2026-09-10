# change_log

Takes the recorded entries and returns the changes between the two versions, in order. The entries are passed in rather than read from the table, so the caller decides which history is rendered.

## Syntax

```sql
change_log(table, from_version, to_version, entries_json)
```

## Returns

ARRAY as JSON text. NULL when any argument is NULL or the entries do not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `from_version` | Version the range starts at. | Not applicable. |
| `entries_json` | JSON array of recorded change entries. | Not applicable. |

## Examples

```sql
SELECT change_log('orders', 1, 2, '[]')
```

An empty result, because no entries were given.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [row_diff_ordinal](row-diff-ordinal.md)
- [row_diff](row-diff.md)
