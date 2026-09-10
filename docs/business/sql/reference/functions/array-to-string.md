# array_to_string

Renders each element as text and joins them with a delimiter. A null element is skipped unless a third argument says what a null renders as, which is the SQL form's own behaviour.

## Syntax

```sql
array_to_string(arr, delimiter [, null_text])
```

## Returns

TEXT.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `null_text` | Renders a null element as this text rather than skipping it. | A null element is left out. |

## Examples

```sql
SELECT array_to_string(ARRAY['a', 'b'], ',')
```

The text a,b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [string_to_array](string-to-array.md)
