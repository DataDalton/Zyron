# format_percentage

Multiplies the value by 100 and appends a percent sign. The input is a fraction, so 0.125 gives 12.5%, and passing 12.5 gives 1250%.

## Syntax

```sql
format_percentage(value [, decimals])
```

## Returns

TEXT. NULL when the value is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `decimals` | How many decimal places to keep. | A default number of places is used. |

## Examples

```sql
SELECT format_percentage(0.125, 1)
```

The text 12.5%.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_number](format-number.md)
