# parse_number

Reads a number written with locale group and decimal separators, reversing format_number. Reading text with the wrong locale misplaces the decimal point rather than failing, because 1.234 is one thousand in one locale and just over one in another.

## Syntax

```sql
parse_number(text [, locale])
```

## Returns

DOUBLE PRECISION. NULL when the text is NULL or is not a number.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `locale` | The locale whose separators the text uses. | The server's default locale is used. |

## Examples

```sql
SELECT parse_number('1,234,567.89')
```

1234567.89.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_number](format-number.md)
