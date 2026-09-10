# gen_nanoid

Draws characters from a 64-character URL-safe alphabet, so the value needs no escaping in a path or a query string. Length is taken from the first row of the argument and applies to every row, rather than varying per row. The value carries no timestamp.

## Syntax

```sql
gen_nanoid([length])
```

## Returns

VARCHAR. Never NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `length` | Characters to generate. | 21. |

## Examples

```sql
SELECT length(gen_nanoid(10))
```

10.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [nanoid](nanoid.md)
- [gen_cuid2](gen-cuid2.md)
- [gen_ulid](gen-ulid.md)
