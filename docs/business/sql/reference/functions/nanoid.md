# nanoid

Behaves as gen_nanoid does, including taking the length from the first row of the argument. A length of 0 produces the default of 21 characters rather than an empty string.

## Syntax

```sql
nanoid([length])
```

## Returns

VARCHAR. Never NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `length` | Characters to generate. 0 means the default. | 21. |

## Examples

```sql
SELECT length(nanoid(12))
```

12.

## Refused

- Called with more than one argument.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_nanoid](gen-nanoid.md)
- [cuid2](cuid2.md)
