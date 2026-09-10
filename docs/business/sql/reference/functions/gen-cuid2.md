# gen_cuid2

Starts with a letter so the value is safe as an HTML element id or a variable name, and continues with hash-derived characters. The value is not time-ordered, so it carries none of the index locality a ULID or a v7 UUID gives.

## Syntax

```sql
gen_cuid2()
```

## Returns

VARCHAR. Never NULL.

## Examples

```sql
SELECT gen_cuid2()
```

An identifier whose first character is a letter.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_nanoid](gen-nanoid.md)
- [gen_ulid](gen-ulid.md)
- [cuid2](cuid2.md)
