# cuid2

Names the same function as gen_cuid2.

## Syntax

```sql
cuid2()
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

- [gen_cuid2](gen-cuid2.md)
