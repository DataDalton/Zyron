# ulid

Names the same function as gen_ulid.

## Syntax

```sql
ulid()
```

## Returns

VARCHAR of 26 characters. Never NULL.

## Examples

```sql
SELECT length(ulid())
```

26.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_ulid](gen-ulid.md)
