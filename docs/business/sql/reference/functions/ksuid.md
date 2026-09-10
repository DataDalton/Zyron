# ksuid

Names the same function as gen_ksuid.

## Syntax

```sql
ksuid()
```

## Returns

VARCHAR of 27 characters. Never NULL.

## Examples

```sql
SELECT length(ksuid())
```

27.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_ksuid](gen-ksuid.md)
