# inet_family

Reports which protocol the address belongs to, as 4 or 6. An IPv4 address mapped into IPv6 space reports 6, matching the form it is stored in.

## Syntax

```sql
inet_family(inet)
```

## Returns

INTEGER, 4 or 6. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_family(inet_parse('10.0.0.1'))
```

4.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_prefix](inet-prefix.md)
