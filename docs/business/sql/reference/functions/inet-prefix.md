# inet_prefix

Reports the prefix length. An address stored without one reports the full width of its family, 32 for IPv4 and 128 for IPv6.

## Syntax

```sql
inet_prefix(inet)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_prefix(inet_parse('10.0.0.1/24'))
```

24.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_netmask](inet-netmask.md)
- [inet_family](inet-family.md)
