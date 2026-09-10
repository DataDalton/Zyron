# inet_host

Drops the prefix length and keeps the address, so a value carrying both reads as the host alone. Use it when comparing addresses that were stored with differing prefixes.

## Syntax

```sql
inet_host(inet)
```

## Returns

INET. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_host(inet_parse('10.0.0.7/24'))
```

10.0.0.7.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_network](inet-network.md)
- [inet_format](inet-format.md)
