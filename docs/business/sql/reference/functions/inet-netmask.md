# inet_netmask

Writes the prefix length as an address whose leading bits are set and whose remainder is clear, so a 24-bit prefix gives 255.255.255.0. It carries the same information as inet_prefix in the form older tools expect.

## Syntax

```sql
inet_netmask(inet)
```

## Returns

INET. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_netmask(inet_parse('10.0.0.7/24'))
```

255.255.255.0.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_prefix](inet-prefix.md)
- [inet_network](inet-network.md)
