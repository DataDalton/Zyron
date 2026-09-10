# inet_broadcast

Sets every bit below the prefix length, giving the last address of the network. On IPv4 that address is the broadcast address. IPv6 has no broadcast, so the result is the highest address in the range.

## Syntax

```sql
inet_broadcast(inet)
```

## Returns

INET. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_broadcast(inet_parse('10.0.0.7/24'))
```

10.0.0.255.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_network](inet-network.md)
- [inet_netmask](inet-netmask.md)
