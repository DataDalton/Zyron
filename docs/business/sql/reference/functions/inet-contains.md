# inet_contains

Tests whether the address falls inside the network, comparing only the bits the network's prefix covers. Two values of different families never contain one another, so an IPv4 address is not inside an IPv6 network.

## Syntax

```sql
inet_contains(network, address)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Examples

```sql
SELECT inet_contains(cidr_parse('10.0.0.0/24'), inet_parse('10.0.0.7'))
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cidr_parse](cidr-parse.md)
- [inet_network](inet-network.md)
