# inet_network

Clears the bits below the prefix length, giving the network the address belongs to. An address with no prefix length is its own network, because the prefix covers every bit.

## Syntax

```sql
inet_network(inet)
```

## Returns

INET. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_network(inet_parse('10.0.0.7/24'))
```

10.0.0.0/24.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_broadcast](inet-broadcast.md)
- [inet_netmask](inet-netmask.md)
- [inet_host](inet-host.md)
