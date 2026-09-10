# inet_is_private

Tests whether the address falls in a range reserved for private networks: 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16 for IPv4, and fc00::/7 for IPv6. A private address is not routable on the public internet, so this distinguishes internal traffic from external without a list of owned ranges.

## Syntax

```sql
inet_is_private(inet)
```

## Returns

BOOLEAN. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_is_private(inet_parse('10.0.0.1'))
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_is_loopback](inet-is-loopback.md)
- [inet_contains](inet-contains.md)
