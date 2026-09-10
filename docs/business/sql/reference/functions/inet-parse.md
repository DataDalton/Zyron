# inet_parse

Reads IPv4 or IPv6 text into an INET value. A prefix length after a slash is kept, so 10.0.0.1/24 holds both the host address and the network it sits in. Text that is not an address is an error.

## Syntax

```sql
inet_parse(text)
```

## Returns

INET. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_parse('10.0.0.1/24')
```

An INET holding the host address and a 24-bit prefix.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cidr_parse](cidr-parse.md)
- [inet_network](inet-network.md)
- [inet_prefix](inet-prefix.md)
