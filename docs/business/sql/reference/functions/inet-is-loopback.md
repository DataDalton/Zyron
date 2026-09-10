# inet_is_loopback

Tests whether the address is in 127.0.0.0/8 for IPv4 or is ::1 for IPv6. A loopback address never leaves the host, so traffic recorded against one came from the machine itself.

## Syntax

```sql
inet_is_loopback(inet)
```

## Returns

BOOLEAN. NULL when the argument is NULL.

## Examples

```sql
SELECT inet_is_loopback(inet_parse('127.0.0.1'))
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [inet_is_private](inet-is-private.md)
