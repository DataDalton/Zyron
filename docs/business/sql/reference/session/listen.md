# LISTEN

Subscribes this session to a channel, so a NOTIFY on it reaches this connection. On a cluster a notification reaches every listening session whichever member it was sent from.

## Syntax

```sql
LISTEN channel
```

## Examples

```sql
LISTEN orders_changed
```

This session receives notifications sent on that channel.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [NOTIFY](notify.md)
