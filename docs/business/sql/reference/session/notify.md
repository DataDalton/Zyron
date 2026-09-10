# NOTIFY

Sends a notification to every session listening on a channel, with an optional payload. It is delivered when the sending transaction commits, so a notification never announces work that was then rolled back.

## Syntax

```sql
NOTIFY channel [, 'payload']
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `'payload'` | Text carried to the listeners alongside the channel name. | The notification carries the channel name alone. |

## Examples

```sql
NOTIFY orders_changed, 'reload'
```

Every listening session receives the channel and the payload, at commit.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [LISTEN](listen.md)
