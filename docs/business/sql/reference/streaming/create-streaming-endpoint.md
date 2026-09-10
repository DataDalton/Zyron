# CREATE STREAMING ENDPOINT

Publishes a publication at an HTTP path. A client subscribes over a WebSocket or as server-sent events and receives changes as they are committed. Unlike CREATE ENDPOINT, which answers one request, this delivers a continuous feed. The backpressure policy determines what happens to a client that cannot keep up.

## Syntax

```sql
CREATE STREAMING ENDPOINT [IF NOT EXISTS] name ON PATH '/path' PROTOCOL WEBSOCKET | SSE BACKED BY PUBLICATION name AUTH NONE | JWT [BACKPRESSURE ...] [MAX CONNECTIONS n] [HEARTBEAT n SECONDS]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `PROTOCOL WEBSOCKET | SSE` | How the feed is carried: a two-way socket, or one-way server-sent events. | Not applicable. |
| `BACKPRESSURE DROP_OLDEST | ...` | What happens to a client that cannot keep up with the feed. | The surface's own policy applies. |
| `MAX CONNECTIONS n` | How many clients may subscribe at once. | The surface's own limit applies. |
| `HEARTBEAT n SECONDS` | Sends a keepalive at that interval, so an idle feed is not mistaken for a dead connection. | No keepalive is sent. |

## Examples

```sql
CREATE STREAMING ENDPOINT live ON PATH '/live' PROTOCOL SSE BACKED BY PUBLICATION pub AUTH NONE
```

A path that streams the publication's changes as server-sent events.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ENDPOINT](create-endpoint.md)
- [CREATE PUBLICATION](create-publication.md)
