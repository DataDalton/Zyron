# CLOSE

Closes a cursor and releases the resources it was holding open. A cursor left open is closed when its transaction ends, or when the session does for one declared WITH HOLD.

## Syntax

```sql
CLOSE name | CLOSE ALL
```

## Examples

```sql
CLOSE c
```

The cursor is closed and its resources released.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [DECLARE](declare.md)
- [FETCH](fetch.md)
