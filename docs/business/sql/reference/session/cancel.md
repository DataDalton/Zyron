# CANCEL

Asks the statement another connection is running to stop. It is a request the running statement notices at its next cancellation point rather than an interruption, so a statement stops promptly without being cut off mid-write.

## Syntax

```sql
CANCEL BACKEND pid
```

## Examples

```sql
CANCEL BACKEND 42
```

The statement that connection is running stops.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SHOW](show.md)
