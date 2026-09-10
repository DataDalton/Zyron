# CHECKPOINT

Writes dirty pages out and records how far the log has been applied. Recovery replays from the last checkpoint, so restart time scales with the log written since it. Checkpoints also run on their own schedule.

## Syntax

```sql
CHECKPOINT
```

## Examples

```sql
CHECKPOINT
```

Dirty pages are written out and the recovery point moves forward.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [VACUUM](vacuum.md)
