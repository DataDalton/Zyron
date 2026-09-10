# EXPORT USER

Collects the data held about one data subject and writes it to a destination. Tables are found through column classifications. EXPORT is a privilege of its own and is not implied by SELECT on the tables involved.

## Syntax

```sql
EXPORT USER 'subject' TO 'destination'
```

## Examples

```sql
EXPORT USER 'u-1' TO 'fs:///tmp/dsar'
```

Everything classified as that subject's is written to the destination.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [FORGET USER](forget-user.md)
- [ALTER COLUMN CLASSIFICATION](alter-column-classification.md)
