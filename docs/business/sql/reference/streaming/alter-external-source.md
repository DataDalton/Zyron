# ALTER EXTERNAL SOURCE

Changes a source's configuration, pauses and resumes reading from it, resets where it reads from, or refreshes the column list it was discovered with. Resetting the position is refused on a source that keeps none, because there would be nothing to reset.

## Syntax

```sql
ALTER EXTERNAL SOURCE name RENAME TO name | REFRESH SCHEMA | RESET POSITION | PAUSE | RESUME | SET OPTIONS (...) | SET CREDENTIALS (...)
```

## Examples

```sql
ALTER EXTERNAL SOURCE src PAUSE
```

Reading from the source stops until it is resumed.

## Refused

- The position is reset on a source that keeps no position.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE EXTERNAL SOURCE](create-external-source.md)
