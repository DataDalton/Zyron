# RESTORE TABLE

Reads rows an archive holds back into the table they came from. The rows arrive as they were archived, so a restore after later writes adds the old rows beside the new rather than replacing them.

## Syntax

```sql
RESTORE TABLE name FROM 'source'
```

## Examples

```sql
RESTORE TABLE orders FROM 's3://bucket/arch/'
```

The archived rows are read back into the table.

## Refused

- The table is a temporary one.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [ARCHIVE TABLE](archive-table.md)
- [RESTORE TABLE VERSION](restore-table-version.md)
