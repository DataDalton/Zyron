# uuid_to_string

Writes the 16 bytes as 32 lower-case hex digits grouped 8-4-4-4-12. A binary column is accepted as well as a UUID column, and a cell that does not hold exactly 16 bytes gives NULL for that row.

## Syntax

```sql
uuid_to_string(uuid)
```

## Returns

VARCHAR of 36 characters. NULL when the value is NULL or is not 16 bytes.

## Examples

```sql
SELECT uuid_to_string(gen_uuid_v7())
```

A 36-character string with four hyphens.

## Refused

- The argument is neither a UUID nor a binary column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v4](gen-uuid-v4.md)
- [gen_uuid_v7](gen-uuid-v7.md)
