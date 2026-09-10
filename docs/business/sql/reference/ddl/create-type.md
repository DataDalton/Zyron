# CREATE TYPE

Names a domain over an existing storage type. A constraint or masking rule attached to the domain applies to every column declared with it, rather than being repeated per column.

## Syntax

```sql
CREATE TYPE name AS (storage = type, ...)
```

## Examples

```sql
CREATE TYPE zip AS (storage = TEXT)
```

A named type columns can be declared with.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP TYPE](drop-type.md)
- [CREATE COLLATION](create-collation.md)
