# CREATE SCHEMA

Creates a namespace. Objects are created inside a schema and addressed as schema.name, and a bare name resolves through the session's search path. There is no default user schema: a client reaches its own objects by creating a schema and setting the search path to it, or by qualifying names.

## Syntax

```sql
CREATE SCHEMA [IF NOT EXISTS] name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF NOT EXISTS` | Does nothing when a schema of that name is already there. | A name already taken fails the statement. |

## Examples

```sql
CREATE SCHEMA app
```

A namespace that objects can be created inside.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP SCHEMA](drop-schema.md)
- [SET](../session/set.md)
