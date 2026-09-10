# SET

Changes a setting the session runs under. SESSION keeps the change until the connection ends, LOCAL until the open transaction does. SET ROLE changes which role the session's privileges are decided by, which is how one connection acts as several principals in turn.

## Syntax

```sql
SET [SESSION | LOCAL] name = value | SET ROLE name | SET search_path = schema, ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `LOCAL` | Keeps the change only until the open transaction ends. | The change lasts for the connection. |
| `ROLE name` | Decides privileges by that role from here on. | Not applicable. |

## Examples

```sql
SET search_path = app
```

Bare names resolve in that schema for the rest of the session.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SHOW](show.md)
- [CREATE SCHEMA](../ddl/create-schema.md)
