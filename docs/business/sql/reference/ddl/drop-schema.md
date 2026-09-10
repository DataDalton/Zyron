# DROP SCHEMA

Removes a namespace. A schema still holding objects refuses the drop unless CASCADE is written, which removes what is inside it as well.

## Syntax

```sql
DROP SCHEMA [IF EXISTS] name [CASCADE | RESTRICT]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `CASCADE` | Removes the objects the schema holds rather than refusing. | A schema holding objects refuses the drop. |

## Examples

```sql
DROP SCHEMA app CASCADE
```

The namespace and everything in it are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SCHEMA](create-schema.md)
