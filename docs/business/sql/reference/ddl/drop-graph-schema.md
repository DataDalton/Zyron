# DROP GRAPH SCHEMA

Removes the declaration. The tables it named are untouched, because the schema described how they connect rather than holding anything.

## Syntax

```sql
DROP GRAPH SCHEMA [IF EXISTS] name
```

## Examples

```sql
DROP GRAPH SCHEMA social
```

The declaration is gone and the tables remain.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE GRAPH SCHEMA](create-graph-schema.md)
