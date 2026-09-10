# CREATE GRAPH SCHEMA

Declares which tables are nodes and which are edges, so a traversal can be written rather than a chain of joins. The rows stay where they were: the schema is a statement about how they connect, not a second copy of them.

## Syntax

```sql
CREATE GRAPH SCHEMA name (NODE Label (col type, ...) [, EDGE Label FROM Label TO Label (...)])
```

## Examples

```sql
CREATE GRAPH SCHEMA social (NODE Person (id INT))
```

A graph a traversal can be written against.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP GRAPH SCHEMA](drop-graph-schema.md)
