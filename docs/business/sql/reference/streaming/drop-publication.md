# DROP PUBLICATION

Removes a publication. Subscribers reading through it stop receiving changes. The tables it named are unaffected.

## Syntax

```sql
DROP PUBLICATION [IF EXISTS] name
```

## Examples

```sql
DROP PUBLICATION pub
```

Subscribers reading through it stop receiving changes.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE PUBLICATION](create-publication.md)
