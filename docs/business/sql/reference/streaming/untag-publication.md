# UNTAG PUBLICATION

Removes a tag from a publication, so a policy naming that tag no longer reaches it.

## Syntax

```sql
UNTAG PUBLICATION name 'tag'
```

## Examples

```sql
UNTAG PUBLICATION pub 'analytics'
```

The publication no longer carries that tag.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [TAG PUBLICATION](tag-publication.md)
