# TAG PUBLICATION

Attaches a tag to a publication. A policy or report that names the tag reaches every publication carrying it, including publications tagged later.

## Syntax

```sql
TAG PUBLICATION name WITH 'tag' [, ...]
```

## Examples

```sql
TAG PUBLICATION pub WITH 'analytics'
```

The publication carries that tag.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [UNTAG PUBLICATION](untag-publication.md)
- [CREATE PUBLICATION](create-publication.md)
