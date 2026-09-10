# ALTER SYNONYM DICTIONARY

Adds or removes a group. Searches pick the change up at once, because synonyms are applied when a query is analyzed rather than when the index was built.

## Syntax

```sql
ALTER SYNONYM DICTIONARY name ADD ('word', 'word', ...) | DROP 'word'
```

## Examples

```sql
ALTER SYNONYM DICTIONARY vehicles ADD ('lorry', 'truck')
```

Searches treat those words as the same from now on.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SYNONYM DICTIONARY](create-synonym-dictionary.md)
