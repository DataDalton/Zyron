# DROP SYNONYM DICTIONARY

Removes the dictionary, so a search finds only the words a row actually holds.

## Syntax

```sql
DROP SYNONYM DICTIONARY [IF EXISTS] name
```

## Examples

```sql
DROP SYNONYM DICTIONARY vehicles
```

Searches match only the words rows actually hold.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SYNONYM DICTIONARY](create-synonym-dictionary.md)
