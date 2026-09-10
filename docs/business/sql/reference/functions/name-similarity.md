# name_similarity

Lower-cases both names and collapses runs of whitespace, then scores them with Jaro-Winkler. A pair recognised as a name and its common nickname is raised to the midpoint between that score and 1.0, so Robert against Bob scores far above its spelling similarity. The nickname list covers common English given names only.

## Syntax

```sql
name_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either name is NULL.

## Examples

```sql
SELECT name_similarity('Robert Smith', 'Bob Smith')
```

0.853, the midpoint between the spelling similarity of about 0.706 and 1.0.

## Refused

- An argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [company_similarity](company-similarity.md)
- [address_similarity](address-similarity.md)
- [jaro_winkler](jaro-winkler.md)
