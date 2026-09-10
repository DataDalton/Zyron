# company_similarity

Lower-cases both names and strips common legal suffixes such as Inc, Corporation, Limited, LLC and Company, then scores what remains with Jaro-Winkler. Two records for one business that differ only in their suffix therefore score 1.0.

## Syntax

```sql
company_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either name is NULL.

## Examples

```sql
SELECT company_similarity('Acme Inc.', 'Acme Corporation')
```

1, because both reduce to acme.

## Refused

- An argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [name_similarity](name-similarity.md)
- [address_similarity](address-similarity.md)
- [jaro_winkler](jaro-winkler.md)
