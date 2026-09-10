# address_similarity

Lower-cases both addresses and replaces written-out words with their postal abbreviations, covering street types such as Street, Avenue and Boulevard, unit words such as Apartment and Suite, and the compass directions, then scores what remains with Jaro-Winkler. The abbreviations are the United States postal set.

## Syntax

```sql
address_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either address is NULL.

## Examples

```sql
SELECT address_similarity('100 Main Street', '100 Main St')
```

1, because both reduce to 100 main st.

## Refused

- An argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [name_similarity](name-similarity.md)
- [company_similarity](company-similarity.md)
- [jaro_winkler](jaro-winkler.md)
