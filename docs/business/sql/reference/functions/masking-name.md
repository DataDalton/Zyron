# masking_name

Takes the first letter of each word. Two different people sharing initials mask identically, so the result groups more coarsely than masking_email does.

## Syntax

```sql
masking_name(name)
```

## Returns

VARCHAR. NULL when the name is NULL.

## Examples

```sql
SELECT masking_name('Ada Lovelace')
```

The two initials.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_email](masking-email.md)
- [name_similarity](name-similarity.md)
