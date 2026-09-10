# masking_ssn

Keeps the final four digits and replaces the rest with a short hash of the whole number, so two rows holding the same number still match without the number being recoverable.

## Syntax

```sql
masking_ssn(ssn)
```

## Returns

VARCHAR. NULL when the number is NULL or does not parse.

## Examples

```sql
SELECT masking_ssn('123-45-6789') = masking_ssn('123-45-6789')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_phone](masking-phone.md)
- [validate_ssn](validate-ssn.md)
