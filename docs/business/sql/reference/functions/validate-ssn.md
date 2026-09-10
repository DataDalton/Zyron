# validate_ssn

Requires nine digits after separators are stripped and rejects the area numbers that are never issued, which are 000, 666 and 900 through 999. Form alone is checked, with no lookup against any register.

## Syntax

```sql
validate_ssn(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_ssn('123-45-6789')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_ssn](masking-ssn.md)
- [validate_credit_card](validate-credit-card.md)
