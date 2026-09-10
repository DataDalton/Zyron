# validate_credit_card

Keeps the digits, ignoring spaces and hyphens, requires between 13 and 19 of them, and checks the Luhn digit. A number passing the check is well formed, which is not the same as issued or active.

## Syntax

```sql
validate_credit_card(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_credit_card('4111 1111 1111 1111')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [validate_iban](validate-iban.md)
- [masking_ssn](masking-ssn.md)
