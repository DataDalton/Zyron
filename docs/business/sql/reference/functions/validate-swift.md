# validate_swift

Requires either 8 characters for a bank's head office or 11 with a branch code, made of four letters for the institution, two for the country, then the location and optional branch. The code's shape is checked, not its registration.

## Syntax

```sql
validate_swift(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_swift('DEUTDEFF')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [validate_iban](validate-iban.md)
- [iban_country](iban-country.md)
