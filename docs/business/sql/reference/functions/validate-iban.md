# validate_iban

Applies the ISO 13616 mod-97 check, which catches a mistyped digit or a transposition. It validates the number's form, not that the account exists.

## Syntax

```sql
validate_iban(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_iban('GB82WEST12345698765432')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [iban_country](iban-country.md)
- [iban_bban](iban-bban.md)
- [validate_swift](validate-swift.md)
