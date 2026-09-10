# iban_bban

Returns what follows the two country letters and the two check digits, which is the account number in its own country's format. The IBAN is checked against the mod-97 rule first, so an invalid one gives NULL. The layout of the result varies by country, so it is not safe to split further without knowing the country.

## Syntax

```sql
iban_bban(iban)
```

## Returns

VARCHAR. NULL when the text is NULL or is not a valid IBAN.

## Examples

```sql
SELECT iban_bban('GB82WEST12345698765432')
```

WEST12345698765432.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [iban_country](iban-country.md)
