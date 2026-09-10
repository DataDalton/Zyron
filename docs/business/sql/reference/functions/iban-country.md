# iban_country

Returns the first two letters, which are the ISO country code. The IBAN is checked against the mod-97 rule first, so an account number with a bad check digit gives NULL rather than its leading letters.

## Syntax

```sql
iban_country(iban)
```

## Returns

VARCHAR of two characters. NULL when the text is NULL or is not a valid IBAN.

## Examples

```sql
SELECT iban_country('GB82WEST12345698765432')
```

GB.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [iban_bban](iban-bban.md)
