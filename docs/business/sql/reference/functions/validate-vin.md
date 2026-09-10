# validate_vin

Requires 17 characters and checks the ninth position against the North American check digit rule. A VIN from a market that leaves that position unused fails even where the rest of the number is correct.

## Syntax

```sql
validate_vin(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_vin('1HGBH41JXMN109186')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vin_year](vin-year.md)
- [vin_country](vin-country.md)
- [vin_manufacturer](vin-manufacturer.md)
