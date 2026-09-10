# currency_lookup

Returns the name, symbol, numeric code and minor-unit digits for an ISO 4217 alphabetic code. An unknown code returns NULL, which distinguishes a code the registry does not hold from one that holds no symbol.

## Syntax

```sql
currency_lookup(code)
```

## Returns

A STRUCT of the currency's details. NULL for an unknown code.

## Examples

```sql
SELECT currency_lookup('USD')
```

The details of the US dollar.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [currency_by_numeric](currency-by-numeric.md)
- [money_minor_digits](money-minor-digits.md)
