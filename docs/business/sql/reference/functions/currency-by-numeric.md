# currency_by_numeric

Returns the same details as currency_lookup, found by the three-digit numeric code instead of the alphabetic one. Payment and banking formats carry the numeric code, so data arriving from them is looked up this way.

## Syntax

```sql
currency_by_numeric(numeric)
```

## Returns

A STRUCT of the currency's details. NULL for an unknown code.

## Examples

```sql
SELECT currency_by_numeric(840)
```

The details of the US dollar.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [currency_lookup](currency-lookup.md)
