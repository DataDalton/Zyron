# vin_country

Reads the first character of the world manufacturer identifier and maps it to a country or region name. Whitespace is removed and letters are upper-cased first. The mapping is by region block rather than by individual country, so several countries share one name.

## Syntax

```sql
vin_country(vin)
```

## Returns

VARCHAR. NULL when the text is NULL or empty.

## Examples

```sql
SELECT vin_country('1HGBH41JXMN109186')
```

United States.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vin_manufacturer](vin-manufacturer.md)
- [vin_year](vin-year.md)
