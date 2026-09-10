# vin_year

Reads the tenth character and maps it to a model year. The VIN year codes repeat on a 30-year cycle, so the most recent matching year is returned and a vehicle from an earlier cycle reports the later year. The VIN must hold exactly 17 characters after whitespace is removed.

## Syntax

```sql
vin_year(vin)
```

## Returns

INTEGER. NULL when the text is NULL, is not 17 characters, or holds an unused year code.

## Examples

```sql
SELECT vin_year('1HGBH41JXMN109186')
```

2021, the most recent year carrying the code M.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vin_country](vin-country.md)
- [vin_manufacturer](vin-manufacturer.md)
