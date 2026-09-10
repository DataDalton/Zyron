# vin_manufacturer

Returns the first three characters, which identify the manufacturer. The code itself is returned rather than a manufacturer name, because naming the maker needs a registry this function does not carry.

## Syntax

```sql
vin_manufacturer(vin)
```

## Returns

VARCHAR of three characters. NULL when the text is NULL or shorter than three characters.

## Examples

```sql
SELECT vin_manufacturer('1HGBH41JXMN109186')
```

1HG.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vin_country](vin-country.md)
- [vin_year](vin-year.md)
