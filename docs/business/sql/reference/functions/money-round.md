# money_round

Rounds half away from zero, so 2.5 minor units become 3 rather than 2. Places at or above the currency's own minor digit count leave the value unchanged, and a negative count rounds into the major units, so -2 rounds to the nearest hundred.

## Syntax

```sql
money_round(money [, decimal_places])
```

## Returns

MONEY. NULL when the value is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `decimal_places` | Places to keep. Negative values round into the major units. | The currency's minor digit count. |

## Examples

```sql
SELECT money_format(money_round(money_create(1255, 'USD'), 1))
```

The value rounded to one decimal place.

## Refused

- The value carries a currency code the table does not hold.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_format](money-format.md)
- [money_create](money-create.md)
- [money_minor_digits](money-minor-digits.md)
