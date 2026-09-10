# money_minor_digits

Returns the number of digits the currency's minor unit occupies: 2 for US dollars, 0 for Japanese yen, 3 for Kuwaiti dinars. Rounding a money amount to a fixed two places is wrong for currencies that do not use two.

## Syntax

```sql
money_minor_digits(money)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT money_minor_digits(money_create(1999, 'USD'))
```

2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_format](money-format.md)
- [currency_lookup](currency-lookup.md)
