# convert_currency

Reads the rate from the server's rate store and converts. The from currency must name the value's own currency, which makes a mismatch an error rather than a silent reinterpretation. A dated request uses the latest rate at or before that date, falling back to the latest rate on record when none is that old, so an old date does not fail for want of history.

## Syntax

```sql
convert_currency(money, from_currency, to_currency [, rate_date])
```

## Returns

MONEY in the target currency. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `rate_date` | Date whose rate to use. | The latest rate on record. |

## Examples

```sql
SELECT money_currency_code(convert_currency(money_create(100, 'USD'), 'USD', 'EUR'))
```

EUR.

## Refused

- The stated from currency is not the value's own.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_convert](money-convert.md)
- [money_create](money-create.md)
- [currency_lookup](currency-lookup.md)
