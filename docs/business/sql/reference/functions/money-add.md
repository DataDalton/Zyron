# money_add

Adds two money values. Both must be in the same currency: adding different currencies raises an error rather than converting, because no rate is stated and a silent conversion would produce a total nobody can check. Use money_convert to bring one to the other's currency first. Overflow of the minor-unit total is an error.

## Syntax

```sql
money_add(money, money)
```

## Returns

MONEY in the shared currency. NULL when either argument is NULL.

## Examples

```sql
SELECT money_add(money_create(1999, 'USD'), money_create(1, 'USD'))
```

A MONEY value of 20.00 US dollars.

## Refused

- The two values are in different currencies.
- The total exceeds what the minor-unit integer holds.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_subtract](money-subtract.md)
- [money_convert](money-convert.md)
- [money_create](money-create.md)
