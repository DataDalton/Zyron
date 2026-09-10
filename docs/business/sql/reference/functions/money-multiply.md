# money_multiply

Multiplies a money value by a plain number, keeping its currency. Multiplying two money values is not defined and has no function, because the product of two currencies is not a currency. The result is rounded to the currency's minor unit.

## Syntax

```sql
money_multiply(money, factor)
```

## Returns

MONEY in the input's currency. NULL when either argument is NULL.

## Examples

```sql
SELECT money_multiply(money_create(1000, 'USD'), 3)
```

A MONEY value of 30.00 US dollars.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_add](money-add.md)
- [money_create](money-create.md)
