# camel_case

Splits on separators and word boundaries, then joins with no separator, lower-casing the first word and capitalising each word after it.

## Syntax

```sql
camel_case(text)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT camel_case('user name')
```

userName.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [snake_case](snake-case.md)
- [kebab_case](kebab-case.md)
- [initcap](initcap.md)
