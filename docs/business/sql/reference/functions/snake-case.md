# snake_case

Splits on separators and on case changes, lower-cases each word, and joins with underscores. A string already in snake case is unchanged.

## Syntax

```sql
snake_case(text)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT snake_case('userName')
```

user_name.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [camel_case](camel-case.md)
- [kebab_case](kebab-case.md)
