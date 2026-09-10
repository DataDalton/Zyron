# kebab_case

Splits on separators and on case changes, lower-cases each word, and joins with hyphens. Suits identifiers that appear in URLs, where an underscore is harder to read.

## Syntax

```sql
kebab_case(text)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT kebab_case('userName')
```

user-name.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [snake_case](snake-case.md)
- [camel_case](camel-case.md)
