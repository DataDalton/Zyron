# pascal_case

Splits on separators and word boundaries, then joins with no separator, capitalising each word including the first. camel_case differs only in leaving the first word lower-case.

## Syntax

```sql
pascal_case(text)
```

## Returns

VARCHAR. NULL when the text is NULL.

## Examples

```sql
SELECT pascal_case('user name')
```

UserName.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [camel_case](camel-case.md)
- [snake_case](snake-case.md)
- [title_case](title-case.md)
