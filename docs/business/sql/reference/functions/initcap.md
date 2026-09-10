# initcap

Upper-cases the first letter of every word and lower-cases every other letter. A word boundary is any run of non-alphanumeric characters. Letters already upper-cased mid-word are lowered, so an acronym does not survive.

## Syntax

```sql
initcap(text)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT initcap('hello WORLD')
```

Hello World.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [camel_case](camel-case.md)
- [snake_case](snake-case.md)
- [kebab_case](kebab-case.md)
