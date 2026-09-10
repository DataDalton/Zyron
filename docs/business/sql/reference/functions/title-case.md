# title_case

Capitalises the first letter of each word, except that a word in the list of title stop words stays lower-case unless it is the first or last word. Runs of whitespace collapse to single spaces. Use initcap to capitalise every word without exception.

## Syntax

```sql
title_case(text)
```

## Returns

VARCHAR. NULL when the text is NULL.

## Examples

```sql
SELECT title_case('the lord of the rings')
```

The Lord of the Rings.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [initcap](initcap.md)
- [pascal_case](pascal-case.md)
- [slug](slug.md)
