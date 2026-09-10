# slug

Lower-cases the letters and digits, turns every other ASCII character into a hyphen, collapses runs of hyphens and drops a leading or trailing one. Common accented characters are transliterated to their ASCII form, and a character with no transliteration is dropped, so text in a non-Latin script can reduce to an empty string.

## Syntax

```sql
slug(text)
```

## Returns

VARCHAR. NULL when the text is NULL.

## Examples

```sql
SELECT slug('Hello, World!')
```

hello-world.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [kebab_case](kebab-case.md)
- [title_case](title-case.md)
