# truncate_words

Keeps the first count words, joined by single spaces. Text already holding that many words or fewer is returned unchanged, keeping its original spacing. No suffix is added, unlike truncate_chars.

## Syntax

```sql
truncate_words(text, count)
```

## Returns

VARCHAR. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `count` | Words to keep. | Not applicable. |

## Examples

```sql
SELECT truncate_words('one two three four', 2)
```

one two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [truncate_chars](truncate-chars.md)
- [word_shingle](word-shingle.md)
