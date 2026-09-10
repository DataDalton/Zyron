# soundex

Reduces a string to a letter and three digits standing for the consonant sounds after it. Two strings pronounced alike in English produce the same code, so comparing codes finds spelling variants of a name. It is tuned to English and gives poor results on other languages and on strings that are not names.

## Syntax

```sql
soundex(text)
```

## Returns

TEXT, four characters. NULL when the argument is NULL.

## Examples

```sql
SELECT soundex('Robert')
```

R163, the same code as soundex('Rupert').

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [metaphone](metaphone.md)
- [nysiis](nysiis.md)
