# string_to_array

Splits text into an array. An empty delimiter splits nothing and yields one element holding the whole text. A third argument names the text that reads back as a null element rather than as itself.

## Syntax

```sql
string_to_array(text, delimiter [, null_text])
```

## Returns

An array of TEXT.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `null_text` | Reads a part equal to this text as a null element. | Every part is an element holding its own text. |

## Examples

```sql
SELECT string_to_array('a,b', ',')
```

An array holding a and b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_to_string](array-to-string.md)
