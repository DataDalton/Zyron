# text_diff

Compares the two strings line by line and returns a patch text naming the lines added and removed. text_patch applies the result. Use text_diff_words on prose, where a line diff reports a whole changed line.

## Syntax

```sql
text_diff(old, new)
```

## Returns

VARCHAR holding a patch. NULL when either string is NULL.

## Examples

```sql
SELECT text_patch('a' || chr(10) || 'b', text_diff('a' || chr(10) || 'b', 'a' || chr(10) || 'c'))
```

The second string, because the patch turns one into the other.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [text_patch](text-patch.md)
- [text_diff_words](text-diff-words.md)
- [row_diff](row-diff.md)
