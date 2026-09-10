# text_diff_words

Splits both strings on whitespace and returns the edits that turn the first into the second, grouping consecutive words of the same kind into one operation. Working by word rather than by line suits prose, where a line diff reports a whole changed line. Original spacing is not preserved, because the comparison is over words.

## Syntax

```sql
text_diff_words(old, new)
```

## Returns

ARRAY of operations as JSON text. NULL when either string is NULL.

## Examples

```sql
SELECT text_diff_words('the quick fox', 'the slow fox')
```

An unchanged run, a deletion of quick, an insertion of slow, then an unchanged run.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [text_diff](text-diff.md)
- [text_patch](text-patch.md)
- [row_diff](row-diff.md)
