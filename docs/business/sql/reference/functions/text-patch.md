# text_patch

Reads the patch text_diff produced and returns the text with the changes applied. A patch whose context does not match the text is an error rather than a partial application, so a patch cannot be applied to the wrong input unnoticed.

## Syntax

```sql
text_patch(text, patch)
```

## Returns

VARCHAR. NULL when either argument is NULL or the patch does not apply.

## Examples

```sql
SELECT text_patch('a', text_diff('a', 'b'))
```

b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [text_diff](text-diff.md)
- [json_patch](json-patch.md)
