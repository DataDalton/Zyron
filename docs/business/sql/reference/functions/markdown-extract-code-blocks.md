# markdown_extract_code_blocks

Returns one entry per fenced block, holding the language written after the opening fence and the block's contents. A block with no language gives an empty language. Indented blocks without a fence are not returned.

## Syntax

```sql
markdown_extract_code_blocks(markdown)
```

## Returns

ARRAY of language and code pairs as JSON text. NULL when the input is NULL.

## Examples

```sql
SELECT markdown_extract_code_blocks('```sql' || chr(10) || 'SELECT 1' || chr(10) || '```')
```

One entry holding sql and the statement.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [markdown_extract_headers](markdown-extract-headers.md)
- [markdown_to_html](markdown-to-html.md)
