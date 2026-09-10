# markdown_extract_headers

Returns one entry per heading, holding the level as a number and the heading text. The entries keep document order, so the result is an outline rather than a set.

## Syntax

```sql
markdown_extract_headers(markdown)
```

## Returns

ARRAY of level and text pairs as JSON text. NULL when the input is NULL.

## Examples

```sql
SELECT markdown_extract_headers('# One' || chr(10) || '## Two')
```

Level 1 with One, then level 2 with Two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [markdown_extract_links](markdown-extract-links.md)
- [markdown_extract_code_blocks](markdown-extract-code-blocks.md)
- [markdown_to_html](markdown-to-html.md)
