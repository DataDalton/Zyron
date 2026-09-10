# markdown_extract_links

Returns one entry per link, holding the link text and the target. Inline links are covered. An image, which markdown writes with a leading exclamation mark, is not a link here.

## Syntax

```sql
markdown_extract_links(markdown)
```

## Returns

ARRAY of text and target pairs as JSON text. NULL when the input is NULL.

## Examples

```sql
SELECT markdown_extract_links('see [docs](https://example.com)')
```

One entry holding docs and the URL.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [markdown_extract_headers](markdown-extract-headers.md)
- [extract_urls](extract-urls.md)
