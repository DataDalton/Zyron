# html_to_text

Removes the markup and decodes entities. No whitespace is inserted where a block element ended, so two adjacent paragraphs run together in the result. Convert through html_to_markdown where the block structure has to survive.

## Syntax

```sql
html_to_text(html)
```

## Returns

VARCHAR. NULL when the input is NULL.

## Examples

```sql
SELECT html_to_text('<p>One</p><p>Two</p>')
```

OneTwo, with nothing inserted between the paragraphs.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [strip_html](strip-html.md)
- [html_to_markdown](html-to-markdown.md)
- [sanitize_html](sanitize-html.md)
