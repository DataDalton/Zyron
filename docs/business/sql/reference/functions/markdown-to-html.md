# markdown_to_html

Renders headings, emphasis, lists, links and fenced code as HTML. The output is not sanitised, so raw HTML written inside the markdown passes through and has to go through sanitize_html before being shown to a reader.

## Syntax

```sql
markdown_to_html(markdown)
```

## Returns

VARCHAR holding HTML. NULL when the input is NULL.

## Examples

```sql
SELECT markdown_to_html('# Title')
```

A level one heading element holding Title.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [html_to_markdown](html-to-markdown.md)
- [sanitize_html](sanitize-html.md)
- [markdown_extract_headers](markdown-extract-headers.md)
