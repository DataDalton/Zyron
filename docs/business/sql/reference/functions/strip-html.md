# strip_html

Drops everything between angle brackets and decodes character entities, keeping the text between the tags. It removes markup rather than making untrusted markup safe to render, so use sanitize_html for that. No whitespace is inserted where a block tag was, so two paragraphs can run together.

## Syntax

```sql
strip_html(html)
```

## Returns

VARCHAR. NULL when the input is NULL.

## Examples

```sql
SELECT strip_html('<p>Hello <b>world</b></p>')
```

Hello world.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sanitize_html](sanitize-html.md)
- [html_to_text](html-to-text.md)
- [html_to_markdown](html-to-markdown.md)
