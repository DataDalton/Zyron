# sanitize_html

Keeps only the tags p, h1 through h6, strong, em, code, pre, ul, ol, li, a, br, hr and blockquote, and drops every other element including script and style. The allow-list is fixed rather than a parameter. Run this on any markup coming from outside before it reaches a reader.

## Syntax

```sql
sanitize_html(html)
```

## Returns

VARCHAR holding HTML. NULL when the input is NULL.

## Examples

```sql
SELECT sanitize_html('<p>ok</p><script>alert(1)</script>')
```

The paragraph alone, with the script element gone.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [strip_html](strip-html.md)
- [html_to_text](html-to-text.md)
- [markdown_to_html](markdown-to-html.md)
