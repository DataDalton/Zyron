# html_to_markdown

Converts the elements markdown has a form for and drops the rest, so styling and attributes are lost. A round trip through markdown_to_html does not return the original HTML.

## Syntax

```sql
html_to_markdown(html)
```

## Returns

VARCHAR holding markdown. NULL when the input is NULL.

## Examples

```sql
SELECT html_to_markdown('<h1>Title</h1>')
```

# Title.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [markdown_to_html](markdown-to-html.md)
- [html_to_text](html-to-text.md)
- [strip_html](strip-html.md)
