# document_to_markdown

Returns the document's text with its structure written as markdown, so headings, lists and tables survive where document_extract_text flattens them. Styling that markdown has no form for is dropped.

## Syntax

```sql
document_to_markdown(document)
```

## Returns

TEXT holding markdown. NULL when the payload is NULL.

## Examples

```sql
SELECT document_to_markdown(attachment) FROM zyron_test.filings
```

Each document as markdown.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [document_extract_text](document-extract-text.md)
- [markdown_to_html](markdown-to-html.md)
