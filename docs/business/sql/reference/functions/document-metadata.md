# document_metadata

Reads the document's own metadata block and returns what it holds. The fields present depend on the format and on what the producing tool wrote, so a field absent from the result was absent from the document.

## Syntax

```sql
document_metadata(document)
```

## Returns

COMPOSITE as JSON text. NULL when the payload is NULL.

## Examples

```sql
SELECT document_metadata(attachment) FROM zyron_test.filings
```

An object naming whichever metadata fields each document records.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [document_page_count](document-page-count.md)
- [document_extract_text](document-extract-text.md)
- [document_to_markdown](document-to-markdown.md)
