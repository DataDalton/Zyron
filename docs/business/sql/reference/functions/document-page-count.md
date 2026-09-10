# document_page_count

Counts the pages the document declares. Formats that have no page concept, such as plain text, have no count to report and raise an error rather than guessing one.

## Syntax

```sql
document_page_count(document)
```

## Returns

BIGINT. NULL when the payload is NULL.

## Examples

```sql
SELECT document_page_count(attachment) FROM zyron_test.filings
```

The page count of each document.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [document_metadata](document-metadata.md)
- [document_extract_text](document-extract-text.md)
