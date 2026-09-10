# document_extract_text

Returns the text the document holds, with no layout, styling or image content. A scanned document holding only images yields nothing, because this reads embedded text rather than recognising characters. Use image_ocr for a scan.

## Syntax

```sql
document_extract_text(document)
```

## Returns

TEXT. NULL when the payload is NULL.

## Examples

```sql
SELECT document_extract_text(attachment) FROM zyron_test.filings
```

The text of each document as one value.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [document_to_markdown](document-to-markdown.md)
- [image_ocr](image-ocr.md)
- [document_page_count](document-page-count.md)
