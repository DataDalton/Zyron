# image_ocr

Runs character recognition over the image and returns what it reads. Backed by an external tool, so the call fails naming the tool and the configuration key when none is configured. The language is read from the first row and applies to every row.

## Syntax

```sql
image_ocr(image [, lang])
```

## Returns

TEXT. NULL when the payload is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `lang` | Language code the recogniser should expect. | eng. |

## Examples

```sql
SELECT image_ocr(scan, 'eng') FROM zyron_test.scans
```

The text recognised in each scan.

## Refused

- No recognition tool is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [document_extract_text](document-extract-text.md)
- [image_metadata](image-metadata.md)
