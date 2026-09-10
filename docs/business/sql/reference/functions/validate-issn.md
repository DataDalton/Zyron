# validate_issn

Requires eight characters after hyphens are stripped and checks the final check digit. An ISSN identifies a serial publication, where an ISBN identifies one book.

## Syntax

```sql
validate_issn(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_issn('0378-5955')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [validate_isbn](validate-isbn.md)
- [validate_ean](validate-ean.md)
