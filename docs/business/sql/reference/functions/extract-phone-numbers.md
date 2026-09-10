# extract_phone_numbers

Scans the text and returns the numbers it recognises, in the order they appear. Recognition covers the common separators and an optional country prefix. Numbers are returned as written rather than normalised to one form.

## Syntax

```sql
extract_phone_numbers(text)
```

## Returns

ARRAY of strings as JSON text. NULL when the text is NULL.

## Examples

```sql
SELECT extract_phone_numbers('call 555-010-1234 today')
```

The number as it appears in the text.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [extract_emails](extract-emails.md)
- [extract_urls](extract-urls.md)
- [masking_phone](masking-phone.md)
