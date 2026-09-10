# extract_emails

Scans the text and returns the addresses it recognises, in the order they appear. Recognition is by shape rather than by delivery, so an address that looks well formed is returned whether or not the domain exists. Use validate_email to test one address.

## Syntax

```sql
extract_emails(text)
```

## Returns

ARRAY of strings as JSON text. NULL when the text is NULL.

## Examples

```sql
SELECT extract_emails('write to a@b.com or c@d.org')
```

["a@b.com","c@d.org"].

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [extract_urls](extract-urls.md)
- [extract_phone_numbers](extract-phone-numbers.md)
- [validate_email](validate-email.md)
