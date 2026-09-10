# validate_email

Checks the shape, with the split taken at the last at sign so a local part holding one is read correctly, and rejects an address over 254 characters. It checks form rather than delivery, so a well-formed address at a domain that does not exist passes.

## Syntax

```sql
validate_email(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_email('a@b.com')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [extract_emails](extract-emails.md)
- [masking_email](masking-email.md)
- [validate_url](validate-url.md)
