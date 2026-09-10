# masking_email

Replaces what precedes the at sign with a hash of it and keeps the domain, so two rows from the same sender still match and the sender cannot be read off. The result is not a deliverable address.

## Syntax

```sql
masking_email(email)
```

## Returns

VARCHAR. NULL when the address is NULL or does not parse.

## Examples

```sql
SELECT masking_email('someone@example.com') = masking_email('someone@example.com')
```

true, because the same address masks the same way.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_name](masking-name.md)
- [masking_phone](masking-phone.md)
- [validate_email](validate-email.md)
