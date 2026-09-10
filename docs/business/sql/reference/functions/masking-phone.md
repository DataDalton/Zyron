# masking_phone

Replaces every digit with an asterisk while leaving separators and spacing in place, so the shape of the number stays readable. With the flag set, a leading country code of up to three digits after a plus sign is kept.

## Syntax

```sql
masking_phone(phone [, keep_country_code])
```

## Returns

VARCHAR. NULL when the number is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `keep_country_code` | Whether to leave a leading country code in place. | false. |

## Examples

```sql
SELECT masking_phone('+1 555-010-1234', true)
```

The country code kept and the rest starred out.

## Refused

- Called with more than two arguments.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [masking_ssn](masking-ssn.md)
- [extract_phone_numbers](extract-phone-numbers.md)
