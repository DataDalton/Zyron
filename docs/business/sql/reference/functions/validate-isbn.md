# validate_isbn

Strips hyphens and spaces, then checks the length and the check digit for whichever form the digit count indicates. An ISBN-10 ending in X is accepted, because X is that form's value for 10.

## Syntax

```sql
validate_isbn(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_isbn('0-306-40615-2')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [isbn_to_13](isbn-to-13.md)
- [isbn_format](isbn-format.md)
- [validate_issn](validate-issn.md)
