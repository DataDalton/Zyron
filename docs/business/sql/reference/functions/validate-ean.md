# validate_ean

Checks the digit count and the check digit for whichever of the three lengths the input has. The number is validated, not looked up, so a well-formed code for no product passes.

## Syntax

```sql
validate_ean(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_ean('4006381333931')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [validate_isbn](validate-isbn.md)
- [barcode_encode](barcode-encode.md)
