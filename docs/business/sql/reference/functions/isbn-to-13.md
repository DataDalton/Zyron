# isbn_to_13

Prepends the 978 prefix and recalculates the check digit. Hyphens and spaces are ignored in the input. The input must be a valid ISBN-10, so an ISBN already in 13-digit form gives NULL rather than passing through.

## Syntax

```sql
isbn_to_13(isbn)
```

## Returns

VARCHAR of 13 digits. NULL when the text is NULL or is not a valid ISBN-10.

## Examples

```sql
SELECT isbn_to_13('0-306-40615-2')
```

9780306406157.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [isbn_format](isbn-format.md)
