# barcode_encode

Returns a PNG of the barcode. The format name is matched without regard to case, and dashes and underscores in it are ignored. Text the format cannot carry, such as letters in a numeric symbology or the wrong digit count, gives NULL for that row.

## Syntax

```sql
barcode_encode(text [, format])
```

## Returns

BYTEA holding a PNG. NULL when the text is NULL or the format cannot carry it.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `format` | One of code128, code39, ean13, ean8 and upca. | code128. |

## Examples

```sql
SELECT barcode_encode('4006381333931', 'ean13')
```

A PNG of the EAN-13 barcode.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [barcode_decode](barcode-decode.md)
- [qr_encode](qr-encode.md)
- [validate_ean](validate-ean.md)
