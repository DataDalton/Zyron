# barcode_decode

Reads the barcode in the image and returns its text. The symbology is detected and then discarded, so the result names the value alone. An image holding no readable barcode gives an empty string rather than NULL.

## Syntax

```sql
barcode_decode(image)
```

## Returns

VARCHAR, empty when no barcode is read. NULL when the image is NULL.

## Examples

```sql
SELECT barcode_decode(barcode_encode('4006381333931', 'ean13'))
```

4006381333931.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [barcode_encode](barcode-encode.md)
- [qr_decode](qr-decode.md)
- [validate_ean](validate-ean.md)
