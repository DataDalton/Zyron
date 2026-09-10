# qr_decode

Reads the code in the image and returns its text. An image holding no readable code gives an empty string rather than NULL, so test for an empty result rather than for NULL.

## Syntax

```sql
qr_decode(image)
```

## Returns

VARCHAR, empty when no code is read. NULL when the image is NULL.

## Examples

```sql
SELECT qr_decode(qr_encode('hello'))
```

hello.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [qr_encode](qr-encode.md)
- [barcode_decode](barcode-decode.md)
- [data_matrix_decode](data-matrix-decode.md)
