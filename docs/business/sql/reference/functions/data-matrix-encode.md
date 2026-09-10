# data_matrix_encode

Returns a PNG of the code. Data Matrix packs a short value into a smaller area than a QR code does, which suits marking a small part. Text too long for the code gives an empty value.

## Syntax

```sql
data_matrix_encode(text)
```

## Returns

BYTEA holding a PNG, empty when the text cannot be encoded. NULL when the text is NULL.

## Examples

```sql
SELECT data_matrix_decode(data_matrix_encode('PART-1'))
```

PART-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [data_matrix_decode](data-matrix-decode.md)
- [qr_encode](qr-encode.md)
- [barcode_encode](barcode-encode.md)
