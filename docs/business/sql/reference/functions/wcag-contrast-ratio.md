# wcag_contrast_ratio

Compares the relative luminance of the two colours by the WCAG 2.0 formula, giving 1.0 for two identical colours and 21.0 for black against white. Alpha is not considered, so a partly transparent foreground is measured as though it were opaque. The ratio is symmetric, so swapping the arguments gives the same figure.

## Syntax

```sql
wcag_contrast_ratio(fg, bg)
```

## Returns

DOUBLE PRECISION between 1.0 and 21.0. NULL when either colour is NULL.

## Examples

```sql
SELECT wcag_contrast_ratio(color_from_rgb(0, 0, 0), color_from_rgb(255, 255, 255))
```

21.

## Refused

- A colour argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [wcag_compliant](wcag-compliant.md)
- [color_to_hsl](color-to-hsl.md)
