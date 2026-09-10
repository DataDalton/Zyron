# wcag_compliant

Compares wcag_contrast_ratio against 4.5 for level AA and 7.0 for level AAA. These are the thresholds for body text, so large text passing at a lower ratio under WCAG is still reported as failing here.

## Syntax

```sql
wcag_compliant(fg, bg, level)
```

## Returns

BOOLEAN. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `level` | AA or AAA, matched without regard to case. | Not applicable. |

## Examples

```sql
SELECT wcag_compliant(color_from_rgb(0, 0, 0), color_from_rgb(255, 255, 255), 'AAA')
```

true.

## Refused

- The level is neither AA nor AAA.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [wcag_contrast_ratio](wcag-contrast-ratio.md)
- [color_darken](color-darken.md)
