# Heitor 2

Mixed OR/AND tip rulebase. This version keeps the same membership functions as
Heitor 1 but changes the rule structure.

## Universe

- `service` in `[0, 10]`
- `food` in `[0, 10]`
- `tip` in `[0, 30]`

## Linguistic Variables

- `food`: `{rancid, okay, delicious}`
- `service`: `{poor, good, excellent}`
- `tip`: `{cheap, average, generous}`

## Memberships

- food rancid: trapezoid `[0, 0, 1, 3]`
- food okay: triangle `[2, 5, 8]`
- food delicious: trapezoid `[7, 9, 10, 10]`
- service poor: trapezoid `[0, 0, 2, 4]`
- service good: triangle `[2, 5, 8]`
- service excellent: trapezoid `[6, 8, 10, 10]`
- tip cheap: triangle `[0, 5, 10]`
- tip average: triangle `[10, 15, 20]`
- tip generous: triangle `[20, 25, 30]`

## Rulebase

1. if `service is excellent` then `tip is generous`
2. if `(service is poor) or (food is rancid)` then `tip is cheap`
3. if `(service is good) and (food is delicious)` then `tip is generous`
4. if `(service is good) and (food is okay)` then `tip is average`
