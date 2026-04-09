# Heitor 3

Simplified variation with two food labels and AND-based rules over the food and
service propositions.

## Universe

- `service` in `[0, 10]`
- `food` in `[0, 10]`
- `tip` in `[0, 30]`

## Linguistic Variables

- `food`: `{rancid, delicious}`
- `service`: `{poor, good, excellent}`
- `tip`: `{cheap, average, generous}`

## Memberships

- food rancid: trapezoid `[0, 0, 4, 6]`
- food delicious: trapezoid `[4, 8, 10, 10]`
- service poor: trapezoid `[0, 0, 3, 4.5]`
- service good: trapezoid `[2.5, 4, 6, 7.5]`
- service excellent: trapezoid `[5.5, 7, 10, 10]`
- tip cheap: triangle `[0, 10, 20]`
- tip average: triangle `[5, 15, 25]`
- tip generous: triangle `[10, 20, 30]`

## Rulebase

1. if `service is excellent` then `tip is generous`
2. if `(service is poor) and (food is delicious)` then `tip is average`
3. if `(service is good) and (food is delicious)` then `tip is average`
4. if `(service is good) and (food is rancid)` then `tip is cheap`
5. if `(service is poor) and (food is rancid)` then `tip is cheap`
