---
source: SciPy v1.17.0 Official Docs
library: SciPy
package: scipy
topic: interp1d replacements, extrapolation, and edge cases
fetched: 2026-06-13T00:00:00Z
official_docs: https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html
---

# interp1d: Replacements, Extrapolation & Edge Cases

## Deprecation / Legacy Status

**`interp1d` is legacy, NOT deprecated.** The SciPy developers have stated clearly:

> "interp1d is considered legacy API and is not recommended for use in new code. Consider using more specific interpolators instead."
>
> "We have no plans to remove it; we are going to keep supporting its existing usages; however we believe there are better alternatives which we recommend using in new code."

It will **not** be removed — it simply won't receive new features/updates.

## Recommended Replacements by `kind`

### `kind="linear"` → Use `numpy.interp` or `make_interp_spline(x, y, k=1)`

```python
import numpy as np
from scipy.interpolate import make_interp_spline

x = np.linspace(0, 10, num=11)
y = np.cos(-x**2 / 9.0)

# Option 1: numpy.interp (simple, efficient)
xnew = np.linspace(0, 10, num=1001)
ynew = np.interp(xnew, x, y)

# Option 2: linear spline (supports extrapolation)
spl = make_interp_spline(x, y, k=1)
ynew = spl(xnew)
```

**Note:** `numpy.interp` is limited — it does **not** allow controlling extrapolation behavior. For extrapolation control, use `make_interp_spline(x, y, k=1)` instead.

### `kind="quadratic"` or `kind="cubic"` → Use `make_interp_spline` directly

Under the hood, `interp1d` already delegates to `make_interp_spline`, so use it directly:

```python
from scipy.interpolate import make_interp_spline

x = np.linspace(0, 10, num=11)
y = np.cos(-x**2 / 9.0)

# Cubic spline (k=3 is the default)
spl = make_interp_spline(x, y, k=3)
ynew = spl(np.linspace(0, 10, num=1001))

# Quadratic spline
spl2 = make_interp_spline(x, y, k=2)
```

### `kind="nearest", "previous", "next"` → Use `numpy.searchsorted`

These are all based on `numpy.searchsorted`:

```python
import numpy as np

x = np.arange(8)
y = x**2
x_new = np.linspace(0, 7, 101)

# "nearest" equivalent:
x_bds = x[:-1] / 2.0 + x[1:] / 2.0  # halfway points
idx = np.searchsorted(x_bds, x_new, side='left')
idx = np.clip(idx, 0, len(x) - 1)
y_nearest = y[idx]

# Or use make_interp_spline with k=0:
spl = make_interp_spline(x, y, k=0)  # equivalent to kind='previous'
```

### `kind="cubic"` → Also consider `CubicSpline`

```python
from scipy.interpolate import CubicSpline

x = np.linspace(0, 10, num=11)
y = np.cos(-x**2 / 9.0)
spl = CubicSpline(x, y)

# Evaluate
ynew = spl(np.linspace(0, 10, num=1001))

# Compute derivatives
y_first_deriv = spl(xnew, nu=1)
y_second_deriv = spl(xnew, nu=2)
```

## Handling Extrapolation

### With `interp1d` (legacy)

| `fill_value` | `bounds_error` | Behavior |
|---|---|---|
| `nan` (default) | `None` (default) | Raises `ValueError` for out-of-bounds |
| scalar (e.g., `0.0`) | `False` | Returns that scalar for out-of-bounds |
| `("below", "above")` tuple | `False` | Different values for left/right extrapolation |
| `"extrapolate"` | (ignored) | Linearly extrapolates beyond data range |

```python
from scipy.interpolate import interp1d

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Extrapolate with NaN fill
f_nan = interp1d(x, y, bounds_error=False, fill_value=np.nan)

# Extrapolate with specific fill values
f_fill = interp1d(x, y, bounds_error=False, fill_value=0.0)

# Different fills for below/above
f_below_above = interp1d(x, y, bounds_error=False, fill_value=(-1, -999))

# Enable extrapolation beyond bounds
f_extrap = interp1d(x, y, fill_value="extrapolate")
```

### With `make_interp_spline` (recommended for extrapolation)

```python
from scipy.interpolate import make_interp_spline
import numpy as np

x = np.linspace(0, 5, 11)
y = 2 * x

# Linear spline with linear extrapolation (default behavior)
spl = make_interp_spline(x, y, k=1)
result = spl([-1, 6])  # Returns [-2., 12.] — extrapolates linearly
```

`make_interp_spline` **extrapolates by default** using the first/last polynomial pieces. This is a key advantage over `numpy.interp` which does not extrapolate:

```python
np.interp([-1, 6], x, y)  # Returns [0., 10.] — clamps to bounds, no extrapolation
```

## Edge Cases & Gotchas

### 1. NaN values
**Calling `interp1d` with NaNs in input values results in undefined behavior.** Interpolation with missing data is not directly supported. You must clean/filter NaNs first.

### 2. Non-unique x values
If `x` has duplicate values, behavior is undefined and depends on the chosen `kind`.

### 3. Memory & copies
By default, `interp1d` copies `x` and `y`. Set `copy=False` to use references (saves memory for large arrays).

### 4. Axis handling
Unlike other SciPy interpolators, `interp1d` defaults to `axis=-1` (last axis), not `axis=0`. For multi-dimensional `y` arrays, you may need to explicitly set `axis=0`.

```python
# For 2D y arrays, interp1d interpolates along the LAST axis by default
y_2d = np.column_stack([y1, y2])  # shape (npoints, 2)
f = interp1d(x, y_2d)  # interpolates along axis=-1

# For axis=0 behavior (like other interpolators):
f = interp1d(x, y_2d, axis=0)
```

### 5. Input type requirements
`x` and `y` must be convertible to float (int or float). Other types will fail.

### 6. `assume_sorted=False` sorting
If `assume_sorted=False` (default), the input `x` is sorted, and `y` is reordered accordingly. For large datasets, set `assume_sorted=True` if you already have sorted data to avoid the sorting overhead.

## Quick Reference: When to Use What

| Your Need | Recommended Tool |
|-----------|-----------------|
| Simple linear interp, no extrapolation | `numpy.interp` |
| Linear interp **with** extrapolation | `make_interp_spline(x, y, k=1)` |
| Cubic spline interpolation | `make_interp_spline(x, y, k=3)` or `CubicSpline` |
| Shape-preserving (monotone) interpolation | `PchipInterpolator` |
| Avoid outlier overshoot | `Akima1DInterpolator` |
| Nearest/previous/next (step functions) | `numpy.searchsorted` |
| N-dimensional `y` (batched interpolation) | `make_interp_spline` (supports any ndim) |
| Computing derivatives of interpolant | `CubicSpline` with `nu=` argument |
| Time-series resampling (general) | `make_interp_spline` (flexible, modern) |
| Still using `interp1d` legacy? | It still works — no plans to remove it |
