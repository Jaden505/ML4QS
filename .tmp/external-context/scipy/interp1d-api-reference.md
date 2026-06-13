---
source: SciPy v1.17.0 Official Docs
library: SciPy
package: scipy
topic: interp1d API Reference
fetched: 2026-06-13T00:00:00Z
official_docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
---

# scipy.interpolate.interp1d — API Reference

**Class:** `scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)`

**Status:** LEGACY — This class is considered legacy and will no longer receive updates. While we currently have no plans to remove it, we recommend that new code uses more modern alternatives instead. See the "Recommended Replacements" docs.

## Description

Interpolate a 1-D function. `x` and `y` are arrays of values used to approximate some function f: `y = f(x)`. This class returns a function whose call method uses interpolation to find the value of new points.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | (npoints,) array_like | — | A 1-D array of real values. |
| `y` | (…, npoints, …) array_like | — | An N-D array of real values. The length of `y` along the interpolation axis must equal the length of `x`. Use the `axis` parameter to select correct axis. **Unlike other interpolators, the default interpolation axis is the last axis of `y` (`axis=-1`).** |
| `kind` | str or int | `'linear'` | Specifies the kind of interpolation. String options: `'linear'`, `'nearest'`, `'nearest-up'`, `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'`, `'previous'`, `'next'`. `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'` refer to spline interpolation of zeroth, first, second, or third order. `'previous'` and `'next'` return the previous or next value. `'nearest-up'` rounds up for half-integers; `'nearest'` rounds down. |
| `axis` | int | `-1` | Axis in `y` array corresponding to x-coordinate values. **Defaults to last axis**, unlike other interpolators. |
| `copy` | bool | `True` | If True, makes internal copies of x and y. If False, uses references if possible. |
| `bounds_error` | bool | `None` | If True, raises ValueError when extrapolation is attempted. If False, out-of-bounds values get `fill_value`. By default, error is raised unless `fill_value="extrapolate"`. |
| `fill_value` | array-like, tuple, or `"extrapolate"` | `nan` | If a scalar/ndarray, fills for out-of-range points (default NaN). If a two-element tuple, first used for `x_new < x[0]`, second for `x_new > x[-1]`. Requires `bounds_error=False`. If `"extrapolate"`, extends beyond data range. |
| `assume_sorted` | bool | `False` | If False, x values can be in any order and are sorted first. If True, x must be monotonically increasing. |

## Attributes

- **`fill_value`**: The fill value configured for the interpolator.

## Methods

- **`__call__(x)`**: Evaluate the interpolant at points `x`.

## Notes

- Calling `interp1d` with NaNs present in input values results in **undefined behavior**.
- Input values `x` and `y` must be convertible to float.
- If values in `x` are not unique, the resulting behavior is undefined and specific to the `kind` chosen.
- Not in-scope for Python Array API Standard support.

## Basic Example

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy import interpolate

>>> x = np.arange(0, 10)
>>> y = np.exp(-x/3.0)
>>> f = interpolate.interp1d(x, y)

>>> xnew = np.arange(0, 9, 0.1)
>>> ynew = f(xnew)  # use interpolation function returned by interp1d
>>> plt.plot(x, y, 'o', xnew, ynew, '-')
>>> plt.show()
```

## See Also

- `splrep`, `splev` — Spline interpolation/smoothing based on FITPACK.
- `UnivariateSpline` — Object-oriented wrapper of FITPACK routines.
- `interp2d` — 2-D interpolation (also legacy).
