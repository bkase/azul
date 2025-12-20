var __esm = (fn, res) => () => (fn && (res = fn(fn = 0)), res);

// node_modules/@jax-js/jax/dist/backend-CoVtc9dx.js
function assertNonNull(value) {}
function unzip2(pairs) {
  const lst1 = [];
  const lst2 = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}
function zip(xs, ys) {
  return xs.map((x, i) => [x, ys[i]]);
}
function zipn(...arrays) {
  const minLength = Math.min(...arrays.map((x) => x.length));
  return Array.from({ length: minLength }, (_, i) => arrays.map((arr) => arr[i]));
}
function rep(length, value) {
  if (value instanceof Function)
    return new Array(length).fill(0).map((_, i) => value(i));
  return new Array(length).fill(value);
}
function prod(arr) {
  return arr.reduce((acc, x) => acc * x, 1);
}
function gcd(...values) {
  let a = 0;
  for (let b of values)
    while (b !== 0)
      [a, b] = [b, a % b];
  return Math.abs(a);
}
function intdiv(a, b) {
  return Math.floor(a / b);
}
function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}
function deepEqual(a, b) {
  if (a === b)
    return true;
  if (typeof a !== "object" || typeof b !== "object")
    return false;
  if (a === null || b === null)
    return false;
  if (Object.keys(a).length !== Object.keys(b).length)
    return false;
  for (const key of Object.keys(a))
    if (!deepEqual(a[key], b[key]))
      return false;
  return true;
}
function mapSetUnion(a, b) {
  if (!b)
    return a;
  for (const [key, setB] of b.entries()) {
    const setA = a.get(key);
    if (setA)
      for (const val of setB)
        setA.add(val);
    else
      a.set(key, setB);
  }
  return a;
}
function partitionList(which, array) {
  const falseList = [];
  const trueList = [];
  for (let i = 0;i < which.length; i++)
    if (which[i])
      trueList.push(array[i]);
    else
      falseList.push(array[i]);
  return [falseList, trueList];
}
function isNumberPair(x) {
  return Array.isArray(x) && x.length === 2 && typeof x[0] === "number" && typeof x[1] === "number";
}
function checkAxis(axis, ndim) {
  if (axis < -ndim || axis >= ndim)
    throw new Error(`Axis ${axis} out of bounds for array of dimension ${ndim}`);
  return axis < 0 ? axis + ndim : axis;
}
function normalizeAxis(axis, ndim) {
  if (axis === null)
    return range(ndim);
  else if (typeof axis === "number")
    return [checkAxis(axis, ndim)];
  else {
    const seen = /* @__PURE__ */ new Set;
    for (const a of axis) {
      const ca = checkAxis(a, ndim);
      if (seen.has(ca))
        throw new Error(`Duplicate axis ${ca} passed to function`);
      seen.add(ca);
    }
    return [...seen].sort();
  }
}
function range(start, stop, step = 1) {
  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  const result = [];
  for (let i = start;i < stop; i += step)
    result.push(i);
  return result;
}
function isPermutation(axis, n) {
  if (axis.length !== n)
    return false;
  const seen = /* @__PURE__ */ new Set;
  for (const x of axis) {
    if (x < 0 || x >= n)
      return false;
    seen.add(x);
  }
  return seen.size === n;
}
function invertPermutation(axis) {
  const n = axis.length;
  if (!isPermutation(axis, n))
    throw new Error("invertPermutation: axis is not a permutation");
  const result = new Array(n);
  for (let i = 0;i < n; i++)
    result[axis[i]] = i;
  return result;
}
function generalBroadcast(a, b) {
  const out = [];
  let i = a.length - 1;
  let j = b.length - 1;
  for (;i >= 0 && j >= 0; i--, j--) {
    const x = a[i];
    const y = b[j];
    if (x === y)
      out.push(x);
    else if (x === 1)
      out.push(y);
    else if (y === 1)
      out.push(x);
    else
      throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
  }
  for (;i >= 0; i--)
    out.push(a[i]);
  for (;j >= 0; j--)
    out.push(b[j]);
  return out.reverse();
}
function recursiveFlatten(ar) {
  if (!Array.isArray(ar))
    return [ar];
  return ar.flat(Infinity);
}
function strip1(str) {
  if (str[0] === "(" && str[str.length - 1] === ")")
    return str.slice(1, -1);
  return str;
}
function runWithCache(cache, key, thunk) {
  if (cache.has(key))
    return cache.get(key);
  else {
    const value = thunk();
    cache.set(key, value);
    return value;
  }
}
function promoteTypes(dtype1, dtype2) {
  if (dtype1 === dtype2)
    return dtype1;
  const rank = {
    [DType.Bool]: 0,
    [DType.Uint32]: 1,
    [DType.Int32]: 2,
    [DType.Float16]: 3,
    [DType.Float32]: 4,
    [DType.Float64]: 5
  };
  return rank[dtype1] > rank[dtype2] ? dtype1 : dtype2;
}
function dtypedArray(dtype, data) {
  const { buffer, byteLength, byteOffset } = data;
  const length = byteLength / byteWidth(dtype);
  switch (dtype) {
    case DType.Float32:
      return new Float32Array(buffer, byteOffset, length);
    case DType.Int32:
    case DType.Bool:
      return new Int32Array(buffer, byteOffset, length);
    case DType.Uint32:
      return new Uint32Array(buffer, byteOffset, length);
    case DType.Float16:
      return new Float16Array(buffer, byteOffset, length);
    case DType.Float64:
      return new Float64Array(buffer, byteOffset, length);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}
function dtypedJsArray(dtype, data) {
  switch (dtype) {
    case DType.Float32:
      return new Float32Array(data);
    case DType.Int32:
    case DType.Bool:
      return new Int32Array(data);
    case DType.Uint32:
      return new Uint32Array(data);
    case DType.Float16:
      return new Float16Array(data);
    case DType.Float64:
      return new Float64Array(data);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}
function accessorGlobal(dtype, gid, st, indices) {
  const [index, valid] = st.toAluExp(indices);
  const [, len] = st.views[0].dataRange();
  return AluExp.where(valid, AluExp.globalIndex(dtype, gid, len, index), AluExp.const(dtype, 0));
}
function accessorAluExp(exp, st, indices) {
  const [index, valid] = st.toAluExp(indices);
  return AluExp.where(valid, exp.substitute({ idx: index }), AluExp.const(exp.dtype, 0));
}
function threefry2x32(k0, k1, c0, c1) {
  const rotl32 = (x, r) => (x << r | x >>> 32 - r) >>> 0;
  const ks0 = k0 >>> 0;
  const ks1 = k1 >>> 0;
  const ks2 = (ks0 ^ ks1 ^ 466688986) >>> 0;
  let x0 = c0 + ks0 >>> 0;
  let x1 = c1 + ks1 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks1 >>> 0;
  x1 = x1 + ks2 + 1 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
  x0 = x0 + ks2 >>> 0;
  x1 = x1 + ks0 + 2 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks0 >>> 0;
  x1 = x1 + ks1 + 3 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
  x0 = x0 + ks1 >>> 0;
  x1 = x1 + ks2 + 4 >>> 0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
  x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
  x0 = x0 + ks2 >>> 0;
  x1 = x1 + ks0 + 5 >>> 0;
  return [x0, x1];
}
function _erfapprox$1(x) {
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const t = 1 / (1 + p * x);
  const P_t = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
  return P_t * Math.exp(-x * x);
}
function erf(x) {
  if (x >= 0)
    return 1 - _erfapprox$1(x);
  else
    return _erfapprox$1(-x) - 1;
}
function erfc(x) {
  if (x >= 0)
    return _erfapprox$1(x);
  else
    return 2 - _erfapprox$1(-x);
}
function canonicalizeStrides(shape, strides) {
  const newStrides = [];
  for (let i = 0;i < shape.length; i++)
    if (shape[i] === 1)
      newStrides.push(0);
    else
      newStrides.push(strides[i]);
  return newStrides;
}
function defaultStrides(shape) {
  if (shape.length === 0)
    return [];
  const strides = rep(shape.length, 1);
  for (let i = shape.length - 1;i > 0; i--)
    strides[i - 1] = shape[i] * strides[i];
  return canonicalizeStrides(shape, strides);
}
function mergeDims(shape, strides, mask) {
  if (shape.length === 0)
    return [];
  if (shape.length !== strides.length || mask && shape.length !== mask.length)
    throw new Error("internal: invalid args to mergeDims");
  const ret = [[
    shape[0],
    strides[0],
    strides[0] !== 0 ? shape[0] : 0
  ]];
  let merging = mask ? mask[0][1] - mask[0][0] === 1 : shape[0] === 1;
  for (let i = 1;i < shape.length; i++) {
    const [s, st] = [shape[i], strides[i]];
    if (s === 1)
      continue;
    const [lastS, lastSt, lastPreExpandS] = ret[ret.length - 1];
    if (merging || lastSt === s * st)
      ret[ret.length - 1] = [
        lastS * s,
        st,
        merging ? s : lastPreExpandS * s
      ];
    else
      ret.push([
        s,
        st,
        s
      ]);
    merging = mask ? mask[i][1] - mask[i][0] === 1 : false;
  }
  return ret;
}
function reshapeMask(maskInput, oldShape, newShape) {
  const newMask = [];
  let rMasksI = maskInput.length;
  let rShapeI = oldShape.length;
  let rNewShapeI = newShape.length;
  const rMasks = () => rMasksI ? maskInput[--rMasksI] : [0, 1];
  const rShape = () => rShapeI ? oldShape[--rShapeI] : 1;
  const rNewShape = () => rNewShapeI ? newShape[--rNewShapeI] : 1;
  let currStride = 1;
  let [oldDim, newDim, mask] = [
    rShape(),
    rNewShape(),
    rMasks()
  ];
  while (newMask.length < newShape.length) {
    const [l, r] = mask;
    const nextStride = newDim * currStride;
    if (oldDim === nextStride) {
      newMask.push([intdiv(l, currStride), intdiv(r - 1, currStride) + 1]);
      currStride = 1;
      [oldDim, newDim, mask] = [
        rShape(),
        rNewShape(),
        rMasks()
      ];
    } else if (oldDim > nextStride) {
      if (oldDim % nextStride !== 0)
        return null;
      if ((l % nextStride !== 0 || r % nextStride !== 0) && intdiv(l, nextStride) !== intdiv(r - 1, nextStride))
        return null;
      newMask.push([intdiv(l % nextStride, currStride), intdiv((r - 1) % nextStride, currStride) + 1]);
      [currStride, newDim] = [nextStride, rNewShape()];
    } else {
      const nextMask = rMasks();
      if (!deepEqual(mask, [0, oldDim]) && l !== r && nextMask[1] - nextMask[0] !== 1)
        return null;
      mask = [nextMask[0] * oldDim + l, (nextMask[1] - 1) * oldDim + r];
      oldDim *= rShape();
    }
  }
  return newMask.reverse();
}
function unravel(shape, offset) {
  let acc = 1;
  const idxs = [];
  for (let i = shape.length - 1;i >= 0; i--) {
    const d = shape[i];
    idxs.push(Math.floor(offset / acc) % d);
    acc *= d;
  }
  return idxs.reverse();
}
function unravelAlu(shape, offset) {
  let acc = 1;
  const idxs = [];
  for (let i = shape.length - 1;i >= 0; i--) {
    const d = shape[i];
    idxs.push(AluExp.mod(AluExp.idiv(offset, AluExp.i32(acc)), AluExp.i32(d)));
    acc *= d;
  }
  return idxs.reverse();
}
function applyLast(ar, f) {
  return ar.toSpliced(ar.length - 1, 1, f(ar[ar.length - 1]));
}
function tuneNullopt(kernel) {
  const vars = {};
  vars.gidx = AluExp.special(DType.Int32, "gidx", kernel.size);
  if (kernel.reduction)
    vars.ridx = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
  return {
    exp: kernel.exp.substitute(vars).rewriteGlobalViews().simplify(),
    outputIdxExp: AluExp.special(DType.Int32, "gidx", kernel.size),
    threadCount: kernel.size,
    size: { reduce: kernel.reduction ? kernel.reduction.size : 0 }
  };
}
function _poly(cg, x, as) {
  if (as.length === 0)
    throw new Error("_poly needs at least one coefficient");
  cg.f32.const(as[as.length - 1]);
  for (let i = as.length - 2;i >= 0; i--) {
    cg.local.get(x);
    cg.f32.mul();
    if (as[i] !== 0) {
      cg.f32.const(as[i]);
      cg.f32.add();
    }
  }
}
function wasm_exp(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const k_f = cg.local.declare(cg.f32);
    const k = cg.local.declare(cg.i32);
    const r = cg.local.declare(cg.f32);
    const p = cg.local.declare(cg.f32);
    const scale = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.const(1 / Math.LN2);
    cg.f32.mul();
    cg.f32.nearest();
    cg.local.tee(k_f);
    cg.i32.trunc_sat_f32_s();
    cg.local.set(k);
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.gt_s();
    cg.if(cg.void);
    cg.f32.const(Infinity);
    cg.return();
    cg.end();
    cg.local.get(k);
    cg.i32.const(-126);
    cg.i32.lt_s();
    cg.if(cg.void);
    cg.f32.const(0);
    cg.return();
    cg.end();
    cg.local.get(0);
    cg.local.get(k_f);
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(r);
    _poly(cg, r, [
      1,
      1,
      1 / 2,
      1 / 6,
      1 / 24,
      1 / 120,
      1 / 720
    ]);
    cg.local.set(p);
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.add();
    cg.i32.const(23);
    cg.i32.shl();
    cg.f32.reinterpret_i32();
    cg.local.set(scale);
    cg.local.get(p);
    cg.local.get(scale);
    cg.f32.mul();
  });
}
function wasm_log(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const bits = cg.local.declare(cg.i32);
    const e = cg.local.declare(cg.i32);
    const m = cg.local.declare(cg.f32);
    const t = cg.local.declare(cg.f32);
    const t2 = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.const(0);
    cg.f32.le();
    cg.if(cg.void);
    cg.f32.const(NaN);
    cg.return();
    cg.end();
    cg.local.get(0);
    cg.i32.reinterpret_f32();
    cg.local.tee(bits);
    cg.i32.const(23);
    cg.i32.shr_u();
    cg.i32.const(255);
    cg.i32.and();
    cg.i32.const(127);
    cg.i32.sub();
    cg.local.set(e);
    cg.local.get(bits);
    cg.i32.const(8388607);
    cg.i32.and();
    cg.i32.const(1065353216);
    cg.i32.or();
    cg.f32.reinterpret_i32();
    cg.local.set(m);
    cg.local.get(m);
    cg.f32.const(1);
    cg.f32.sub();
    cg.local.get(m);
    cg.f32.const(1);
    cg.f32.add();
    cg.f32.div();
    cg.local.set(t);
    cg.local.get(t);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.set(t2);
    _poly(cg, t2, [
      2,
      2 / 3,
      2 / 5,
      2 / 7
    ]);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.get(e);
    cg.f32.convert_i32_s();
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.add();
  });
}
function _sincos(cg) {
  const y = cg.local.declare(cg.f32);
  const qf = cg.local.declare(cg.f32);
  const q = cg.local.declare(cg.i32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const sz = cg.local.declare(cg.f32);
  const cz = cg.local.declare(cg.f32);
  cg.local.get(0);
  cg.local.get(0);
  cg.f32.const(1 / (2 * Math.PI));
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.f32.const(2 * Math.PI);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.set(y);
  cg.local.get(y);
  cg.f32.const(2 / Math.PI);
  cg.f32.mul();
  cg.f32.nearest();
  cg.local.tee(qf);
  cg.i32.trunc_f32_s();
  cg.local.set(q);
  cg.local.get(y);
  cg.local.get(qf);
  cg.f32.const(Math.PI / 2);
  cg.f32.mul();
  cg.f32.sub();
  cg.local.tee(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);
  _poly(cg, z2, [
    1,
    -1 / 6,
    1 / 120,
    -1 / 5040
  ]);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(sz);
  _poly(cg, z2, [
    1,
    -1 / 2,
    1 / 24,
    -1 / 720,
    1 / 40320
  ]);
  cg.local.set(cz);
  return {
    q,
    sz,
    cz
  };
}
function wasm_sin(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);
    cg.local.get(cz);
    cg.local.get(sz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}
function wasm_cos(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    const { q, sz, cz } = _sincos(cg);
    const mag = cg.local.declare(cg.f32);
    cg.local.get(sz);
    cg.local.get(cz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.tee(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.add();
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}
function _atan(cg) {
  const x = cg.local.declare(cg.f32);
  const abs_x = cg.local.declare(cg.f32);
  const z = cg.local.declare(cg.f32);
  const z2 = cg.local.declare(cg.f32);
  const p = cg.local.declare(cg.f32);
  cg.local.set(x);
  cg.local.get(x);
  cg.f32.abs();
  cg.local.set(abs_x);
  cg.f32.const(1);
  cg.local.get(abs_x);
  cg.f32.div();
  cg.local.get(abs_x);
  cg.local.get(abs_x);
  cg.f32.const(1);
  cg.f32.ge();
  cg.select();
  cg.local.set(z);
  cg.local.get(z);
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(z2);
  _poly(cg, z2, [
    0.999998614341,
    0.661705427875,
    0.0415796528637
  ]);
  _poly(cg, z2, [
    1,
    0.994987933645,
    0.173698870181
  ]);
  cg.f32.div();
  cg.local.get(z);
  cg.f32.mul();
  cg.local.set(p);
  cg.f32.const(Math.PI / 2);
  cg.local.get(p);
  cg.f32.sub();
  cg.local.get(p);
  cg.local.get(abs_x);
  cg.f32.const(1);
  cg.f32.ge();
  cg.select();
  cg.local.get(x);
  cg.f32.copysign();
}
function wasm_atan(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0);
    _atan(cg);
  });
}
function wasm_asin(cg) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.local.get(0);
    cg.f32.const(1);
    cg.local.get(0);
    cg.local.get(0);
    cg.f32.mul();
    cg.f32.sub();
    cg.f32.sqrt();
    cg.f32.const(1);
    cg.f32.add();
    cg.f32.div();
    _atan(cg);
    cg.f32.const(2);
    cg.f32.mul();
  });
}
function _erfapprox(cg, exp_func) {
  const x = cg.local.declare(cg.f32);
  const t = cg.local.declare(cg.f32);
  cg.local.set(x);
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  cg.f32.const(1);
  cg.f32.const(1);
  cg.f32.const(p);
  cg.local.get(x);
  cg.f32.mul();
  cg.f32.add();
  cg.f32.div();
  cg.local.set(t);
  _poly(cg, t, [
    0,
    a1,
    a2,
    a3,
    a4,
    a5
  ]);
  cg.local.get(x);
  cg.f32.neg();
  cg.local.get(x);
  cg.f32.mul();
  cg.call(exp_func);
  cg.f32.mul();
}
function wasm_erf(cg, exp) {
  return cg.function([cg.f32], [cg.f32], () => {
    cg.f32.const(1);
    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp);
    cg.f32.sub();
    cg.local.get(0);
    cg.f32.copysign();
  });
}
function wasm_erfc(cg, exp) {
  return cg.function([cg.f32], [cg.f32], () => {
    const e = cg.local.declare(cg.f32);
    cg.local.get(0);
    cg.f32.abs();
    _erfapprox(cg, exp);
    cg.local.set(e);
    cg.f32.const(2);
    cg.local.get(e);
    cg.f32.sub();
    cg.local.get(e);
    cg.local.get(0);
    cg.f32.const(0);
    cg.f32.lt();
    cg.select();
  });
}
function wasm_threefry2x32(cg) {
  return cg.function([
    cg.i32,
    cg.i32,
    cg.i32,
    cg.i32
  ], [cg.i32, cg.i32], () => {
    const ks0 = cg.local.declare(cg.i32);
    const ks1 = cg.local.declare(cg.i32);
    const ks2 = cg.local.declare(cg.i32);
    const x0 = cg.local.declare(cg.i32);
    const x1 = cg.local.declare(cg.i32);
    const mix = (rot) => {
      cg.local.get(x0);
      cg.local.get(x1);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.i32.const(rot);
      cg.i32.rotl();
      cg.local.get(x0);
      cg.i32.xor();
      cg.local.set(x1);
    };
    const keySchedule = (k0, k1, round) => {
      cg.local.get(x0);
      cg.local.get(k0);
      cg.i32.add();
      cg.local.set(x0);
      cg.local.get(x1);
      cg.local.get(k1);
      cg.i32.add();
      cg.i32.const(round);
      cg.i32.add();
      cg.local.set(x1);
    };
    cg.local.get(0);
    cg.local.set(ks0);
    cg.local.get(1);
    cg.local.set(ks1);
    cg.local.get(0);
    cg.local.get(1);
    cg.i32.xor();
    cg.i32.const(466688986);
    cg.i32.xor();
    cg.local.set(ks2);
    cg.local.get(2);
    cg.local.get(ks0);
    cg.i32.add();
    cg.local.set(x0);
    cg.local.get(3);
    cg.local.get(ks1);
    cg.i32.add();
    cg.local.set(x1);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks1, ks2, 1);
    mix(17), mix(29), mix(16), mix(24);
    keySchedule(ks2, ks0, 2);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks0, ks1, 3);
    mix(17), mix(29), mix(16), mix(24);
    keySchedule(ks1, ks2, 4);
    mix(13), mix(15), mix(26), mix(6);
    keySchedule(ks2, ks0, 5);
    cg.local.get(x0);
    cg.local.get(x1);
  });
}
function assert(condition, message) {
  if (!condition)
    throw new Error(message || "Assertion failed");
}
function encodeSigned(n) {
  const out = [];
  let more = true;
  while (more) {
    let byte = n & 127;
    n >>= 7;
    if (n === 0 && (byte & 64) === 0 || n === -1 && (byte & 64) !== 0)
      more = false;
    else
      byte |= 128;
    out.push(byte);
  }
  return out;
}
function encodeUnsigned(n) {
  const out = [];
  do {
    let byte = n & 127;
    n = n >>> 7;
    if (n !== 0)
      byte |= 128;
    out.push(byte);
  } while (n !== 0);
  return out;
}
function encodeString(s) {
  const bytes = new TextEncoder().encode(s);
  return [bytes.length, ...bytes];
}
function encodeBlocktype(type) {
  assert(type.length > 0, "blocktype must have at least one type");
  if (type.length === 1)
    return [type[0].typeId];
  return [
    96,
    ...encodeUnsigned(0),
    ...encodeUnsigned(type.length),
    ...type.map((t) => t.typeId)
  ];
}
function encodeOpcode(opcode) {
  if (typeof opcode === "number")
    return [opcode];
  return [opcode[0], ...encodeUnsigned(opcode[1])];
}
function concat(out, inp) {
  out.push(...inp);
}
function UNARY_OP(op, opcode, inType, outType) {
  return function() {
    const t = this.cg._pop();
    assert(t.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inType} -> ${outType})`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._push(this.cg[outType]);
  };
}
function BINARY_OP(op, opcode, typeA, typeB, outType) {
  return function() {
    const b = this.cg._pop();
    const a = this.cg._pop();
    assert(a.typeId === this.cg[typeA].typeId && b.typeId === this.cg[typeB].typeId, `invalid type for ${op} (${typeA}, ${typeB} -> ${outType})`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._push(this.cg[outType]);
  };
}
function LOAD_OP(op, opcode, outType) {
  return function(align = 0, offset = 0) {
    const idxType = this.cg._pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
    this.cg._push(this.cg[outType]);
  };
}
function STORE_OP(op, opcode, inType) {
  return function(align = 0, offset = 0) {
    const valType = this.cg._pop();
    const idxType = this.cg._pop();
    assert(valType.typeId === this.cg[inType].typeId, `invalid value type for ${op} (${inType})`);
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode(opcode));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
  };
}
function VECTOR_OP(op, vopcode, inTypes, outType) {
  return function() {
    for (const inType of inTypes.toReversed()) {
      const actualType = this.cg._pop();
      assert(actualType.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes.join(", ")} -> ${outType})`);
    }
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._push(this.cg[outType]);
  };
}
function VECTOR_OPL(op, vopcode, inTypes, outType) {
  return function(lane) {
    for (const inType of inTypes.toReversed()) {
      const actualType = this.cg._pop();
      assert(actualType.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes} -> ${outType})`);
    }
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._emit(lane);
    this.cg._push(this.cg[outType]);
  };
}
function VECTOR_LOAD_OP(op, vopcode) {
  return function(align = 0, offset = 0) {
    const idxType = this.cg._pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg._emit(encodeOpcode([253, vopcode]));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
    this.cg._push(this.cg.v128);
  };
}
function codegenWasm(kernel) {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;
  if (DEBUG >= 3)
    console.info(`kernel.exp: ${kernel.exp}
tune.exp: ${tune.exp}`);
  const cg = new CodeGenerator;
  cg.memory.import("env", "memory");
  const distinctOps = mapSetUnion(tune.exp.distinctOps(), re?.epilogue.distinctOps());
  const funcs = {};
  if (distinctOps.has(AluOp.Sin))
    funcs.sin = wasm_sin(cg);
  if (distinctOps.has(AluOp.Cos))
    funcs.cos = wasm_cos(cg);
  if (distinctOps.has(AluOp.Asin))
    funcs.asin = wasm_asin(cg);
  if (distinctOps.has(AluOp.Atan))
    funcs.atan = wasm_atan(cg);
  if (distinctOps.has(AluOp.Exp) || distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc))
    funcs.exp = wasm_exp(cg);
  if (distinctOps.has(AluOp.Log))
    funcs.log = wasm_log(cg);
  if (distinctOps.has(AluOp.Erf))
    funcs.erf = wasm_erf(cg, funcs.exp);
  if (distinctOps.has(AluOp.Erfc))
    funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (distinctOps.has(AluOp.Threefry2x32))
    funcs.threefry2x32 = wasm_threefry2x32(cg);
  const kernelFunc = cg.function(rep(kernel.nargs + 1, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);
    cg.loop(cg.void);
    cg.block(cg.void);
    cg.local.get(gidx);
    cg.i32.const(kernel.size);
    cg.i32.ge_u();
    cg.br_if(0);
    cg.local.get(kernel.nargs);
    cg.local.get(gidx);
    cg.i32.const(byteWidth(kernel.dtype));
    cg.i32.mul();
    cg.i32.add();
    if (re) {
      const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
      dty(cg, null, kernel.exp.dtype).const(re.identity);
      cg.local.set(acc);
      const ridx = cg.local.declare(cg.i32);
      cg.i32.const(0);
      cg.local.set(ridx);
      cg.loop(cg.void);
      cg.block(cg.void);
      cg.local.get(ridx);
      cg.i32.const(re.size);
      cg.i32.ge_u();
      cg.br_if(0);
      translateExp(cg, funcs, tune.exp, {
        gidx,
        ridx
      });
      if (re.op === AluOp.Add) {
        cg.local.get(acc);
        if (re.dtype === DType.Bool)
          cg.i32.or();
        else
          dty(cg, re.op, re.dtype).add();
      } else if (re.op === AluOp.Mul) {
        cg.local.get(acc);
        if (re.dtype === DType.Bool)
          cg.i32.and();
        else
          dty(cg, re.op, re.dtype).mul();
      } else if (re.op === AluOp.Min || re.op === AluOp.Max)
        if (isFloatDtype(re.dtype)) {
          cg.local.get(acc);
          if (re.op === AluOp.Min)
            dtyF(cg, re.op, re.dtype).min();
          else
            dtyF(cg, re.op, re.dtype).max();
        } else if ([
          DType.Int32,
          DType.Uint32,
          DType.Bool
        ].includes(re.dtype)) {
          const local = cg.local.declare(cg.i32);
          cg.local.tee(local);
          cg.local.get(acc);
          cg.local.get(local);
          cg.local.get(acc);
          if (re.op === AluOp.Min)
            if (re.dtype === DType.Int32)
              cg.i32.lt_s();
            else
              cg.i32.lt_u();
          else if (re.dtype === DType.Int32)
            cg.i32.gt_s();
          else
            cg.i32.gt_u();
          cg.select();
        } else
          throw new Error(`invalid reduction min/max over ${re.dtype}`);
      else
        throw new Error(`invalid wasm reduction op: ${re.op}`);
      cg.local.set(acc);
      cg.local.get(ridx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(ridx);
      cg.br(1);
      cg.end();
      cg.end();
      translateExp(cg, funcs, kernel.reduction.epilogue, { acc });
    } else
      translateExp(cg, funcs, tune.exp, { gidx });
    dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
    cg.local.get(gidx);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.set(gidx);
    cg.br(1);
    cg.end();
    cg.end();
  });
  cg.export(kernelFunc, "kernel");
  return cg.finish();
}
function translateExp(cg, funcs, exp, ctx) {
  const references = /* @__PURE__ */ new Map;
  const seen = /* @__PURE__ */ new Set;
  const countReferences = (exp$1) => {
    references.set(exp$1, (references.get(exp$1) ?? 0) + 1);
    if (!seen.has(exp$1)) {
      seen.add(exp$1);
      for (const src of exp$1.src)
        countReferences(src);
    }
  };
  const expContext = /* @__PURE__ */ new Map;
  const gen = (exp$1) => {
    if (expContext.has(exp$1))
      return cg.local.get(expContext.get(exp$1));
    const { op, src, dtype, arg } = exp$1;
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      gen(src[0]);
      gen(src[1]);
      if (op === AluOp.Add)
        if (dtype === DType.Bool)
          cg.i32.or();
        else
          dty(cg, op, dtype).add();
      else if (op === AluOp.Sub)
        dty(cg, op, dtype).sub();
      else if (op === AluOp.Mul)
        if (dtype === DType.Bool)
          cg.i32.and();
        else
          dty(cg, op, dtype).mul();
      else if (op === AluOp.Idiv)
        if (isFloatDtype(dtype)) {
          dtyF(cg, op, dtype).div();
          dtyF(cg, op, dtype).trunc();
        } else if (dtype === DType.Uint32)
          cg.i32.div_u();
        else if (dtype === DType.Int32)
          cg.i32.div_s();
        else
          throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === AluOp.Mod)
        if (isFloatDtype(dtype)) {
          const dt = dtyF(cg, op, dtype);
          const a = cg.local.declare(dt);
          const b = cg.local.declare(dt);
          cg.local.set(b);
          cg.local.tee(a);
          cg.local.get(a);
          cg.local.get(b);
          dt.div();
          dt.trunc();
          cg.local.get(b);
          dt.mul();
          dt.sub();
        } else if (dtype === DType.Uint32)
          cg.i32.rem_u();
        else if (dtype === DType.Int32)
          cg.i32.rem_s();
        else
          throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === AluOp.Min || op === AluOp.Max)
        if (isFloatDtype(dtype))
          if (op === AluOp.Min)
            dtyF(cg, op, dtype).min();
          else
            dtyF(cg, op, dtype).max();
        else if (dtype === DType.Int32 || dtype === DType.Uint32) {
          const a = cg.local.declare(cg.i32);
          const b = cg.local.declare(cg.i32);
          cg.local.set(b);
          cg.local.tee(a);
          cg.local.get(b);
          cg.local.get(a);
          cg.local.get(b);
          if (dtype === DType.Int32)
            if (op === AluOp.Min)
              cg.i32.lt_s();
            else
              cg.i32.gt_s();
          else if (op === AluOp.Min)
            cg.i32.lt_u();
          else
            cg.i32.gt_u();
          cg.select();
        } else
          throw new UnsupportedOpError(op, dtype, "wasm");
      else if (op === AluOp.Cmplt) {
        const srcDtype = src[0].dtype;
        if (isFloatDtype(srcDtype))
          dtyF(cg, op, srcDtype).lt();
        else if (srcDtype === DType.Int32)
          cg.i32.lt_s();
        else if (srcDtype === DType.Uint32)
          cg.i32.lt_u();
        else
          throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Cmpne)
        dty(cg, op, src[0].dtype).ne();
      else
        throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      const callFuncF32 = (func) => {
        if (dtype !== DType.Float32)
          if (dtype === DType.Float64)
            cg.f32.demote_f64();
          else
            throw new UnsupportedOpError(op, dtype, "wasm");
        cg.call(func);
        if (dtype === DType.Float64)
          cg.f64.promote_f32();
      };
      if (op === AluOp.Sin)
        gen(src[0]), callFuncF32(funcs.sin);
      else if (op === AluOp.Cos)
        gen(src[0]), callFuncF32(funcs.cos);
      else if (op === AluOp.Asin)
        gen(src[0]), callFuncF32(funcs.asin);
      else if (op === AluOp.Atan)
        gen(src[0]), callFuncF32(funcs.atan);
      else if (op === AluOp.Exp)
        gen(src[0]), callFuncF32(funcs.exp);
      else if (op === AluOp.Log)
        gen(src[0]), callFuncF32(funcs.log);
      else if (op === AluOp.Erf)
        gen(src[0]), callFuncF32(funcs.erf);
      else if (op === AluOp.Erfc)
        gen(src[0]), callFuncF32(funcs.erfc);
      else if (op === AluOp.Sqrt)
        gen(src[0]), dtyF(cg, op, dtype).sqrt();
      else if (op === AluOp.Reciprocal) {
        const dt = dtyF(cg, op, dtype);
        dt.const(1), gen(src[0]), dt.div();
      } else if (op === AluOp.Cast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32 || dtype0 === DType.Bool;
        if (dtype === DType.Int32)
          if (dtype0 === DType.Float32)
            cg.i32.trunc_sat_f32_s();
          else if (dtype0 === DType.Float64)
            cg.i32.trunc_sat_f64_s();
          else if (i32repr)
            ;
          else
            throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === DType.Uint32)
          if (dtype0 === DType.Float32)
            cg.i32.trunc_sat_f32_u();
          else if (dtype0 === DType.Float64)
            cg.i32.trunc_sat_f64_u();
          else if (i32repr)
            ;
          else
            throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === DType.Float32)
          if (dtype0 === DType.Float32)
            ;
          else if (dtype0 === DType.Float64)
            cg.f32.demote_f64();
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f32.convert_i32_s();
          else if (dtype0 === DType.Uint32)
            cg.f32.convert_i32_u();
          else
            throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === DType.Float64)
          if (dtype0 === DType.Float32)
            cg.f64.promote_f32();
          else if (dtype0 === DType.Float64)
            ;
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f64.convert_i32_s();
          else if (dtype0 === DType.Uint32)
            cg.f64.convert_i32_u();
          else
            throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else if (dtype === DType.Bool)
          if (dtype0 === DType.Bool)
            ;
          else if (i32repr)
            cg.i32.const(0), cg.i32.ne();
          else if (dtype0 === DType.Float32)
            cg.f32.const(0), cg.f32.ne();
          else if (dtype0 === DType.Float64)
            cg.f64.const(0), cg.f64.ne();
          else
            throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        else
          throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Bitcast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        if (dtype !== dtype0) {
          const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32;
          if (dtype === DType.Int32 || dtype === DType.Uint32)
            if (dtype0 === DType.Float32)
              cg.i32.reinterpret_f32();
            else if (i32repr)
              ;
            else
              throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          else if (dtype === DType.Float32)
            if (i32repr)
              cg.f32.reinterpret_i32();
            else if (dtype0 === DType.Float32)
              ;
            else
              throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          else
            throw new UnsupportedOpError(op, dtype, "wasm");
        }
      } else
        throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (op === AluOp.Where) {
      gen(src[1]);
      gen(src[2]);
      gen(src[0]);
      cg.select();
    } else if (op === AluOp.Threefry2x32) {
      for (let i = 0;i < 4; i++)
        gen(src[i]);
      cg.call(funcs.threefry2x32);
      if (arg === "xor")
        cg.i32.xor();
      else if (arg === 0)
        cg.drop();
      else if (arg === 1) {
        const local = cg.local.declare(cg.i32);
        cg.local.set(local);
        cg.drop();
        cg.local.get(local);
      } else
        throw new UnsupportedOpError(op, dtype, "wasm", arg);
    } else if (op === AluOp.Const)
      return dty(cg, op, dtype).const(arg);
    else if (op === AluOp.Special)
      return cg.local.get(ctx[arg[0]]);
    else if (op === AluOp.Variable)
      return cg.local.get(ctx[arg]);
    else if (op === AluOp.GlobalIndex) {
      const [gid, len] = arg;
      gen(src[0]);
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.i32.const(0);
      cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
      cg.select();
      cg.i32.const(byteWidth(dtype));
      cg.i32.mul();
      cg.local.get(gid);
      cg.i32.add();
      dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
    } else
      throw new UnsupportedOpError(op, dtype, "wasm");
    if ((references.get(exp$1) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(exp$1, local);
    }
  };
  countReferences(exp);
  gen(exp);
}
function dty(cg, op, dtype) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
function dtyF(cg, op, dtype) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
function getBackend(device) {
  device = device ?? defaultBackend;
  const backend = initializedBackends.get(device);
  if (!backend)
    throw new Error(`${device} backend not ready, call init() first`);
  return backend;
}
var PPrint = class PPrint2 {
  constructor(indents, lines) {
    this.indents = indents;
    this.lines = lines;
  }
  indent(spaces) {
    return new PPrint2(this.indents.map((i) => i + spaces), this.lines);
  }
  concat(...items) {
    return new PPrint2((this.indents ?? []).concat(...items.map((i) => i.indents)), (this.lines ?? []).concat(...items.map((i) => i.lines)));
  }
  stack(other) {
    if (!other.lines.length)
      return this;
    if (!this.lines.length)
      return other;
    const indent = this.indents[this.indents.length - 1];
    const s = this.lines[this.lines.length - 1];
    const indentedBlock = other.indent(indent + s.length);
    return new PPrint2(this.indents.concat(indentedBlock.indents.slice(1)), this.lines.slice(0, -1).concat(s + " ".repeat(other.indents[0]) + other.lines[0], ...indentedBlock.lines.slice(1)));
  }
  toString() {
    return this.lines.map((line, i) => " ".repeat(this.indents[i]) + line).join(`
`);
  }
  static pp(s) {
    const lines = s.toString().split(`
`);
    return new PPrint2(Array(lines.length).fill(0), lines);
  }
}, DEBUG = 0, _stagingbuf, FpHash = class FpHash2 {
  value = 8773157n;
  #update(x) {
    const base = 873192869n;
    const modulus = 3189051996290219n;
    this.value = (this.value * base + x) % modulus;
  }
  update(x) {
    if (typeof x === "string") {
      this.#update(BigInt(x.length));
      for (let i = 0;i < x.length; i++)
        this.#update(BigInt(199 + x.charCodeAt(i)));
    } else if (typeof x === "number")
      if (Number.isInteger(x))
        this.#update(68265653n ^ BigInt(x));
      else {
        _stagingbuf.setFloat64(0, x, true);
        this.#update(_stagingbuf.getBigUint64(0, true));
      }
    else if (typeof x === "boolean")
      this.#update(x ? 69069841n : 63640693n);
    else if (typeof x === "bigint")
      this.#update(x ^ 71657401n);
    else if (x === null)
      this.#update(37832657n);
    else if (x === undefined)
      this.#update(18145117n);
    else
      x.hash(this);
    return this;
  }
  static hash(...values) {
    const h = new FpHash2;
    for (const x of values)
      h.update(x);
    return h.value;
  }
}, DType, byteWidth = (dtype) => {
  switch (dtype) {
    case DType.Float32:
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return 4;
    case DType.Float16:
      return 2;
    case DType.Float64:
      return 8;
    default:
      throw new TypeError(`Unknown dtype: ${dtype}`);
  }
}, isFloatDtype = (dtype) => dtype === DType.Float32 || dtype === DType.Float16 || dtype === DType.Float64, AluExp = class AluExp2 {
  #hash;
  #simplified;
  #range;
  constructor(op, dtype, src, arg = undefined) {
    this.op = op;
    this.dtype = dtype;
    this.src = src;
    this.arg = arg;
    if (AluGroup.RequiredFloat.has(op) && !isFloatDtype(dtype))
      throw new TypeError(`Unsupported dtype for ${op}: ${dtype}`);
    if (op === AluOp.Bitcast && (dtype === DType.Bool || src[0].dtype === DType.Bool || byteWidth(dtype) !== byteWidth(src[0].dtype)))
      throw new TypeError(`Bitcast from ${src[0].dtype} -> ${dtype}`);
    if (op === AluOp.Threefry2x32 && (dtype !== DType.Uint32 || src.some((x) => x.dtype !== DType.Uint32)))
      throw new TypeError("Threefry2x32 requires uint32 types");
  }
  static add(a, b) {
    return new AluExp2(AluOp.Add, a.dtype, [a, b]);
  }
  static sub(a, b) {
    return new AluExp2(AluOp.Sub, a.dtype, [a, b]);
  }
  static mul(a, b) {
    return new AluExp2(AluOp.Mul, a.dtype, [a, b]);
  }
  static idiv(a, b) {
    return new AluExp2(AluOp.Idiv, a.dtype, [a, b]);
  }
  static mod(a, b) {
    return new AluExp2(AluOp.Mod, a.dtype, [a, b]);
  }
  static min(a, b) {
    return new AluExp2(AluOp.Min, a.dtype, [a, b]);
  }
  static max(a, b) {
    return new AluExp2(AluOp.Max, a.dtype, [a, b]);
  }
  static sin(a) {
    return new AluExp2(AluOp.Sin, a.dtype, [a]);
  }
  static cos(a) {
    return new AluExp2(AluOp.Cos, a.dtype, [a]);
  }
  static asin(a) {
    return new AluExp2(AluOp.Asin, a.dtype, [a]);
  }
  static atan(a) {
    return new AluExp2(AluOp.Atan, a.dtype, [a]);
  }
  static exp(a) {
    return new AluExp2(AluOp.Exp, a.dtype, [a]);
  }
  static log(a) {
    return new AluExp2(AluOp.Log, a.dtype, [a]);
  }
  static erf(a) {
    return new AluExp2(AluOp.Erf, a.dtype, [a]);
  }
  static erfc(a) {
    return new AluExp2(AluOp.Erfc, a.dtype, [a]);
  }
  static sqrt(a) {
    return new AluExp2(AluOp.Sqrt, a.dtype, [a]);
  }
  static reciprocal(a) {
    return new AluExp2(AluOp.Reciprocal, a.dtype, [a]);
  }
  static cast(dtype, a) {
    if (a.dtype === dtype)
      return a;
    return new AluExp2(AluOp.Cast, dtype, [a]);
  }
  static bitcast(dtype, a) {
    if (a.dtype === dtype)
      return a;
    return new AluExp2(AluOp.Bitcast, dtype, [a]);
  }
  static threefry2x32(k0, k1, c0, c1, mode = "xor") {
    return new AluExp2(AluOp.Threefry2x32, DType.Uint32, [
      k0,
      k1,
      c0,
      c1
    ], mode);
  }
  static cmplt(a, b) {
    return new AluExp2(AluOp.Cmplt, DType.Bool, [a, b]);
  }
  static cmpne(a, b) {
    return new AluExp2(AluOp.Cmpne, DType.Bool, [a, b]);
  }
  static where(cond, a, b) {
    return new AluExp2(AluOp.Where, a.dtype, [
      cond,
      a,
      b
    ]);
  }
  static const(dtype, value) {
    if (dtype === DType.Bool)
      value = Number(Boolean(value));
    else if (dtype === DType.Int32)
      value = Math.trunc(value) | 0;
    else if (dtype === DType.Uint32)
      value = Math.trunc(value) >>> 0;
    if (typeof value !== "number")
      throw new TypeError(`Expected a number for constant, got ${typeof value}: ${value}`);
    return new AluExp2(AluOp.Const, dtype, [], value);
  }
  static special(dtype, name, n) {
    return new AluExp2(AluOp.Special, dtype, [], [name, n]);
  }
  static variable(dtype, name) {
    return new AluExp2(AluOp.Variable, dtype, [], name);
  }
  static globalIndex(dtype, gid, len, bufidx) {
    return new AluExp2(AluOp.GlobalIndex, dtype, [bufidx], [gid, len]);
  }
  static globalView(dtype, gid, st, indices) {
    return new AluExp2(AluOp.GlobalView, dtype, indices, [gid, st]);
  }
  static f32(value) {
    return AluExp2.const(DType.Float32, value);
  }
  static i32(value) {
    return AluExp2.const(DType.Int32, value);
  }
  static u32(value) {
    return AluExp2.const(DType.Uint32, value);
  }
  static bool(value) {
    return AluExp2.const(DType.Bool, Number(value));
  }
  static f16(value) {
    return AluExp2.const(DType.Float16, value);
  }
  static f64(value) {
    return AluExp2.const(DType.Float64, value);
  }
  not() {
    if (this.dtype !== DType.Bool)
      throw new Error("not() can only be called on boolean expressions");
    return AluExp2.cmpne(this, AluExp2.const(DType.Bool, true));
  }
  getHash() {
    if (this.#hash !== undefined)
      return this.#hash;
    const hasher = new FpHash;
    hasher.update(this.op);
    hasher.update(this.dtype);
    if (this.op === AluOp.Const)
      hasher.update(this.arg);
    else
      hasher.update(JSON.stringify(this.arg));
    hasher.update(this.src.length);
    for (const s of this.src)
      hasher.update(s);
    this.#hash = hasher.value;
    return this.#hash;
  }
  hash(state) {
    state.update(this.getHash());
  }
  substitute(variables) {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.Variable && Object.hasOwn(variables, exp.arg)) {
        if (exp.dtype !== variables[exp.arg].dtype)
          throw new Error(`Type mismatch: ${exp.dtype} vs ${variables[exp.arg].dtype}`);
        return variables[exp.arg];
      }
    });
  }
  reindexGids(gidMap) {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.GlobalIndex) {
        const [gid, len] = exp.arg;
        const newGid = gidMap.get(gid);
        if (newGid !== undefined && newGid !== gid)
          return AluExp2.globalIndex(exp.dtype, newGid, len, exp.src[0]);
      } else if (exp.op === AluOp.GlobalView) {
        const gid = exp.arg[0];
        const newGid = gidMap.get(gid);
        if (newGid !== undefined && newGid !== gid)
          return AluExp2.globalView(exp.dtype, newGid, exp.arg[1], exp.src);
      }
    });
  }
  #computeRange() {
    if (this.#range !== undefined)
      return this.#range;
    const src = this.src;
    const minMax4 = (f) => {
      const [r1, r2] = [src[0].#computeRange(), src[1].#computeRange()];
      const values = [
        f(r1[0], r2[0]),
        f(r1[0], r2[1]),
        f(r1[1], r2[0]),
        f(r1[1], r2[1])
      ];
      return [Math.min(...values), Math.max(...values)];
    };
    let ret;
    switch (this.op) {
      case AluOp.Add:
        ret = [src[0].min + src[1].min, src[0].max + src[1].max];
        break;
      case AluOp.Sub:
        ret = [src[0].min - src[1].max, src[0].max - src[1].min];
        break;
      case AluOp.Mul:
        ret = minMax4((a, b) => a * b);
        break;
      case AluOp.Idiv:
        ret = minMax4((a, b) => Math.trunc(a / b));
        break;
      case AluOp.Mod: {
        let divisorRange = src[1].#computeRange();
        if (divisorRange[0] <= 0 && divisorRange[1] >= 0)
          divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
        if (divisorRange[1] < 0)
          divisorRange = [-divisorRange[1], -divisorRange[0]];
        const maxDivisor = isFloatDtype(this.dtype) ? divisorRange[1] : divisorRange[1] - 1;
        ret = [clamp(src[0].min, -maxDivisor, 0), clamp(src[0].max, 0, maxDivisor)];
        break;
      }
      case AluOp.Min:
        ret = [Math.min(src[0].min, src[1].min), Math.min(src[0].max, src[1].max)];
        break;
      case AluOp.Max:
        ret = [Math.max(src[0].min, src[1].min), Math.max(src[0].max, src[1].max)];
        break;
      case AluOp.Sin:
        ret = [-1, 1];
        break;
      case AluOp.Cos:
        ret = [-1, 1];
        break;
      case AluOp.Asin:
        ret = [-Math.PI / 2, Math.PI / 2];
        break;
      case AluOp.Atan:
        ret = [-Math.PI / 2, Math.PI / 2];
        break;
      case AluOp.Exp:
        ret = [Math.exp(src[0].min), Math.exp(src[0].max)];
        break;
      case AluOp.Log:
        ret = [Math.log(src[0].min), Math.log(src[0].max)];
        break;
      case AluOp.Erf:
        ret = [erf(src[0].min), erf(src[0].max)];
        break;
      case AluOp.Erfc:
        ret = [erfc(src[0].max), erfc(src[0].min)];
        break;
      case AluOp.Sqrt:
        ret = [Math.sqrt(src[0].min), Math.sqrt(src[0].max)];
        break;
      case AluOp.Reciprocal:
        if (src[0].min <= 0 && src[0].max >= 0)
          return [-Infinity, Infinity];
        ret = [1 / src[0].max, 1 / src[0].min];
        break;
      case AluOp.Cast: {
        const wasFloat = isFloatDtype(src[0].dtype);
        const bounded = Number.isFinite(src[0].min) && Number.isFinite(src[0].max);
        if (this.dtype === DType.Bool) {
          const canBeZero = src[0].min <= 0 && src[0].max >= 0;
          const mustBeZero = src[0].min === 0 && src[0].max === 0;
          ret = mustBeZero ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
        } else if (this.dtype === DType.Int32) {
          const a = wasFloat ? clamp(src[0].min, -2147483648, 2147483647) | 0 : src[0].min | 0;
          const b = wasFloat ? clamp(src[0].max, -2147483648, 2147483647) | 0 : src[0].max | 0;
          ret = bounded && a <= b ? [a, b] : [-Infinity, Infinity];
        } else if (this.dtype === DType.Uint32) {
          const a = wasFloat ? clamp(src[0].min, 0, 4294967295) >>> 0 : src[0].min >>> 0;
          const b = wasFloat ? clamp(src[0].max, 0, 4294967295) >>> 0 : src[0].max >>> 0;
          ret = bounded && a <= b ? [a, b] : [0, Infinity];
        } else
          ret = [src[0].min, src[0].max];
        break;
      }
      case AluOp.Cmplt:
        ret = [0, 1];
        break;
      case AluOp.Cmpne:
        ret = [0, 1];
        break;
      case AluOp.Where:
        ret = [Math.min(src[1].min, src[2].min), Math.max(src[1].max, src[2].max)];
        break;
      case AluOp.Const:
        ret = [this.arg, this.arg];
        break;
      case AluOp.Special:
        ret = [0, this.arg[1] - 1];
        break;
      default:
        ret = [-Infinity, Infinity];
    }
    if (isNaN(ret[0]) || isNaN(ret[1]))
      ret = [-Infinity, Infinity];
    if (this.dtype === DType.Bool) {
      ret[0] = clamp(ret[0], 0, 1);
      ret[1] = clamp(ret[1], 0, 1);
    }
    if (this.dtype === DType.Uint32)
      ret[0] = Math.max(0, ret[0]);
    this.#range = ret;
    return ret;
  }
  get min() {
    return this.#computeRange()[0];
  }
  get max() {
    return this.#computeRange()[1];
  }
  constFactor() {
    if (this.op === AluOp.Const)
      return Math.abs(this.arg);
    if (this.op === AluOp.Add)
      return gcd(this.src[0].constFactor(), this.src[1].constFactor());
    if (this.op === AluOp.Mul) {
      if (this.src[0].op === AluOp.Const)
        return Math.abs(this.src[0].arg);
      if (this.src[1].op === AluOp.Const)
        return Math.abs(this.src[1].arg);
    }
    return 1;
  }
  divides(v) {
    if (v === 1)
      return this;
    if (this.op === AluOp.Const && this.arg % v === 0)
      return AluExp2.const(this.dtype, this.arg / v);
    if (this.op === AluOp.Add) {
      const a = this.src[0].divides(v);
      if (a !== null) {
        const b = this.src[1].divides(v);
        if (b !== null)
          return AluExp2.add(a, b);
      }
    }
    if (this.op === AluOp.Mul) {
      const a = this.src[0].divides(v);
      if (a !== null)
        return AluExp2.mul(a, this.src[1]);
      const b = this.src[1].divides(v);
      if (b !== null)
        return AluExp2.mul(this.src[0], b);
    }
    return null;
  }
  #isConstInt() {
    return this.op === AluOp.Const && (this.dtype === DType.Int32 || this.dtype === DType.Uint32);
  }
  *splitOp(sep) {
    if (this.op === sep)
      for (const src of this.src)
        yield* src.splitOp(sep);
    else
      yield this;
  }
  simplify(cache = /* @__PURE__ */ new Map) {
    if (this.#simplified !== undefined)
      return this.#simplified;
    const hash = this.getHash();
    const prevCachedValue = cache.get(hash);
    if (prevCachedValue !== undefined)
      return this.#simplified = prevCachedValue;
    const simplified = this.#simplifyInner(cache);
    const simplifiedHash = simplified.getHash();
    const prevSimplified = cache.get(simplifiedHash);
    if (prevSimplified !== undefined) {
      cache.set(hash, prevSimplified);
      this.#simplified = prevSimplified;
      return prevSimplified;
    } else {
      cache.set(hash, simplified);
      cache.set(simplifiedHash, simplified);
      this.#simplified = simplified;
      return simplified;
    }
  }
  #simplifyInner(cache) {
    const src = this.src.map((x) => x.simplify(cache));
    const { op } = this;
    if (src.every((x) => x.op === AluOp.Const) && !AluGroup.Variable.has(op)) {
      const newExp$1 = new AluExp2(op, this.dtype, src, this.arg);
      return AluExp2.const(this.dtype, newExp$1.evaluate({}));
    }
    if (op !== AluOp.Const && this.min === this.max)
      return AluExp2.const(this.dtype, this.min);
    if (AluGroup.Binary.has(op))
      for (let i = 0;i < 2; i++) {
        if (src[i].op !== AluOp.Const)
          continue;
        const x = src[i].arg;
        if (op === AluOp.Add && x === 0)
          return src[1 - i];
        if (op === AluOp.Sub && i === 1 && x === 0)
          return src[1 - i];
        if (op === AluOp.Mul && x === 1)
          return src[1 - i];
        if (op === AluOp.Mul && x === 0)
          return AluExp2.const(this.dtype, 0);
        if (op === AluOp.Idiv && i === 1 && x === 1)
          return src[1 - i];
        if (op === AluOp.Cmpne && src[i].dtype === DType.Bool && x === 0)
          return src[1 - i];
      }
    if ((op === AluOp.Add || op === AluOp.Sub) && src[1].op === AluOp.Mul) {
      const [a, b] = src[1].src;
      const opNeg = op === AluOp.Add ? AluOp.Sub : AluOp.Add;
      if (a.op === AluOp.Const && a.arg === -1)
        return new AluExp2(opNeg, this.dtype, [src[0], b]);
      else if (b.op === AluOp.Const && b.arg === -1)
        return new AluExp2(opNeg, this.dtype, [src[0], a]);
    }
    if (op === AluOp.Where && src.slice(1).every((s, i) => s.op === AluOp.Const && s.arg === 1 - i))
      return AluExp2.cast(this.dtype, src[0]);
    if (op === AluOp.Cmplt) {
      if (src[0].min >= src[1].max)
        return AluExp2.const(DType.Bool, false);
      if (src[0].max < src[1].min)
        return AluExp2.const(DType.Bool, true);
    }
    if (op === AluOp.Cmpne) {
      if (src[0].max < src[1].min || src[0].min > src[1].max)
        return AluExp2.const(DType.Bool, true);
    }
    if (op === AluOp.Where) {
      if (src[0].max === 0)
        return src[2];
      if (src[0].min === 1)
        return src[1];
    }
    if (op === AluOp.Mod && src[1].op === AluOp.Const && src[0].min >= 0 && src[0].max < src[1].arg)
      return src[0];
    if (op === AluOp.Mod && src[0].op === AluOp.Mod && src[1].#isConstInt() && src[0].src[1].#isConstInt()) {
      const A = src[0].src[1].arg;
      const B = src[1].arg;
      if (A > 0 && B > 0 && (A % B === 0 || B % A === 0))
        return AluExp2.mod(src[0].src[0], AluExp2.const(this.dtype, Math.min(A, B))).simplify();
    }
    if (op === AluOp.Add && src[0].op === AluOp.Mul && src[0].src[1].#isConstInt() && src[1].op === AluOp.Mod && src[1].src[1].#isConstInt() && src[0].src[1].arg === src[1].src[1].arg) {
      const [mul, mod] = src;
      const check = (exp) => {
        return exp.op === AluOp.Idiv && exp.src[1].#isConstInt() && exp.src[1].arg === mod.src[1].arg && exp.src[0] === mod.src[0];
      };
      if (check(mul.src[0]))
        return mod.src[0];
      if (mul.src[0].op === AluOp.Mod) {
        const [x, y] = mul.src[0].src;
        if (check(x))
          return AluExp2.mod(mod.src[0], AluExp2.mul(mod.src[1], y)).simplify(cache);
      }
    }
    if (op === AluOp.Idiv && src[1].#isConstInt()) {
      const [numer, denom] = src;
      const B = denom.arg;
      for (let i = 0;i < 2; i++) {
        if (numer.op === AluOp.Mul && numer.src[i].#isConstInt()) {
          const A = numer.src[i].arg;
          if (A % B === 0) {
            let ret = numer.src[1 - i];
            if (A / B !== 1)
              ret = AluExp2.mul(ret, AluExp2.const(ret.dtype, A / B));
            return ret.simplify(cache);
          }
        }
        for (let j = 0;j < 2; j++)
          if (numer.op === AluOp.Add && numer.src[j].op === AluOp.Mul && numer.src[j].src[i].#isConstInt()) {
            const A = numer.src[j].src[i].arg;
            if (A % B === 0) {
              let ret = numer.src[j].src[1 - i];
              if (A / B !== 1)
                ret = AluExp2.mul(ret, AluExp2.const(ret.dtype, A / B));
              ret = AluExp2.add(ret, AluExp2.idiv(numer.src[1 - j], AluExp2.const(ret.dtype, B)));
              return ret.simplify(cache);
            }
          }
      }
    }
    if (op === AluOp.Mod && src[1].#isConstInt() && src[1].arg > 0 && src[0].min >= 0) {
      const [numer, denom] = src;
      const B = denom.arg;
      for (let i = 0;i < 2; i++)
        if (numer.op === AluOp.Add) {
          if (numer.src[i].#isConstInt()) {
            const A = numer.src[i].arg;
            const x = numer.src[1 - i];
            if (A % B === 0 && x.min >= 0)
              return AluExp2.mod(x, denom).simplify(cache);
          }
          for (let j = 0;j < 2; j++)
            if (numer.src[i].op === AluOp.Mul && numer.src[i].src[j].#isConstInt()) {
              const A = numer.src[i].src[j].arg;
              const x = numer.src[1 - i];
              if (A % B === 0 && x.min >= 0)
                return AluExp2.mod(x, denom).simplify(cache);
            }
        } else if (numer.op === AluOp.Mul) {
          if (numer.src[i].#isConstInt()) {
            const A = numer.src[i].arg;
            if (A % B === 0)
              return AluExp2.const(this.dtype, 0);
            if (A % B === 1)
              return AluExp2.mod(numer.src[1 - i], denom).simplify(cache);
          }
        }
    }
    const commOps = [
      AluOp.Add,
      AluOp.Mul,
      AluOp.Max,
      AluOp.Min
    ];
    if (commOps.includes(op)) {
      const p = (a, b) => new AluExp2(op, this.dtype, [a, b]);
      if (src[0].op === AluOp.Const)
        return p(src[1], src[0]).simplify(cache);
      if (src[0].op === op && src[0].src[1].op === AluOp.Const)
        if (src[1].op === AluOp.Const)
          return p(src[0].src[0], p(src[0].src[1], src[1])).simplify(cache);
        else
          return p(p(src[0].src[0], src[1]), src[0].src[1]).simplify(cache);
      if (src[1].op === op && src[1].src[1].op === AluOp.Const)
        return p(p(src[0], src[1].src[0]), src[1].src[1]).simplify(cache);
    }
    if (op === AluOp.Mod || op === AluOp.Idiv && src[1].#isConstInt()) {
      const [x, y] = src;
      {
        const factors = [];
        const terms = [];
        for (const u of x.splitOp(AluOp.Add)) {
          const factor = u.constFactor();
          factors.push(factor);
          terms.push(u.divides(factor));
        }
        const g = gcd(y.arg, ...factors);
        if (g !== 1) {
          let ret = new AluExp2(op, this.dtype, [factors.map((f, i) => AluExp2.mul(AluExp2.const(terms[i].dtype, f / g), terms[i])).reduceRight((a, x$1) => AluExp2.add(x$1, a)), AluExp2.const(y.dtype, y.arg / g)]);
          if (op === AluOp.Mod)
            ret = AluExp2.mul(ret, AluExp2.const(this.dtype, g));
          return ret.simplify(cache);
        }
      }
      if (y.arg > 0) {
        let [xNoConst, constVal] = [x, 0];
        if (x.op === AluOp.Add && x.src[1].op === AluOp.Const)
          [xNoConst, constVal] = [x.src[0], x.src[1].arg];
        const terms = [];
        const factors = [];
        for (const u of xNoConst.splitOp(AluOp.Add)) {
          const f = u.constFactor();
          const divided = u.divides(f);
          terms.push(divided ?? u);
          factors.push(divided ? f : 1);
        }
        const quotients = factors.map((f) => Math.floor(f / y.arg));
        const remainders = factors.map((f) => f % y.arg);
        const gcdVal = remainders.reduce((g, r) => gcd(g, r), y.arg);
        if (constVal % y.arg !== constVal || gcdVal !== 1 || remainders.some((r, i) => r === 0 || r !== factors[i] && op === AluOp.Mod)) {
          let quo = AluExp2.const(x.dtype, Math.floor(constVal / y.arg));
          let rem = AluExp2.const(x.dtype, Math.floor(constVal % y.arg / gcdVal));
          for (let i = 0;i < terms.length; i++)
            if (op === AluOp.Idiv && remainders[i] !== 0)
              rem = AluExp2.add(rem, AluExp2.mul(AluExp2.const(x.dtype, Math.floor(factors[i] / gcdVal)), terms[i]));
            else {
              rem = AluExp2.add(rem, AluExp2.mul(AluExp2.const(x.dtype, Math.floor(remainders[i] / gcdVal)), terms[i]));
              quo = AluExp2.add(quo, AluExp2.mul(AluExp2.const(x.dtype, quotients[i]), terms[i]));
            }
          if (!((x.min < 0 || rem.min < 0) && remainders.some((r) => r !== 0)))
            if (op === AluOp.Mod)
              return AluExp2.add(AluExp2.mul(AluExp2.const(x.dtype, gcdVal), AluExp2.mod(rem, AluExp2.const(x.dtype, Math.floor(y.arg / gcdVal)))), AluExp2.const(x.dtype, constVal % gcdVal)).simplify(cache);
            else
              return AluExp2.add(AluExp2.idiv(rem, AluExp2.const(x.dtype, Math.floor(y.arg / gcdVal))), quo).simplify(cache);
        }
      }
    }
    const newExp = src.every((s, i) => s === this.src[i]) ? this : new AluExp2(op, this.dtype, src, this.arg);
    return newExp;
  }
  resolve() {
    const x = this.simplify();
    if (x.op === AluOp.Const)
      return x.arg;
    return;
  }
  evaluate(context, globals) {
    if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      const y = this.src[1].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Add:
          return this.dtype === DType.Bool ? Number(x || y) : x + y;
        case AluOp.Sub:
          return x - y;
        case AluOp.Mul:
          return this.dtype === DType.Bool ? Number(x && y) : x * y;
        case AluOp.Idiv:
          return Math.trunc(x / y);
        case AluOp.Mod:
          return x % y;
        case AluOp.Min:
          return Math.min(x, y);
        case AluOp.Max:
          return Math.max(x, y);
        case AluOp.Cmplt:
          return Number(x < y);
        case AluOp.Cmpne:
          return Number(x != y);
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }
    if (AluGroup.Unary.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Sin:
          return Math.sin(x);
        case AluOp.Cos:
          return Math.cos(x);
        case AluOp.Asin:
          return Math.asin(x);
        case AluOp.Atan:
          return Math.atan(x);
        case AluOp.Exp:
          return Math.exp(x);
        case AluOp.Log:
          return Math.log(x);
        case AluOp.Erf:
          return erf(x);
        case AluOp.Erfc:
          return erfc(x);
        case AluOp.Sqrt:
          return Math.sqrt(x);
        case AluOp.Reciprocal:
          return 1 / x;
        case AluOp.Cast: {
          const wasFloat = isFloatDtype(this.src[0].dtype);
          if (this.dtype === DType.Int32)
            return (wasFloat ? clamp(x, -2147483648, 2147483647) : x) | 0;
          else if (this.dtype === DType.Uint32)
            return (wasFloat ? clamp(x, 0, 4294967295) : x) >>> 0;
          else if (isFloatDtype(this.dtype))
            return x;
          else if (this.dtype === DType.Bool)
            return Number(Boolean(x));
          else
            throw new Error(`Unsupported cast to ${this.dtype}`);
        }
        case AluOp.Bitcast: {
          const buf = new ArrayBuffer(byteWidth(this.dtype));
          const view = new DataView(buf);
          const fromType = this.src[0].dtype;
          if (fromType === DType.Float32)
            view.setFloat32(0, x, true);
          else if (fromType === DType.Int32)
            view.setInt32(0, x, true);
          else if (fromType === DType.Uint32)
            view.setUint32(0, x, true);
          else if (fromType === DType.Float16)
            view.setFloat16(0, x, true);
          else if (fromType === DType.Float64)
            view.setFloat64(0, x, true);
          else
            throw new Error(`Unsupported bitcast from ${fromType}`);
          if (this.dtype === DType.Float32)
            return view.getFloat32(0, true);
          else if (this.dtype === DType.Int32)
            return view.getInt32(0, true);
          else if (this.dtype === DType.Uint32)
            return view.getUint32(0, true);
          else if (this.dtype === DType.Float16)
            return view.getFloat16(0, true);
          else if (this.dtype === DType.Float64)
            return view.getFloat64(0, true);
          else
            throw new Error(`Unsupported bitcast to ${this.dtype}`);
        }
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }
    switch (this.op) {
      case AluOp.Where:
        return this.src[0].evaluate(context, globals) ? this.src[1].evaluate(context, globals) : this.src[2].evaluate(context, globals);
      case AluOp.Threefry2x32: {
        const [k0, k1, c0, c1] = this.src.map((x) => x.evaluate(context, globals));
        const [x0, x1] = threefry2x32(k0, k1, c0, c1);
        if (this.arg === "xor")
          return (x0 ^ x1) >>> 0;
        else if (this.arg === 0)
          return x0;
        else if (this.arg === 1)
          return x1;
        else
          throw new Error(`Invalid Threefry2x32 mode: ${this.arg}`);
      }
      case AluOp.Const:
        return this.arg;
      case AluOp.Special: {
        const x = context[this.arg[0]];
        if (x === undefined)
          throw new Error(`Missing special: ${this.arg[0]}`);
        return x;
      }
      case AluOp.Variable: {
        const x = context[this.arg];
        if (x === undefined)
          throw new Error(`Missing variable: ${this.arg}`);
        return x;
      }
      case AluOp.GlobalIndex: {
        if (!globals)
          throw new Error("Missing globals function");
        const gid = this.arg[0];
        const bufidx = this.src[0].evaluate(context, globals);
        return globals(gid, bufidx);
      }
      case AluOp.GlobalView: {
        if (!globals)
          throw new Error("Missing globals function");
        const gid = this.arg[0];
        const st = this.arg[1];
        const [iexpr, vexpr] = st.toAluExp(this.src);
        if (vexpr.evaluate(context, globals)) {
          const bufidx = iexpr.evaluate(context, globals);
          return globals(gid, bufidx);
        } else
          return 0;
      }
      default:
        throw new Error(`Missing implemementation for ${this.op}`);
    }
  }
  toString() {
    const BIN_SYM = {
      [AluOp.Add]: "+",
      [AluOp.Sub]: "-",
      [AluOp.Mul]: "*",
      [AluOp.Idiv]: "/",
      [AluOp.Mod]: "%"
    };
    const CMP_SYM = {
      [AluOp.Cmplt]: "<",
      [AluOp.Cmpne]: "!="
    };
    const UNARY_SYM = { [AluOp.Reciprocal]: "1/" };
    return this.fold((node, parts) => {
      switch (node.op) {
        case AluOp.Const:
          return "" + (node.dtype === DType.Bool ? Boolean(node.arg) : node.arg);
        case AluOp.Variable:
          return `$${node.arg}:${node.dtype}`;
        case AluOp.Special: {
          const [name, n] = node.arg;
          return `#${name}{${n}}`;
        }
        case AluOp.GlobalIndex:
          return `G_${node.arg[0]}<${node.dtype}>[${strip1(parts[0])}]`;
        case AluOp.GlobalView: {
          const [gid, st] = node.arg;
          const shape = st.shape.join(",");
          const lastStrides = st.lastStrides.join(",");
          const cont = st.contiguous ? "c" : "nc";
          return `GV_${gid}<${node.dtype}>{${shape}:${lastStrides}:${cont}}[${parts.map(strip1).join(", ")}]`;
        }
      }
      if (BIN_SYM[node.op])
        return `(${parts[0]} ${BIN_SYM[node.op]} ${parts[1]})`;
      if (CMP_SYM[node.op])
        return `(${parts[0]} ${CMP_SYM[node.op]} ${parts[1]})`;
      if (UNARY_SYM[node.op])
        return `${UNARY_SYM[node.op]}${parts[0]}`;
      if (node.op === AluOp.Cast)
        return `Cast<${node.dtype}>(${strip1(parts[0])})`;
      if (node.op === AluOp.Bitcast)
        return `Bitcast<${node.dtype}>(${strip1(parts[0])})`;
      return `${node.op}(${parts.map(strip1).join(", ")})`;
    });
  }
  fold(reducer) {
    const visited = /* @__PURE__ */ new Map;
    const recurse = (exp) => {
      if (visited.has(exp))
        return visited.get(exp);
      const mappedSrc = exp.src.map((s) => recurse(s));
      const result = reducer(exp, mappedSrc);
      visited.set(exp, result);
      return result;
    };
    return recurse(this);
  }
  some(predicate) {
    const visited = /* @__PURE__ */ new Set;
    const recurse = (exp) => {
      if (visited.has(exp))
        return false;
      if (predicate(exp))
        return true;
      visited.add(exp);
      return exp.src.some(recurse);
    };
    return recurse(this);
  }
  rewrite(visitor) {
    return this.fold((exp, newSrc) => {
      if (newSrc.length === exp.src.length && newSrc.every((s, i) => s === exp.src[i]))
        return visitor(exp) ?? exp;
      else {
        const newExp = new AluExp2(exp.op, exp.dtype, newSrc, exp.arg);
        return visitor(newExp) ?? newExp;
      }
    });
  }
  collect(predicate) {
    const result = [];
    this.fold((exp) => {
      if (predicate(exp))
        result.push(exp);
    });
    return result;
  }
  distinctOps() {
    const ops = /* @__PURE__ */ new Map;
    this.fold((exp) => {
      const s = ops.get(exp.op) ?? /* @__PURE__ */ new Set;
      if (!s.has(exp.dtype)) {
        s.add(exp.dtype);
        ops.set(exp.op, s);
      }
    });
    return ops;
  }
  rewriteGlobalViews() {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.GlobalView) {
        const [gid, st] = exp.arg;
        return accessorGlobal(exp.dtype, gid, st, exp.src);
      }
    });
  }
}, AluOp, AluGroup, AluVar, Kernel = class {
  constructor(nargs, size, exp, reduction) {
    this.nargs = nargs;
    this.size = size;
    this.exp = exp;
    this.reduction = reduction;
    this.exp = exp.simplify();
  }
  hash(state) {
    state.update(this.nargs).update(this.size).update(this.exp).update(this.reduction);
  }
  pprint() {
    let details = PPrint.pp(`exp = ${this.exp}`);
    details = details.concat(PPrint.pp(`size = ${this.size}`));
    if (this.reduction)
      details = details.concat(PPrint.pp(`reduction = ${this.reduction}`));
    return PPrint.pp("{ ").stack(details).stack(PPrint.pp(" }"));
  }
  toString() {
    return this.pprint().toString();
  }
  get dtype() {
    if (this.reduction)
      return this.reduction.epilogue.dtype;
    else
      return this.exp.dtype;
  }
  get bytes() {
    return this.size * byteWidth(this.dtype);
  }
}, Reduction = class {
  constructor(dtype, op, size, epilogue = AluVar.acc(dtype)) {
    this.dtype = dtype;
    this.op = op;
    this.size = size;
    this.epilogue = epilogue;
    if (!AluGroup.Reduce.has(op))
      throw new TypeError(`Unsupported reduction: ${op}`);
    this.epilogue = epilogue.simplify();
  }
  hash(state) {
    state.update(this.dtype).update(this.op).update(this.size).update(this.epilogue);
  }
  toString() {
    return `${this.op}{${this.size}} -> ${this.epilogue}`;
  }
  get identity() {
    if (this.dtype === DType.Bool)
      return this.op === AluOp.Add || this.op === AluOp.Max ? 0 : 1;
    else if (this.dtype === DType.Int32) {
      if (this.op === AluOp.Add)
        return 0;
      else if (this.op === AluOp.Mul)
        return 1;
      else if (this.op === AluOp.Min)
        return -1 >>> 1;
      else if (this.op === AluOp.Max)
        return 1 << 31;
    } else if (this.dtype === DType.Uint32) {
      if (this.op === AluOp.Add)
        return 0;
      else if (this.op === AluOp.Mul)
        return 1;
      else if (this.op === AluOp.Min)
        return -1 >>> 0;
      else if (this.op === AluOp.Max)
        return 0;
    } else if (isFloatDtype(this.dtype)) {
      if (this.op === AluOp.Add)
        return 0;
      else if (this.op === AluOp.Mul)
        return 1;
      else if (this.op === AluOp.Min)
        return Infinity;
      else if (this.op === AluOp.Max)
        return -Infinity;
    }
    throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
  }
  evaluate(...values) {
    if (this.dtype === DType.Bool) {
      if (this.op === AluOp.Add || this.op === AluOp.Max)
        return values.reduce((a, b) => a || b, true);
      else if (this.op === AluOp.Mul || this.op === AluOp.Min)
        return values.reduce((a, b) => a && b, true);
    } else if (this.dtype === DType.Int32) {
      if (this.op === AluOp.Add)
        return values.reduce((a, b) => a + b | 0, 0);
      else if (this.op === AluOp.Mul)
        return values.reduce((a, b) => a * b | 0, 1);
      else if (this.op === AluOp.Min)
        return values.reduce((a, b) => Math.min(a, b), -1 >>> 1);
      else if (this.op === AluOp.Max)
        return values.reduce((a, b) => Math.max(a, b), 1 << 31);
    } else if (this.dtype === DType.Uint32) {
      if (this.op === AluOp.Add)
        return values.reduce((a, b) => a + b >>> 0, 0);
      else if (this.op === AluOp.Mul)
        return values.reduce((a, b) => a * b >>> 0, 1);
      else if (this.op === AluOp.Min)
        return values.reduce((a, b) => Math.min(a, b), -1 >>> 0);
      else if (this.op === AluOp.Max)
        return values.reduce((a, b) => Math.max(a, b), 0);
    } else if (isFloatDtype(this.dtype)) {
      if (this.op === AluOp.Add)
        return values.reduce((a, b) => a + b, 0);
      else if (this.op === AluOp.Mul)
        return values.reduce((a, b) => a * b, 1);
      else if (this.op === AluOp.Min)
        return values.reduce((a, b) => Math.min(a, b), Infinity);
      else if (this.op === AluOp.Max)
        return values.reduce((a, b) => Math.max(a, b), -Infinity);
    }
    throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
  }
}, jstr, View = class View2 {
  #size;
  #contiguous;
  constructor(shape, strides, offset, mask) {
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.mask = mask;
  }
  static create(shape, strides, offset = 0, mask = null) {
    if (shape.some((s) => s < 0))
      throw new Error("View shape must be non-negative");
    strides = strides ? canonicalizeStrides(shape, strides) : defaultStrides(shape);
    if (shape.includes(0))
      return new View2(shape, rep(shape.length, 0), 0, null);
    if (mask !== null && mask.every(([b, e], i) => b === 0 && e === shape[i]))
      mask = null;
    if (mask !== null) {
      const elimDims = [];
      let hasNoData = false;
      for (let i = 0;i < shape.length; i++) {
        const [b, e] = mask[i];
        if (b + 1 >= e)
          elimDims.push(i);
        if (b >= e)
          hasNoData = true;
      }
      if (elimDims.length) {
        if (hasNoData) {
          strides = rep(shape.length, 0);
          offset = 0;
          mask = rep(shape.length, () => [0, 0]);
        }
        for (const i of elimDims) {
          offset += strides[i] * mask[i][0];
          strides[i] = 0;
        }
      }
    }
    return new View2(shape, strides, offset, mask);
  }
  get ndim() {
    return this.shape.length;
  }
  get size() {
    if (this.#size === undefined)
      this.#size = prod(this.shape);
    return this.#size;
  }
  get contiguous() {
    if (this.#contiguous === undefined)
      this.#contiguous = this.size === 0 || this.offset === 0 && this.mask === null && deepEqual(this.strides, defaultStrides(this.shape));
    return this.#contiguous;
  }
  dataRange() {
    if (this.size === 0 || this.mask && this.mask[0][0] === this.mask[0][1])
      return [0, 0];
    let min = this.offset;
    let max = this.offset;
    for (let i = 0;i < this.ndim; i++) {
      let [lo, hi] = this.mask ? this.mask[i] : [0, this.shape[i]];
      --hi;
      const s = this.strides[i];
      if (s > 0) {
        min += s * lo;
        max += s * hi;
      } else if (s < 0) {
        min += s * hi;
        max += s * lo;
      }
    }
    return [min, max + 1];
  }
  toAluExp(idxs) {
    let iexpr = AluExp.i32(this.offset);
    let vexpr = AluExp.bool(true);
    for (let i = this.ndim - 1;i >= 0; i--) {
      const idx = idxs[i];
      if (this.shape[i] !== 1 && this.strides[i] !== 0)
        iexpr = AluExp.add(AluExp.mul(idx, AluExp.i32(this.strides[i])), iexpr);
      if (this.mask) {
        if (this.mask[i][0] !== 0)
          vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][0])).not(), vexpr);
        if (this.mask[i][1] !== this.shape[i])
          vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][1])), vexpr);
      }
    }
    return [iexpr, vexpr];
  }
  compose(v1) {
    const v2 = this;
    if (v2.contiguous)
      return v1;
    if (v1.contiguous) {
      if (deepEqual(v1.shape, v2.shape))
        return v2;
      if (v1.size === v2.size) {
        const ret = v2.reshape(v1.shape);
        if (ret !== null)
          return ret;
      }
    }
    if (v1.mask !== null) {
      const newV1 = v1.shrink(v1.mask);
      const merged = v2.compose(newV1);
      return merged ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]])) : null;
    }
    const origin = unravel(v2.shape, v1.offset);
    const terms = rep(v2.ndim, () => []);
    const strides = rep(v1.ndim, 0);
    for (let d1 = 0;d1 < v1.strides.length; d1++) {
      const st = v1.strides[d1];
      if (st === 0)
        continue;
      const unravelOffset = unravel(v2.shape, v1.offset + st);
      for (let d2 = 0;d2 < v2.ndim; d2++) {
        const o = origin[d2];
        const diff = unravelOffset[d2] - o;
        if (diff === 0)
          continue;
        terms[d2].push([d1, diff]);
        strides[d1] += diff * v2.strides[d2];
      }
    }
    let [mergedSize, mergedTermMin, mergedTermMax] = [
      1,
      0,
      0
    ];
    const extents = [];
    for (let i = v2.ndim - 1;i >= 0; i--) {
      const term = terms[i];
      const s = v2.shape[i];
      let [tmin, tmax] = [origin[i], origin[i]];
      for (const [d1, s1] of term)
        if (s1 > 0)
          tmax += (v1.shape[d1] - 1) * s1;
        else if (s1 < 0)
          tmin += (v1.shape[d1] - 1) * s1;
      mergedTermMin += tmin * mergedSize;
      mergedTermMax += tmax * mergedSize;
      mergedSize *= s;
      if (mergedTermMin >= 0 && mergedTermMax < mergedSize) {
        extents.push([
          mergedSize,
          mergedTermMin,
          mergedTermMax
        ]);
        [mergedSize, mergedTermMin, mergedTermMax] = [
          1,
          0,
          0
        ];
      }
    }
    if (mergedTermMin !== 0 || mergedTermMax !== 0)
      return null;
    extents.reverse();
    const v2Shape = extents.map(([s]) => s);
    if (!deepEqual(v2Shape, v2.shape)) {
      const reshapedV2 = v2.reshape(v2Shape);
      if (reshapedV2 === null)
        return null;
      if (!deepEqual(reshapedV2.shape, v2.shape))
        return reshapedV2.compose(v1);
    }
    if (v2.mask !== null) {
      const newB = rep(v1.ndim, 0);
      const newE = v1.shape.slice();
      let bad = false;
      for (let d2 = 0;d2 < v2.ndim; d2++) {
        const [b, e] = v2.mask[d2];
        const o = origin[d2];
        const term = terms[d2];
        const [_, tmin, tmax] = extents[d2];
        if (b <= tmin && tmax < e)
          continue;
        if (term.length !== 1)
          if (term.length === 0 && newE.length)
            newE[0] = 0;
          else
            bad = true;
        else {
          const [d1, s1] = term[0];
          newB[d1] = Math.max(newB[d1], Math.ceil((s1 > 0 ? b - o : e - o - 1) / s1));
          newE[d1] = Math.min(newE[d1], Math.floor((s1 < 0 ? b - o : e - o - 1) / s1) + 1);
        }
      }
      for (let d1 = 0;d1 < v1.ndim; d1++)
        if (newB[d1] !== 0 || newE[d1] !== v1.shape[d1])
          return v2.compose(View2.create(v1.shape, v1.strides, v1.offset, zip(newB, newE)));
      if (bad)
        return null;
    }
    let finalOffset = v2.offset;
    for (let d2 = 0;d2 < v2.ndim; d2++)
      finalOffset += origin[d2] * v2.strides[d2];
    return View2.create(v1.shape, strides, finalOffset, null);
  }
  minify() {
    const minShape = mergeDims(this.shape, this.strides, this.mask).map((x) => x[0]);
    const nv = this.reshape(minShape);
    return nv ? nv : this;
  }
  pad(arg) {
    if (arg.length !== this.ndim || !arg.every(([b, e]) => b >= 0 && e >= 0))
      throw new Error(`invalid pad ${jstr(arg)} for ${jstr(this.shape)}`);
    if (arg.every(([b, e]) => b === 0 && e === 0))
      return this;
    const zvarg = arg.map(([b, e], i) => [-b, this.shape[i] + e]);
    const mask = arg.map(([b, _e], i) => [b, this.shape[i] + b]);
    return this.#unsafeResize(zvarg, mask);
  }
  shrink(arg) {
    if (arg.length !== this.ndim || !arg.every(([b, e], i) => 0 <= b && b <= e && e <= this.shape[i]))
      throw new Error(`invalid shrink ${jstr(arg)} for ${jstr(this.shape)}`);
    return this.#unsafeResize(arg);
  }
  #unsafeResize(arg, mask) {
    const offset = this.strides.map((s, i) => s * arg[i][0]).reduce((a, b) => a + b, 0);
    if (this.mask) {
      const nmask = this.mask.map(([mx, my], i) => [Math.max(0, Math.min(mx - arg[i][0], arg[i][1] - arg[i][0])), Math.max(0, Math.min(my - arg[i][0], arg[i][1] - arg[i][0]))]);
      mask = mask ? mask.map(([mx, my], i) => [Math.max(mx, nmask[i][0]), Math.min(my, nmask[i][1])]) : nmask;
    }
    return View2.create(arg.map(([b, e]) => e - b), this.strides, this.offset + offset, mask);
  }
  expand(newShape) {
    if (newShape.length !== this.ndim)
      throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
    for (let i = 0;i < this.ndim; i++)
      if (newShape[i] !== this.shape[i] && this.shape[i] !== 1)
        throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
    if (this.size === 0)
      return View2.create(newShape);
    const mask = this.mask ? this.mask.map((m, i) => this.shape[i] === newShape[i] ? m : m[0] === 0 && m[1] === 1 ? [0, newShape[i]] : [0, 0]) : null;
    return View2.create(newShape, this.strides, this.offset, mask);
  }
  permute(axis) {
    if (!isPermutation(axis, this.ndim))
      throw new Error(`Invalid permutation ${jstr(axis)} of len ${this.ndim}`);
    const newShape = axis.map((a) => this.shape[a]);
    const newStrides = axis.map((a) => this.strides[a]);
    const newMask = this.mask ? axis.map((a) => this.mask[a]) : null;
    return View2.create(newShape, newStrides, this.offset, newMask);
  }
  flip(arg) {
    if (arg.length !== this.ndim)
      throw new Error(`Invalid flip ${jstr(arg)} for ${jstr(this.shape)}`);
    const strides = this.strides.slice();
    let offset = this.offset;
    const mask = this.mask ? this.mask.slice() : null;
    for (let i = 0;i < this.ndim; i++) {
      const s = this.shape[i];
      if (arg[i]) {
        strides[i] = -strides[i];
        offset += (s - 1) * this.strides[i];
        if (mask)
          mask[i] = [s - mask[i][1], s - mask[i][0]];
      }
    }
    return View2.create(this.shape, strides, offset, mask);
  }
  reshape(newShape) {
    if (deepEqual(this.shape, newShape))
      return this;
    if (newShape.some((s) => s < 0))
      throw new Error(`Reshape cannot have negative numbers ${jstr(newShape)}`);
    if (this.size !== prod(newShape))
      throw new Error(`Reshape size ${jstr(this.shape)} -> ${jstr(newShape)}`);
    if (this.size === 0)
      return View2.create(newShape);
    if (newShape.length === 0 && this.mask?.some(([b, e]) => b === e))
      return null;
    if (this.contiguous)
      return View2.create(newShape);
    const rStrides = [];
    const merge = mergeDims(this.shape, this.strides, this.mask);
    let rShapeIdx = newShape.length;
    for (let i = merge.length - 1;i >= 0; i--) {
      let [mergedSize, newStride, realSize] = merge[i];
      let acc = 1;
      while (acc < mergedSize && rShapeIdx > 0) {
        const newDim = newShape[--rShapeIdx];
        rStrides.push(newStride * acc);
        acc *= newDim;
        if (acc >= realSize)
          newStride = 0;
      }
      if (acc !== mergedSize)
        return null;
    }
    const newStrides = rep(newShape.length - rStrides.length, 0).concat(rStrides.reverse());
    if (!this.mask)
      return View2.create(newShape, newStrides, this.offset);
    const newMask = reshapeMask(this.mask, this.shape, newShape);
    if (!newMask)
      return null;
    let newOffset = this.offset;
    for (let i = 0;i < this.ndim; i++)
      newOffset += this.strides[i] * this.mask[i][0];
    for (let i = 0;i < newShape.length; i++)
      newOffset -= newStrides[i] * newMask[i][0];
    return View2.create(newShape, newStrides, newOffset, newMask);
  }
}, ShapeTracker = class ShapeTracker2 {
  constructor(views) {
    this.views = views;
  }
  compose(other) {
    if (this.contiguous)
      return other;
    let ret = this;
    for (const v of other.views)
      ret = new ShapeTracker2(ret.views.concat(v)).simplify();
    return ret;
  }
  static fromShape(shape) {
    return new ShapeTracker2([View.create(shape)]);
  }
  get contiguous() {
    return this.views.length === 1 && this.views[0].contiguous;
  }
  get consecutive() {
    return this.views.length === 1 && this.views[0].mask === null && deepEqual(this.views[0].strides, defaultStrides(this.views[0].shape));
  }
  get lastStrides() {
    return this.views[this.views.length - 1].strides;
  }
  get shape() {
    return this.views[this.views.length - 1].shape;
  }
  get size() {
    return this.views[this.views.length - 1].size;
  }
  toAluExp(idxs) {
    let [iexpr, vexpr] = this.views[this.views.length - 1].toAluExp(idxs);
    for (let i = this.views.length - 2;i >= 0; i--) {
      const view = this.views[i].minify();
      const exprs = view.toAluExp(unravelAlu(view.shape, iexpr));
      iexpr = exprs[0];
      vexpr = AluExp.mul(vexpr, exprs[1]);
    }
    return [iexpr.simplify(), vexpr.simplify()];
  }
  simplify() {
    const views = this.views.slice();
    while (views.length >= 2) {
      const newView = views[views.length - 2].compose(views[views.length - 1]);
      if (newView === null)
        break;
      views.splice(views.length - 2, 2, newView);
    }
    return new ShapeTracker2(views);
  }
  pad(arg) {
    return new ShapeTracker2(applyLast(this.views, (x) => x.pad(arg)));
  }
  shrink(arg) {
    return new ShapeTracker2(applyLast(this.views, (x) => x.shrink(arg)));
  }
  expand(newShape) {
    return new ShapeTracker2(applyLast(this.views, (x) => x.expand(newShape)));
  }
  permute(axis) {
    return new ShapeTracker2(applyLast(this.views, (x) => x.permute(axis)));
  }
  flip(arg) {
    return new ShapeTracker2(applyLast(this.views, (x) => x.flip(arg)));
  }
  reshape(newShape) {
    const newView = this.views[this.views.length - 1].reshape(newShape);
    return new ShapeTracker2(newView === null ? this.views.concat(View.create(newShape)) : this.views.toSpliced(this.views.length - 1, 1, newView));
  }
  broadcast(newShape, axis) {
    let st = this;
    if (axis.length > 0) {
      const unsqueezed = [...st.shape];
      for (const i of axis.toSorted())
        unsqueezed.splice(i, 0, 1);
      st = st.reshape(unsqueezed);
    }
    return st.expand(newShape);
  }
  repeat(reps, tile = true) {
    if (reps.length > this.shape.length)
      throw new Error(`Too many repeats ${jstr(reps)} for shape ${jstr(this.shape)}`);
    if (reps.some((c) => c <= 0))
      throw new Error(`Invalid repeats ${jstr(reps)}`);
    if (reps.length === 0)
      return this;
    const noop = this.shape.slice(0, -reps.length);
    const shape = this.shape.slice(-reps.length);
    return this.broadcast([...noop, ...shape.flatMap((s, i) => tile ? [reps[i], s] : [s, reps[i]])], shape.map((_, i) => noop.length + 2 * i + (tile ? 0 : 1))).reshape([...noop, ...shape.map((s, i) => s * reps[i])]);
  }
  moveaxis(i, j) {
    const perm = range(this.shape.length);
    perm.splice(i, 1);
    perm.splice(j, 0, i);
    return this.permute(perm);
  }
  padOrShrink(arg) {
    const padArg = [];
    const shrinkArg = [];
    for (let i = 0;i < arg.length; i++) {
      const [b, e] = arg[i];
      if (b < -this.shape[i] || e < -this.shape[i] || b + e < -this.shape[i])
        throw new Error(`Invalid padOrShrink ${jstr(arg)} for ${jstr(this.shape)}`);
      padArg.push([Math.max(0, b), Math.max(0, e)]);
      shrinkArg.push([Math.max(0, -b), this.shape[i] - Math.max(0, -e)]);
    }
    return this.shrink(shrinkArg).pad(padArg);
  }
}, CpuBackend = class {
  type = "cpu";
  maxArgs = Infinity;
  #buffers;
  #nextSlot;
  constructor() {
    this.#buffers = /* @__PURE__ */ new Map;
    this.#nextSlot = 1;
  }
  malloc(size, initialData) {
    const buffer = new Uint8Array(size);
    if (initialData) {
      if (initialData.byteLength !== size)
        throw new Error("initialData size does not match buffer size");
      buffer.set(initialData);
    }
    const slot = this.#nextSlot++;
    this.#buffers.set(slot, {
      buffer,
      ref: 1
    });
    return slot;
  }
  incRef(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    buffer.ref++;
  }
  decRef(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0)
      this.#buffers.delete(slot);
  }
  async read(slot, start, count) {
    return this.readSync(slot, start, count);
  }
  readSync(slot, start, count) {
    const buffer = this.#getBuffer(slot);
    if (start === undefined)
      start = 0;
    if (count === undefined)
      count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }
  async prepare(kernel) {
    return this.prepareSync(kernel);
  }
  prepareSync(kernel) {
    return new Executable(kernel, undefined);
  }
  dispatch({ kernel }, inputs, outputs) {
    const { exp } = tuneNullopt(kernel);
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    const usedArgs = new Map(exp.collect((exp$1) => exp$1.op === AluOp.GlobalIndex).map((exp$1) => [exp$1.arg[0], exp$1.dtype]));
    const inputArrays = inputBuffers.map((buf, i) => {
      const dtype = usedArgs.get(i);
      if (!dtype)
        return null;
      return dtypedArray(dtype, buf);
    });
    const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);
    const globals = (gid, bufidx) => {
      if (gid < 0 || gid >= inputArrays.length)
        throw new Error("gid out of bounds: " + gid);
      if (bufidx < 0 || bufidx >= inputArrays[gid].length)
        throw new Error("bufidx out of bounds: " + bufidx);
      return inputArrays[gid][bufidx];
    };
    if (!kernel.reduction)
      for (let i = 0;i < kernel.size; i++)
        outputArray[i] = exp.evaluate({ gidx: i }, globals);
    else
      for (let i = 0;i < kernel.size; i++) {
        let acc = kernel.reduction.identity;
        for (let j = 0;j < kernel.reduction.size; j++) {
          const item = exp.evaluate({
            gidx: i,
            ridx: j
          }, globals);
          acc = kernel.reduction.evaluate(acc, item);
        }
        outputArray[i] = kernel.reduction.epilogue.evaluate({ acc });
      }
  }
  #getBuffer(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    return buffer.buffer;
  }
}, WasmAllocator = class {
  #memory;
  #headPtr;
  #freeLists;
  #allocatedBuffers;
  constructor(memory) {
    this.#memory = memory;
    this.#headPtr = 64;
    this.#freeLists = /* @__PURE__ */ new Map;
    this.#allocatedBuffers = /* @__PURE__ */ new Map;
  }
  malloc(size) {
    if (size === 0)
      return 0;
    const sizeClass = this.#findSizeClass(size);
    const freeList = this.#freeLists.get(sizeClass);
    let ptr;
    if (freeList && freeList.length > 0)
      ptr = freeList.pop();
    else
      ptr = this.#bumpAlloc(sizeClass);
    this.#allocatedBuffers.set(ptr, sizeClass);
    return ptr;
  }
  free(ptr) {
    if (ptr === 0)
      return;
    const sizeClass = this.#allocatedBuffers.get(ptr);
    if (sizeClass === undefined)
      throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
    const freeList = this.#freeLists.get(sizeClass);
    if (freeList)
      freeList.push(ptr);
    else
      this.#freeLists.set(sizeClass, [ptr]);
    this.#allocatedBuffers.delete(ptr);
  }
  #bumpAlloc(size) {
    const ptr = this.#headPtr;
    size = size + 63 & -64;
    this.#headPtr += size;
    if (ptr + size > this.#memory.buffer.byteLength)
      this.#memory.grow((ptr + size + 65535 >> 16) - (this.#memory.buffer.byteLength >> 16));
    return ptr;
  }
  #findSizeClass(size) {
    if (size <= 512)
      return size + 63 & -64;
    if (size <= 2048)
      return size + 511 & -512;
    if (size <= 65536) {
      let sizeClass = 4096;
      while (sizeClass < size)
        sizeClass *= 2;
      return sizeClass;
    }
    return size + 65535 & -65536;
  }
  getStats() {
    const freeListSizes = /* @__PURE__ */ new Map;
    for (const [sizeClass, freeList] of this.#freeLists)
      if (freeList.length > 0)
        freeListSizes.set(sizeClass, freeList.length);
    return {
      totalAllocated: this.#headPtr,
      freeListSizes
    };
  }
}, magicModuleHeader, moduleVersion, Function_ = class {
  inputTypes;
  outputTypes;
  body;
  locals = [];
  constructor(inputTypes, outputTypes, body) {
    this.inputTypes = inputTypes;
    this.outputTypes = outputTypes;
    this.body = body || (() => {});
  }
  emit() {
    this.locals = [];
    this.body();
  }
}, Memory = class {
  min = 0;
  max = 0;
  isShared = false;
  aString = "";
  bString = "";
  constructor(cg) {
    this.cg = cg;
  }
  pages(min, max = 0) {
    assert(this.min === 0 && this.max === 0);
    this.min = min;
    this.max = max;
    return this;
  }
  export(a) {
    assert(!this.isImport && !this.isExport, "already set");
    this.aString = a;
    return this;
  }
  shared(isShared) {
    this.isShared = isShared;
    return this;
  }
  import(a, b) {
    assert(!this.isImport && !this.isExport, "already set");
    this.aString = a;
    this.bString = b;
    return this;
  }
  size() {
    this.cg._emit(63);
    this.cg._emit(0);
  }
  grow() {
    this.cg._emit(64);
    this.cg._emit(0);
  }
  get isImport() {
    return this.aString.length > 0 && this.bString.length > 0;
  }
  get isExport() {
    return this.aString.length > 0 && this.bString.length === 0;
  }
}, CodeGenerator = class {
  local;
  i32;
  f32;
  f64;
  v128;
  i32x4;
  f32x4;
  memory;
  void = {
    typeId: 64,
    name: "void"
  };
  #functions = [];
  #importedFunctions = [];
  #exportedFunctions = /* @__PURE__ */ new Map;
  #curFunction = null;
  #curBytes = [];
  #typeStack = [];
  #blockFrames = [];
  constructor() {
    this.local = new Local(this);
    this.i32 = new I32(this);
    this.f32 = new F32(this);
    this.f64 = new F64(this);
    this.v128 = new V128(this);
    this.i32x4 = new I32x4(this);
    this.f32x4 = new F32x4(this);
    this.memory = new Memory(this);
  }
  unreachable() {
    this._emit(0);
  }
  nop() {
    this._emit(1);
  }
  block(...type) {
    this.#blockFrames.push({
      idx: this.#typeStack.length,
      ty: type
    });
    this._emit(2);
    this._emit(encodeBlocktype(type));
  }
  loop(...type) {
    this.#blockFrames.push({
      idx: this.#typeStack.length,
      ty: type
    });
    this._emit(3);
    this._emit(encodeBlocktype(type));
  }
  if(...type) {
    assert(this._pop().typeId === this.i32.typeId, "if_: expected i32");
    this.#blockFrames.push({
      idx: this.#typeStack.length,
      ty: type
    });
    this._emit(4);
    this._emit(encodeBlocktype(type));
  }
  else() {
    assert(this.#blockFrames.length > 0, "else: no block to else");
    const frame = this.#blockFrames[this.#blockFrames.length - 1];
    this.#typeStack = this.#typeStack.slice(0, frame.idx);
    this._emit(5);
  }
  end() {
    const frame = this.#blockFrames.pop();
    assert(frame !== undefined, "end: no block to end");
    this.#typeStack = this.#typeStack.slice(0, frame.idx);
    for (const ty of frame.ty)
      if (ty.typeId !== this.void.typeId)
        this._push(ty);
    this._emit(11);
  }
  br(depth) {
    this._emit(12);
    this._emit(encodeUnsigned(depth));
  }
  br_if(depth) {
    assert(this._pop().typeId === this.i32.typeId, "br_if: expected i32");
    this._emit(13);
    this._emit(encodeUnsigned(depth));
  }
  br_table(...depths) {
    assert(this._pop().typeId === this.i32.typeId, "br_table: expected i32");
    assert(depths.length > 0, "br_table: expected at least one default depth");
    this._emit(14);
    this._emit(encodeUnsigned(depths.length - 1));
    for (const d of depths)
      this._emit(encodeUnsigned(d));
  }
  return() {
    this._emit(15);
  }
  call(fn) {
    const totalFunctions = this.#importedFunctions.length + this.#functions.length;
    assert(fn < totalFunctions, "function index does not exist");
    const func = fn < this.#importedFunctions.length ? this.#importedFunctions[fn] : this.#functions[fn - this.#importedFunctions.length];
    for (let i = func.inputTypes.length - 1;i >= 0; i--) {
      const argType = this._pop();
      assert(argType.typeId === func.inputTypes[i].typeId, `call: argument ${i} type mismatch, expected ${func.inputTypes[i].name} got ${argType.name}`);
    }
    for (const outputType of func.outputTypes)
      this._push(outputType);
    this._emit(16);
    this._emit(encodeUnsigned(fn));
  }
  drop() {
    this._pop();
    this._emit(26);
  }
  select() {
    assert(this._pop().typeId === this.i32.typeId, "select: expected i32 condition");
    const [b, a] = [this._pop(), this._pop()];
    assert(a.typeId === b.typeId, "select: expected same type for both operands");
    this._push(a);
    this._emit(27);
  }
  importFunction(module, name, inputTypes, outputTypes) {
    if (this.#functions.length > 0)
      throw new Error("function imports must precede defining functions");
    const idx = this.#importedFunctions.length;
    this.#importedFunctions.push({
      module,
      name,
      inputTypes,
      outputTypes
    });
    return idx;
  }
  export(fn, name) {
    this.#exportedFunctions.set(fn, name);
  }
  function(inputTypes, outputTypes, body) {
    const idx = this.#importedFunctions.length + this.#functions.length;
    this.#functions.push(new Function_(inputTypes, outputTypes, body));
    return idx;
  }
  _declareLocal(type) {
    assert(this.#curFunction !== null, "No current function");
    const idx = this.#curFunction.locals.length + this.#curFunction.inputTypes.length;
    this.#curFunction.locals.push(type);
    return idx;
  }
  _inputTypes() {
    assert(this.#curFunction !== null, "No current function");
    return this.#curFunction.inputTypes;
  }
  _locals() {
    assert(this.#curFunction !== null, "No current function");
    return this.#curFunction.locals;
  }
  _push(type) {
    if (!type)
      throw new Error(`pushing type ${type}`);
    this.#typeStack.push(type);
  }
  _pop() {
    assert(this.#typeStack.length > 0, "popping empty stack");
    return this.#typeStack.pop();
  }
  _emit(bytes) {
    if (typeof bytes === "number")
      this.#curBytes.push(bytes);
    else
      this.#curBytes.push(...bytes);
  }
  finish() {
    this.#curBytes = [];
    const emittedBytes = [];
    concat(emittedBytes, magicModuleHeader);
    concat(emittedBytes, moduleVersion);
    const typeSectionBytes = [];
    const totalFunctionTypes = this.#importedFunctions.length + this.#functions.length;
    concat(typeSectionBytes, encodeUnsigned(totalFunctionTypes));
    for (const f of [...this.#importedFunctions, ...this.#functions]) {
      typeSectionBytes.push(96);
      concat(typeSectionBytes, encodeUnsigned(f.inputTypes.length));
      for (const t of f.inputTypes)
        typeSectionBytes.push(t.typeId);
      concat(typeSectionBytes, encodeUnsigned(f.outputTypes.length));
      for (const t of f.outputTypes)
        typeSectionBytes.push(t.typeId);
    }
    emittedBytes.push(1);
    concat(emittedBytes, encodeUnsigned(typeSectionBytes.length));
    concat(emittedBytes, typeSectionBytes);
    const importSectionBytes = [];
    const numImports = this.#importedFunctions.length + (this.memory.isImport ? 1 : 0);
    if (numImports > 0) {
      concat(importSectionBytes, encodeUnsigned(numImports));
      for (let i = 0;i < this.#importedFunctions.length; i++) {
        const f = this.#importedFunctions[i];
        concat(importSectionBytes, encodeString(f.module));
        concat(importSectionBytes, encodeString(f.name));
        importSectionBytes.push(0);
        concat(importSectionBytes, encodeUnsigned(i));
      }
      if (this.memory.isImport) {
        concat(importSectionBytes, encodeString(this.memory.aString));
        concat(importSectionBytes, encodeString(this.memory.bString));
        importSectionBytes.push(2);
        if (this.memory.min && this.memory.max) {
          if (this.memory.isShared)
            importSectionBytes.push(3);
          else
            importSectionBytes.push(1);
          concat(importSectionBytes, encodeUnsigned(this.memory.min));
          concat(importSectionBytes, encodeUnsigned(this.memory.max));
        } else {
          assert(!this.memory.isShared, "shared memory must have a max size");
          importSectionBytes.push(0);
          concat(importSectionBytes, encodeUnsigned(this.memory.min));
        }
      }
      emittedBytes.push(2);
      concat(emittedBytes, encodeUnsigned(importSectionBytes.length));
      concat(emittedBytes, importSectionBytes);
    }
    const functionSectionBytes = [];
    concat(functionSectionBytes, encodeUnsigned(this.#functions.length));
    for (let i = 0;i < this.#functions.length; i++) {
      const typeIndex = this.#importedFunctions.length + i;
      concat(functionSectionBytes, encodeUnsigned(typeIndex));
    }
    emittedBytes.push(3);
    concat(emittedBytes, encodeUnsigned(functionSectionBytes.length));
    concat(emittedBytes, functionSectionBytes);
    const memorySectionBytes = [];
    if (!this.memory.isImport && (this.memory.min || this.memory.max)) {
      memorySectionBytes.push(1);
      if (this.memory.min && this.memory.max) {
        if (this.memory.isShared)
          memorySectionBytes.push(3);
        else
          memorySectionBytes.push(1);
        concat(memorySectionBytes, encodeUnsigned(this.memory.min));
        concat(memorySectionBytes, encodeUnsigned(this.memory.max));
      } else {
        assert(!this.memory.isShared, "shared memory must have a max size");
        memorySectionBytes.push(0);
        concat(memorySectionBytes, encodeUnsigned(this.memory.min));
      }
      emittedBytes.push(5);
      concat(emittedBytes, encodeUnsigned(memorySectionBytes.length));
      concat(emittedBytes, memorySectionBytes);
    }
    const exportSectionBytes = [];
    const numExports = this.#exportedFunctions.size + (this.memory.isExport ? 1 : 0);
    concat(exportSectionBytes, encodeUnsigned(numExports));
    if (this.memory.isExport) {
      concat(exportSectionBytes, encodeString(this.memory.aString));
      exportSectionBytes.push(2);
      exportSectionBytes.push(0);
    }
    for (const [key, name] of this.#exportedFunctions.entries()) {
      concat(exportSectionBytes, encodeString(name));
      exportSectionBytes.push(0);
      concat(exportSectionBytes, encodeUnsigned(key));
    }
    emittedBytes.push(7);
    concat(emittedBytes, encodeUnsigned(exportSectionBytes.length));
    concat(emittedBytes, exportSectionBytes);
    const codeSectionBytes = [];
    concat(codeSectionBytes, encodeUnsigned(this.#functions.length));
    for (const f of this.#functions) {
      this.#typeStack = [];
      this.#blockFrames = [{
        idx: 0,
        ty: f.outputTypes
      }];
      this.#curFunction = f;
      this.#curBytes = [];
      f.emit();
      this.end();
      const bodyBytes = [...this.#curBytes];
      this.#curBytes = [];
      concat(this.#curBytes, encodeUnsigned(f.locals.length));
      for (const l of f.locals) {
        this._emit(1);
        this._emit(l.typeId);
      }
      const headerBytes = [...this.#curBytes];
      const fnSize = headerBytes.length + bodyBytes.length;
      concat(codeSectionBytes, encodeUnsigned(fnSize));
      concat(codeSectionBytes, headerBytes);
      concat(codeSectionBytes, bodyBytes);
    }
    this.#curFunction = null;
    emittedBytes.push(10);
    concat(emittedBytes, encodeUnsigned(codeSectionBytes.length));
    concat(emittedBytes, codeSectionBytes);
    return new Uint8Array(emittedBytes);
  }
}, Local = class {
  constructor(cg) {
    this.cg = cg;
  }
  declare(type) {
    return this.cg._declareLocal(type);
  }
  get(idx) {
    assert(Number.isInteger(idx), "getting non-integer local");
    const inputTypes = this.cg._inputTypes();
    if (idx < inputTypes.length)
      this.cg._push(inputTypes[idx]);
    else
      this.cg._push(this.cg._locals()[idx - inputTypes.length]);
    this.cg._emit(32);
    this.cg._emit(encodeUnsigned(idx));
  }
  set(idx) {
    const t = this.cg._pop();
    const inputTypes = this.cg._inputTypes();
    const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
    assert(expectedType.typeId === t.typeId, "can't set local to this value (wrong type)");
    this.cg._emit(33);
    this.cg._emit(encodeUnsigned(idx));
  }
  tee(idx) {
    const t = this.cg._pop();
    const inputTypes = this.cg._inputTypes();
    const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
    assert(expectedType.typeId === t.typeId, "can't tee local to this value (wrong type)");
    this.cg._emit(34);
    this.cg._emit(encodeUnsigned(idx));
    this.cg._push(expectedType);
  }
}, I32 = class {
  constructor(cg) {
    this.cg = cg;
  }
  get typeId() {
    return 127;
  }
  get name() {
    return "i32";
  }
  const(i) {
    this.cg._emit(65);
    this.cg._emit(encodeSigned(i));
    this.cg._push(this);
  }
  clz = UNARY_OP("clz", 103, "i32", "i32");
  ctz = UNARY_OP("ctz", 104, "i32", "i32");
  popcnt = UNARY_OP("popcnt", 105, "i32", "i32");
  lt_s = BINARY_OP("lt_s", 72, "i32", "i32", "i32");
  lt_u = BINARY_OP("lt_u", 73, "i32", "i32", "i32");
  gt_s = BINARY_OP("gt_s", 74, "i32", "i32", "i32");
  gt_u = BINARY_OP("gt_u", 75, "i32", "i32", "i32");
  le_s = BINARY_OP("le_s", 76, "i32", "i32", "i32");
  le_u = BINARY_OP("le_u", 77, "i32", "i32", "i32");
  ge_s = BINARY_OP("ge_s", 78, "i32", "i32", "i32");
  ge_u = BINARY_OP("ge_u", 79, "i32", "i32", "i32");
  add = BINARY_OP("add", 106, "i32", "i32", "i32");
  sub = BINARY_OP("sub", 107, "i32", "i32", "i32");
  mul = BINARY_OP("mul", 108, "i32", "i32", "i32");
  div_s = BINARY_OP("div_s", 109, "i32", "i32", "i32");
  div_u = BINARY_OP("div_u", 110, "i32", "i32", "i32");
  rem_s = BINARY_OP("rem_s", 111, "i32", "i32", "i32");
  rem_u = BINARY_OP("rem_u", 112, "i32", "i32", "i32");
  and = BINARY_OP("and", 113, "i32", "i32", "i32");
  or = BINARY_OP("or", 114, "i32", "i32", "i32");
  xor = BINARY_OP("xor", 115, "i32", "i32", "i32");
  shl = BINARY_OP("shl", 116, "i32", "i32", "i32");
  shr_s = BINARY_OP("shr_s", 117, "i32", "i32", "i32");
  shr_u = BINARY_OP("shr_u", 118, "i32", "i32", "i32");
  rotl = BINARY_OP("rotl", 119, "i32", "i32", "i32");
  rotr = BINARY_OP("rotr", 120, "i32", "i32", "i32");
  eqz = BINARY_OP("eqz", 69, "i32", "i32", "i32");
  eq = BINARY_OP("eq", 70, "i32", "i32", "i32");
  ne = BINARY_OP("ne", 71, "i32", "i32", "i32");
  trunc_f32_s = UNARY_OP("trunc_f32_s", 168, "f32", "i32");
  trunc_f32_u = UNARY_OP("trunc_f32_u", 169, "f32", "i32");
  trunc_f64_s = UNARY_OP("trunc_f64_s", 170, "f64", "i32");
  trunc_f64_u = UNARY_OP("trunc_f64_u", 171, "f64", "i32");
  load = LOAD_OP("load", 40, "i32");
  load8_s = LOAD_OP("load8_s", 44, "i32");
  load8_u = LOAD_OP("load8_u", 45, "i32");
  load16_s = LOAD_OP("load16_s", 46, "i32");
  load16_u = LOAD_OP("load16_u", 47, "i32");
  store = STORE_OP("store", 54, "i32");
  store8 = STORE_OP("store8", 58, "i32");
  store16 = STORE_OP("store16", 59, "i32");
  reinterpret_f32 = UNARY_OP("reinterpret_f32", 188, "f32", "i32");
  trunc_sat_f32_s = UNARY_OP("trunc_sat_f32_s", [252, 0], "f32", "i32");
  trunc_sat_f32_u = UNARY_OP("trunc_sat_f32_u", [252, 1], "f32", "i32");
  trunc_sat_f64_s = UNARY_OP("trunc_sat_f64_s", [252, 2], "f64", "i32");
  trunc_sat_f64_u = UNARY_OP("trunc_sat_f64_u", [252, 3], "f64", "i32");
}, F32 = class {
  constructor(cg) {
    this.cg = cg;
  }
  get typeId() {
    return 125;
  }
  get name() {
    return "f32";
  }
  const(f) {
    this.cg._emit(67);
    const buffer = /* @__PURE__ */ new ArrayBuffer(4);
    new DataView(buffer).setFloat32(0, f, true);
    const bytes = new Uint8Array(buffer);
    for (let i = 0;i < 4; i++)
      this.cg._emit(bytes[i]);
    this.cg._push(this);
  }
  load = LOAD_OP("load", 42, "f32");
  store = STORE_OP("store", 56, "f32");
  eq = BINARY_OP("eq", 91, "f32", "f32", "i32");
  ne = BINARY_OP("ne", 92, "f32", "f32", "i32");
  lt = BINARY_OP("lt", 93, "f32", "f32", "i32");
  gt = BINARY_OP("gt", 94, "f32", "f32", "i32");
  le = BINARY_OP("le", 95, "f32", "f32", "i32");
  ge = BINARY_OP("ge", 96, "f32", "f32", "i32");
  abs = UNARY_OP("abs", 139, "f32", "f32");
  neg = UNARY_OP("neg", 140, "f32", "f32");
  ceil = UNARY_OP("ceil", 141, "f32", "f32");
  floor = UNARY_OP("floor", 142, "f32", "f32");
  trunc = UNARY_OP("trunc", 143, "f32", "f32");
  nearest = UNARY_OP("nearest", 144, "f32", "f32");
  sqrt = UNARY_OP("sqrt", 145, "f32", "f32");
  add = BINARY_OP("add", 146, "f32", "f32", "f32");
  sub = BINARY_OP("sub", 147, "f32", "f32", "f32");
  mul = BINARY_OP("mul", 148, "f32", "f32", "f32");
  div = BINARY_OP("div", 149, "f32", "f32", "f32");
  min = BINARY_OP("min", 150, "f32", "f32", "f32");
  max = BINARY_OP("max", 151, "f32", "f32", "f32");
  copysign = BINARY_OP("copysign", 152, "f32", "f32", "f32");
  convert_i32_s = UNARY_OP("convert_i32_s", 178, "i32", "f32");
  convert_i32_u = UNARY_OP("convert_i32_u", 179, "i32", "f32");
  demote_f64 = UNARY_OP("demote_f64", 182, "f64", "f32");
  reinterpret_i32 = UNARY_OP("reinterpret_i32", 190, "i32", "f32");
}, F64 = class {
  constructor(cg) {
    this.cg = cg;
  }
  get typeId() {
    return 124;
  }
  get name() {
    return "f64";
  }
  const(f) {
    this.cg._emit(68);
    const buffer = /* @__PURE__ */ new ArrayBuffer(8);
    new DataView(buffer).setFloat64(0, f, true);
    const bytes = new Uint8Array(buffer);
    for (let i = 0;i < 8; i++)
      this.cg._emit(bytes[i]);
    this.cg._push(this);
  }
  load = LOAD_OP("load", 43, "f64");
  store = STORE_OP("store", 57, "f64");
  eq = BINARY_OP("eq", 97, "f64", "f64", "i32");
  ne = BINARY_OP("ne", 98, "f64", "f64", "i32");
  lt = BINARY_OP("lt", 99, "f64", "f64", "i32");
  gt = BINARY_OP("gt", 100, "f64", "f64", "i32");
  le = BINARY_OP("le", 101, "f64", "f64", "i32");
  ge = BINARY_OP("ge", 102, "f64", "f64", "i32");
  abs = UNARY_OP("abs", 153, "f64", "f64");
  neg = UNARY_OP("neg", 154, "f64", "f64");
  ceil = UNARY_OP("ceil", 155, "f64", "f64");
  floor = UNARY_OP("floor", 156, "f64", "f64");
  trunc = UNARY_OP("trunc", 157, "f64", "f64");
  nearest = UNARY_OP("nearest", 158, "f64", "f64");
  sqrt = UNARY_OP("sqrt", 159, "f64", "f64");
  add = BINARY_OP("add", 160, "f64", "f64", "f64");
  sub = BINARY_OP("sub", 161, "f64", "f64", "f64");
  mul = BINARY_OP("mul", 162, "f64", "f64", "f64");
  div = BINARY_OP("div", 163, "f64", "f64", "f64");
  min = BINARY_OP("min", 164, "f64", "f64", "f64");
  max = BINARY_OP("max", 165, "f64", "f64", "f64");
  copysign = BINARY_OP("copysign", 166, "f64", "f64", "f64");
  convert_i32_s = UNARY_OP("convert_i32_s", 183, "i32", "f64");
  convert_i32_u = UNARY_OP("convert_i32_u", 184, "i32", "f64");
  promote_f32 = UNARY_OP("promote_f32", 187, "f32", "f64");
}, V128 = class {
  constructor(cg) {
    this.cg = cg;
  }
  get typeId() {
    return 123;
  }
  get name() {
    return "v128";
  }
  load = VECTOR_LOAD_OP("load", 0);
  load32x2_s = VECTOR_LOAD_OP("load32x2_s", 5);
  load32x2_u = VECTOR_LOAD_OP("load32x2_u", 6);
  load32_splat = VECTOR_LOAD_OP("load32_splat", 9);
  load32_zero = VECTOR_LOAD_OP("load32_zero", 92);
  store(align = 0, offset = 0) {
    const valType = this.cg._pop();
    assert(valType.typeId === this.cg.v128.typeId, `invalid type for store`);
    const idxType = this.cg._pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for store`);
    this.cg._emit(253);
    this.cg._emit(encodeUnsigned(11));
    this.cg._emit(encodeUnsigned(align));
    this.cg._emit(encodeUnsigned(offset));
  }
  not = VECTOR_OP("not", 77, ["v128"], "v128");
  and = VECTOR_OP("and", 78, ["v128", "v128"], "v128");
  andnot = VECTOR_OP("andnot", 79, ["v128", "v128"], "v128");
  or = VECTOR_OP("or", 80, ["v128", "v128"], "v128");
  xor = VECTOR_OP("xor", 81, ["v128", "v128"], "v128");
  bitselect = VECTOR_OP("bitselect", 82, [
    "v128",
    "v128",
    "v128"
  ], "v128");
  any_true = VECTOR_OP("any_true", 83, ["v128"], "i32");
}, I32x4, F32x4, WasmBackend = class {
  type = "wasm";
  maxArgs = 64;
  #memory;
  #nextSlot;
  #allocator;
  #buffers;
  constructor() {
    this.#memory = new WebAssembly.Memory({ initial: 0 });
    this.#allocator = new WasmAllocator(this.#memory);
    this.#nextSlot = 1;
    this.#buffers = /* @__PURE__ */ new Map;
  }
  malloc(size, initialData) {
    const ptr = this.#allocator.malloc(size);
    if (initialData) {
      if (initialData.byteLength !== size)
        throw new Error("initialData size does not match buffer size");
      new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
    }
    const slot = this.#nextSlot++;
    this.#buffers.set(slot, {
      ptr,
      size,
      ref: 1
    });
    return slot;
  }
  incRef(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    buffer.ref++;
  }
  decRef(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.#allocator.free(buffer.ptr);
      this.#buffers.delete(slot);
    }
  }
  async read(slot, start, count) {
    return this.readSync(slot, start, count);
  }
  readSync(slot, start, count) {
    const buffer = this.#getBuffer(slot);
    if (start === undefined)
      start = 0;
    if (count === undefined)
      count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }
  async prepare(kernel) {
    return this.prepareSync(kernel);
  }
  prepareSync(kernel) {
    const bytes = codegenWasm(kernel);
    const module = new WebAssembly.Module(bytes);
    return new Executable(kernel, { module });
  }
  dispatch(exe, inputs, outputs) {
    const instance = new WebAssembly.Instance(exe.data.module, { env: { memory: this.#memory } });
    const func = instance.exports.kernel;
    const ptrs = [...inputs, ...outputs].map((slot) => this.#buffers.get(slot).ptr);
    func(...ptrs);
  }
  #getBuffer(slot) {
    const buffer = this.#buffers.get(slot);
    if (!buffer)
      throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }
}, initializedBackends, defaultBackend, Executable = class {
  constructor(kernel, data) {
    this.kernel = kernel;
    this.data = data;
  }
}, SlotError, UnsupportedOpError;
var init_backend_CoVtc9dx = __esm(() => {
  _stagingbuf = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
  DType = /* @__PURE__ */ function(DType$1) {
    DType$1["Float32"] = "float32";
    DType$1["Int32"] = "int32";
    DType$1["Uint32"] = "uint32";
    DType$1["Bool"] = "bool";
    DType$1["Float16"] = "float16";
    DType$1["Float64"] = "float64";
    return DType$1;
  }({});
  AluOp = /* @__PURE__ */ function(AluOp$1) {
    AluOp$1["Add"] = "Add";
    AluOp$1["Sub"] = "Sub";
    AluOp$1["Mul"] = "Mul";
    AluOp$1["Idiv"] = "Idiv";
    AluOp$1["Mod"] = "Mod";
    AluOp$1["Min"] = "Min";
    AluOp$1["Max"] = "Max";
    AluOp$1["Sin"] = "Sin";
    AluOp$1["Cos"] = "Cos";
    AluOp$1["Asin"] = "Asin";
    AluOp$1["Atan"] = "Atan";
    AluOp$1["Exp"] = "Exp";
    AluOp$1["Log"] = "Log";
    AluOp$1["Erf"] = "Erf";
    AluOp$1["Erfc"] = "Erfc";
    AluOp$1["Sqrt"] = "Sqrt";
    AluOp$1["Reciprocal"] = "Reciprocal";
    AluOp$1["Cast"] = "Cast";
    AluOp$1["Bitcast"] = "Bitcast";
    AluOp$1["Cmplt"] = "Cmplt";
    AluOp$1["Cmpne"] = "Cmpne";
    AluOp$1["Where"] = "Where";
    AluOp$1["Threefry2x32"] = "Threefry2x32";
    AluOp$1["Const"] = "Const";
    AluOp$1["Special"] = "Special";
    AluOp$1["Variable"] = "Variable";
    AluOp$1["GlobalIndex"] = "GlobalIndex";
    AluOp$1["GlobalView"] = "GlobalView";
    return AluOp$1;
  }({});
  AluGroup = {
    Binary: new Set([
      AluOp.Add,
      AluOp.Sub,
      AluOp.Mul,
      AluOp.Idiv,
      AluOp.Mod,
      AluOp.Min,
      AluOp.Max
    ]),
    Unary: new Set([
      AluOp.Sin,
      AluOp.Cos,
      AluOp.Asin,
      AluOp.Atan,
      AluOp.Exp,
      AluOp.Log,
      AluOp.Erf,
      AluOp.Erfc,
      AluOp.Sqrt,
      AluOp.Reciprocal,
      AluOp.Cast,
      AluOp.Bitcast
    ]),
    Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
    Variable: new Set([
      AluOp.Special,
      AluOp.Variable,
      AluOp.GlobalIndex,
      AluOp.GlobalView
    ]),
    Reduce: new Set([
      AluOp.Add,
      AluOp.Mul,
      AluOp.Min,
      AluOp.Max
    ]),
    RequiredFloat: new Set([
      AluOp.Sin,
      AluOp.Cos,
      AluOp.Asin,
      AluOp.Atan,
      AluOp.Exp,
      AluOp.Log,
      AluOp.Erf,
      AluOp.Erfc,
      AluOp.Sqrt,
      AluOp.Reciprocal
    ])
  };
  AluVar = {
    gidx: AluExp.variable(DType.Int32, "gidx"),
    ridx: AluExp.variable(DType.Int32, "ridx"),
    acc: (dtype) => AluExp.variable(dtype, "acc"),
    idx: AluExp.variable(DType.Int32, "idx"),
    unroll: AluExp.variable(DType.Int32, "unroll"),
    upcast: AluExp.variable(DType.Int32, "upcast")
  };
  jstr = JSON.stringify;
  magicModuleHeader = [
    0,
    97,
    115,
    109
  ];
  moduleVersion = [
    1,
    0,
    0,
    0
  ];
  I32x4 = class extends V128 {
    splat = VECTOR_OP("splat", 17, ["i32"], "v128");
    extract_lane = VECTOR_OPL("extract_lane", 27, ["v128"], "i32");
    replace_lane = VECTOR_OPL("replace_lane", 28, ["v128", "i32"], "v128");
    eq = VECTOR_OP("eq", 55, ["v128", "v128"], "v128");
    ne = VECTOR_OP("ne", 56, ["v128", "v128"], "v128");
    lt_s = VECTOR_OP("lt_s", 57, ["v128", "v128"], "v128");
    lt_u = VECTOR_OP("lt_u", 58, ["v128", "v128"], "v128");
    gt_s = VECTOR_OP("gt_s", 59, ["v128", "v128"], "v128");
    gt_u = VECTOR_OP("gt_u", 60, ["v128", "v128"], "v128");
    le_s = VECTOR_OP("le_s", 61, ["v128", "v128"], "v128");
    le_u = VECTOR_OP("le_u", 62, ["v128", "v128"], "v128");
    ge_s = VECTOR_OP("ge_s", 63, ["v128", "v128"], "v128");
    ge_u = VECTOR_OP("ge_u", 64, ["v128", "v128"], "v128");
    abs = VECTOR_OP("abs", 160, ["v128"], "v128");
    neg = VECTOR_OP("neg", 161, ["v128"], "v128");
    all_true = VECTOR_OP("all_true", 163, ["v128"], "i32");
    bitmask = VECTOR_OP("bitmask", 164, ["v128"], "i32");
    shl = VECTOR_OP("shl", 171, ["v128", "i32"], "v128");
    shr_s = VECTOR_OP("shr_s", 172, ["v128", "i32"], "v128");
    shr_u = VECTOR_OP("shr_u", 173, ["v128", "i32"], "v128");
    add = VECTOR_OP("add", 174, ["v128", "v128"], "v128");
    sub = VECTOR_OP("sub", 177, ["v128", "v128"], "v128");
    mul = VECTOR_OP("mul", 181, ["v128", "v128"], "v128");
    min_s = VECTOR_OP("min_s", 182, ["v128", "v128"], "v128");
    min_u = VECTOR_OP("min_u", 183, ["v128", "v128"], "v128");
    max_s = VECTOR_OP("max_s", 184, ["v128", "v128"], "v128");
    max_u = VECTOR_OP("max_u", 185, ["v128", "v128"], "v128");
  };
  F32x4 = class extends V128 {
    splat = VECTOR_OP("splat", 19, ["f32"], "v128");
    extract_lane = VECTOR_OPL("extract_lane", 31, ["v128"], "f32");
    replace_lane = VECTOR_OPL("replace_lane", 32, ["v128", "f32"], "v128");
    eq = VECTOR_OP("eq", 65, ["v128", "v128"], "v128");
    ne = VECTOR_OP("ne", 66, ["v128", "v128"], "v128");
    lt = VECTOR_OP("lt", 67, ["v128", "v128"], "v128");
    gt = VECTOR_OP("gt", 68, ["v128", "v128"], "v128");
    le = VECTOR_OP("le", 69, ["v128", "v128"], "v128");
    ge = VECTOR_OP("ge", 70, ["v128", "v128"], "v128");
    ceil = VECTOR_OP("ceil", 103, ["v128"], "v128");
    floor = VECTOR_OP("floor", 104, ["v128"], "v128");
    trunc = VECTOR_OP("trunc", 105, ["v128"], "v128");
    nearest = VECTOR_OP("nearest", 106, ["v128"], "v128");
    abs = VECTOR_OP("abs", 224, ["v128"], "v128");
    neg = VECTOR_OP("neg", 225, ["v128"], "v128");
    sqrt = VECTOR_OP("sqrt", 227, ["v128"], "v128");
    add = VECTOR_OP("add", 228, ["v128", "v128"], "v128");
    sub = VECTOR_OP("sub", 229, ["v128", "v128"], "v128");
    mul = VECTOR_OP("mul", 230, ["v128", "v128"], "v128");
    div = VECTOR_OP("div", 231, ["v128", "v128"], "v128");
    min = VECTOR_OP("min", 232, ["v128", "v128"], "v128");
    max = VECTOR_OP("max", 233, ["v128", "v128"], "v128");
    pmin = VECTOR_OP("pmin", 234, ["v128", "v128"], "v128");
    pmax = VECTOR_OP("pmax", 235, ["v128", "v128"], "v128");
  };
  initializedBackends = /* @__PURE__ */ new Map;
  initializedBackends.set("cpu", new CpuBackend);
  if (typeof WebAssembly !== "undefined")
    initializedBackends.set("wasm", new WasmBackend);
  defaultBackend = initializedBackends.has("wasm") ? "wasm" : "cpu";
  SlotError = class extends Error {
    constructor(slot) {
      super(`Used a buffer that is invalid or already freed: ${slot}`);
    }
  };
  UnsupportedOpError = class extends Error {
    constructor(op, dtype, device, arg) {
      let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
      if (arg !== undefined)
        msg += ` with arg ${JSON.stringify(arg)}`;
      super(msg);
    }
  };
});

// main.ts
import init2, {
  action_space_size,
  new_game_state,
  observation_size
} from "./pkg/azul_wasm.js";

// node_modules/@jax-js/jax/dist/chunk-Cl8Af3a2.js
var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true
    });
};

// node_modules/@jax-js/jax/dist/index.js
init_backend_CoVtc9dx();
var tree_exports = {};
__export(tree_exports, {
  JsTreeDef: () => JsTreeDef,
  NodeType: () => NodeType,
  dispose: () => dispose,
  flatten: () => flatten,
  leaves: () => leaves,
  map: () => map,
  ref: () => ref,
  structure: () => structure,
  unflatten: () => unflatten
});
var JsArray$1 = globalThis.Array;
var NodeType = /* @__PURE__ */ function(NodeType$1) {
  NodeType$1["Array"] = "Array";
  NodeType$1["Object"] = "Object";
  NodeType$1["Leaf"] = "Leaf";
  return NodeType$1;
}({});
var JsTreeDef = class JsTreeDef2 {
  static leaf = new JsTreeDef2(NodeType.Leaf, null, []);
  constructor(nodeType, nodeMetadata, childTreedefs) {
    this.nodeType = nodeType;
    this.nodeMetadata = nodeMetadata;
    this.childTreedefs = childTreedefs;
  }
  get size() {
    return this.nodeType === NodeType.Leaf ? 1 : this.childTreedefs.reduce((a, b) => a + b.size, 0);
  }
  toString(root = true) {
    if (root)
      return "JsTreeDef(" + this.toString(false) + ")";
    switch (this.nodeType) {
      case NodeType.Leaf:
        return "*";
      case NodeType.Array:
        return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
      case NodeType.Object: {
        const parts = [];
        for (let i = 0;i < this.childTreedefs.length; i++)
          parts.push(`${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`);
        return `{${parts.join(", ")}}`;
      }
    }
  }
  equals(other) {
    return this.nodeType === other.nodeType && deepEqual(this.nodeMetadata, other.nodeMetadata) && this.childTreedefs.length === other.childTreedefs.length && this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]));
  }
};
function quoteObjectKey(key$1) {
  if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key$1))
    return key$1;
  return JSON.stringify(key$1);
}
function flatten(tree) {
  const leaves$1 = [];
  const treedef = _flatten(tree, leaves$1);
  return [leaves$1, treedef];
}
function _flatten(tree, leaves$1) {
  if (JsArray$1.isArray(tree)) {
    const childTrees = tree.map((c) => _flatten(c, leaves$1));
    return new JsTreeDef(NodeType.Array, null, childTrees);
  } else if (typeof tree === "object" && tree !== null && tree.constructor === Object) {
    const [keys, values] = unzip2(Object.entries(tree));
    const childTrees = values.map((c) => _flatten(c, leaves$1));
    return new JsTreeDef(NodeType.Object, keys, childTrees);
  } else {
    leaves$1.push(tree);
    return JsTreeDef.leaf;
  }
}
function leaves(tree) {
  return flatten(tree)[0];
}
function structure(tree) {
  return flatten(tree)[1];
}
function unflatten(treedef, leaves$1) {
  return _unflatten(treedef, leaves$1[Symbol.iterator]());
}
function _unflatten(treedef, leaves$1) {
  switch (treedef.nodeType) {
    case NodeType.Leaf: {
      const { value, done } = leaves$1.next();
      if (done)
        throw new TypeError("Ran out of leaves while unflattening JsTree");
      return value;
    }
    case NodeType.Array:
      return treedef.childTreedefs.map((c) => _unflatten(c, leaves$1));
    case NodeType.Object: {
      const obj = {};
      for (let i = 0;i < treedef.childTreedefs.length; i++)
        obj[treedef.nodeMetadata[i]] = _unflatten(treedef.childTreedefs[i], leaves$1);
      return obj;
    }
  }
}
function map(fn, tree, ...rest) {
  const [leaves$1, treedef] = flatten(tree);
  const restLeaves = rest.map((x) => flatten(x)[0]);
  const resultLeaves = [];
  for (let i = 0;i < leaves$1.length; i++)
    resultLeaves.push(fn(leaves$1[i], ...restLeaves.map((x) => x[i])));
  return unflatten(treedef, resultLeaves);
}
function ref(tree) {
  return map((x) => x.ref, tree);
}
function dispose(tree) {
  if (tree)
    map((x) => x.dispose(), tree);
}
function checkConvShape(lhsShape, rhsShape, { strides, padding, lhsDilation, rhsDilation }) {
  if (lhsShape.length !== rhsShape.length)
    throw new Error(`conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`);
  const n = lhsShape.length - 2;
  if (n < 0)
    throw new Error("conv() requires at least 2D inputs");
  if (strides.length !== n)
    throw new Error("conv() strides != spatial dims");
  if (padding.length !== n)
    throw new Error("conv() padding != spatial dims");
  if (lhsDilation.length !== n)
    throw new Error("conv() lhsDilation != spatial dimensions");
  if (rhsDilation.length !== n)
    throw new Error("conv() rhsDilation != spatial dimensions");
  if (lhsShape[1] !== rhsShape[1])
    throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
  const outShape = [lhsShape[0], rhsShape[0]];
  for (let i = 0;i < n; i++) {
    if (strides[i] <= 0 || !Number.isInteger(strides[i]))
      throw new Error(`conv() strides[${i}] must be a positive integer`);
    if (padding[i].length !== 2 || !padding[i].every(Number.isInteger))
      throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
    if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i]))
      throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
    if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i]))
      throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);
    const [x, k] = [lhsShape[i + 2], rhsShape[i + 2]];
    if (k <= 0)
      throw new Error("conv() kernel size must be positive");
    const [pl, pr] = padding[i];
    if (pl < -x || pr < -x || pl + pr < -x)
      throw new Error(`conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`);
    const kernelSize = (k - 1) * rhsDilation[i] + 1;
    const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
    if (kernelSize > inSize)
      throw new Error(`conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`);
    outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
  }
  return outShape;
}
function checkPoolShape(inShape, window, strides) {
  if (strides.length !== window.length)
    throw new Error("pool() strides != window dims");
  if (window.length > inShape.length)
    throw new Error("pool() window has more dimensions than input");
  const outShape = inShape.slice(0, inShape.length - window.length);
  for (let i = 0;i < window.length; i++) {
    const k = window[i];
    const s = strides[i];
    const size$1 = inShape[inShape.length - window.length + i];
    if (k <= 0 || !Number.isInteger(k))
      throw new Error(`pool() window[${i}] must be a positive integer`);
    if (k > size$1)
      throw new Error(`pool() window[${i}]=${k} > input size ${size$1}`);
    if (s <= 0 || !Number.isInteger(s))
      throw new Error(`pool() strides[${i}] must be a positive integer`);
    outShape.push(Math.ceil((size$1 - k + 1) / s));
  }
  return outShape.concat(window);
}
function pool(st, ks, strides = 1, dilation = 1) {
  if (ks.length === 0)
    return st;
  if (st.shape.length < ks.length)
    throw new Error("pool() called with too many dimensions");
  if (typeof strides === "number")
    strides = rep(ks.length, strides);
  if (typeof dilation === "number")
    dilation = rep(ks.length, dilation);
  if (strides.some((s) => s <= 0 || !Number.isInteger(s)))
    throw new Error("pool() strides must be positive integers");
  if (dilation.some((d) => d <= 0 || !Number.isInteger(d)))
    throw new Error("pool() dilation must be positive integers");
  const noop = st.shape.slice(0, -ks.length);
  const i_ = st.shape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    st = st.padOrShrink([...noop.map(() => [0, 0]), ...zipn(i_, o_, s_).map(([i, o, s]) => [0, o * s - i])]);
    st = st.reshape([...noop, ...zip(o_, s_).flatMap(([o, s]) => [o, s])]).shrink([...noop.map((x) => [0, x]), ...zip(o_, ks).flatMap(([o, k]) => [[0, o], [0, k]])]);
    st = st.permute([
      ...range(noop.length),
      ...ks.map((_, j) => noop.length + 2 * j),
      ...ks.map((_, j) => noop.length + 2 * j + 1)
    ]);
    return st;
  }
  const f_ = zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
  const kidf = zipn(ks, i_, d_, f_);
  st = st.repeat([...rep(noop.length, 1), ...kidf.map(([k, i, d, f]) => Math.ceil(k * (i * f + d) / i))]);
  st = st.shrink([...noop.map((x) => [0, x]), ...kidf.map(([k, i, d, f]) => [0, k * (i * f + d)])]).reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);
  const kos = zipn(ks, o_, s_);
  st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o, s]) => [[0, k], [0, o * s]])]).reshape([...noop, ...kos.flat(1)]);
  st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o]) => [
    [0, k],
    [0, o],
    [0, 1]
  ])]).reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);
  st = st.permute([
    ...range(noop.length),
    ...ks.map((_, j) => noop.length + 2 * j + 1),
    ...ks.map((_, j) => noop.length + 2 * j)
  ]);
  return st;
}
function poolTranspose(st, inShape, ks, strides = 1, dilation = 1) {
  if (ks.length === 0)
    return st;
  if (typeof strides === "number")
    strides = rep(ks.length, strides);
  if (typeof dilation === "number")
    dilation = rep(ks.length, dilation);
  const noop = inShape.slice(0, -ks.length);
  const i_ = inShape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    st = st.permute([...range(noop.length), ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j])]);
    st = st.pad([...noop.map(() => [0, 0]), ...zip(s_, ks).flatMap(([s, k]) => [[0, 0], [0, s - k]])]).reshape([...noop, ...zip(o_, s_).map(([o, s]) => o * s)]);
    st = st.padOrShrink([...noop.map(() => [0, 0]), ...zipn(i_, o_, s_).map(([i, o, s]) => [0, i - o * s])]);
    return st.reshape(st.shape.concat(rep(ks.length, 1)));
  }
  if (!deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length)))
    throw new Error("poolTranspose() called with mismatched output shape");
  const f_ = zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
  const kidf = zipn(ks, i_, d_, f_);
  const kos = zipn(ks, o_, s_);
  st = st.permute([...range(noop.length), ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j])]);
  st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [
    k,
    o,
    1
  ])]).pad([...noop.map(() => [0, 0]), ...s_.flatMap((s) => [
    [0, 0],
    [0, 0],
    [0, s - 1]
  ])]);
  st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([...noop.map(() => [0, 0]), ...kidf.flatMap(([_k, i, d, f], j) => [[0, 0], [0, i * f + d - o_[j] * s_[j]]])]);
  st = st.reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))]).pad([...noop.map(() => [0, 0]), ...kidf.map(([k, i, d, f]) => [0, Math.ceil(k * (i * f + d) / i) * i - k * (i * f + d)])]);
  st = st.reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [Math.ceil(k * (i * f + d) / i), i])]).permute([
    ...range(noop.length),
    ...ks.map((_, j) => noop.length + 2 * j + 1),
    ...ks.map((_, j) => noop.length + 2 * j)
  ]);
  return st;
}
function applyDilation(st, dilation) {
  if (dilation.every((s) => s === 1))
    return st;
  const s_ = dilation;
  const [a, b, ...k_] = st.shape;
  st = st.reshape([
    a,
    b,
    ...k_.flatMap((k) => [k, 1])
  ]);
  st = st.pad([
    [0, 0],
    [0, 0],
    ...s_.flatMap((s) => [[0, 0], [0, s - 1]])
  ]);
  st = st.reshape([
    a,
    b,
    ...k_.map((k, i) => k * s_[i])
  ]);
  st = st.shrink([
    [0, a],
    [0, b],
    ...k_.map((k, i) => [0, (k - 1) * s_[i] + 1])
  ]);
  return st;
}
function prepareConv(stX, stY, params) {
  const n = stX.shape.length - 2;
  stX = applyDilation(stX, params.lhsDilation);
  const ks = stY.shape.slice(2);
  stX = stX.padOrShrink([
    [0, 0],
    [0, 0],
    ...params.padding
  ]);
  stX = pool(stX, ks, params.strides, params.rhsDilation);
  stX = stX.moveaxis(1, n + 1).reshape([
    stX.shape[0],
    1,
    ...stX.shape.slice(2, n + 2),
    stX.shape[1] * prod(ks)
  ]);
  stY = stY.reshape([
    stY.shape[0],
    ...rep(n, 1),
    stY.shape[1] * prod(ks)
  ]);
  return [stX, stY];
}
var Primitive = /* @__PURE__ */ function(Primitive$1) {
  Primitive$1["Add"] = "add";
  Primitive$1["Mul"] = "mul";
  Primitive$1["Idiv"] = "idiv";
  Primitive$1["Neg"] = "neg";
  Primitive$1["Reciprocal"] = "reciprocal";
  Primitive$1["StopGradient"] = "stop_gradient";
  Primitive$1["Cast"] = "cast";
  Primitive$1["Bitcast"] = "bitcast";
  Primitive$1["RandomBits"] = "random_bits";
  Primitive$1["Sin"] = "sin";
  Primitive$1["Cos"] = "cos";
  Primitive$1["Asin"] = "asin";
  Primitive$1["Atan"] = "atan";
  Primitive$1["Exp"] = "exp";
  Primitive$1["Log"] = "log";
  Primitive$1["Erf"] = "erf";
  Primitive$1["Erfc"] = "erfc";
  Primitive$1["Sqrt"] = "sqrt";
  Primitive$1["Min"] = "min";
  Primitive$1["Max"] = "max";
  Primitive$1["Reduce"] = "reduce";
  Primitive$1["Dot"] = "dot";
  Primitive$1["Conv"] = "conv";
  Primitive$1["Pool"] = "pool";
  Primitive$1["PoolTranspose"] = "pool_transpose";
  Primitive$1["Compare"] = "compare";
  Primitive$1["Where"] = "where";
  Primitive$1["Transpose"] = "transpose";
  Primitive$1["Broadcast"] = "broadcast";
  Primitive$1["Reshape"] = "reshape";
  Primitive$1["Flip"] = "flip";
  Primitive$1["Shrink"] = "shrink";
  Primitive$1["Pad"] = "pad";
  Primitive$1["Gather"] = "gather";
  Primitive$1["JitCall"] = "jit_call";
  return Primitive$1;
}({});
var CompareOp = /* @__PURE__ */ function(CompareOp$1) {
  CompareOp$1["Less"] = "less";
  CompareOp$1["Equal"] = "equal";
  CompareOp$1["NotEqual"] = "not_equal";
  CompareOp$1["LessEqual"] = "less_equal";
  return CompareOp$1;
}({});
function add$1(x, y) {
  return bind1(Primitive.Add, [x, y]);
}
function mul(x, y) {
  return bind1(Primitive.Mul, [x, y]);
}
function idiv(x, y) {
  return bind1(Primitive.Idiv, [x, y]);
}
function neg(x) {
  return bind1(Primitive.Neg, [x]);
}
function reciprocal$1(x) {
  return bind1(Primitive.Reciprocal, [x]);
}
function stopGradient(x) {
  return bind1(Primitive.StopGradient, [x]);
}
function cast(x, dtype) {
  return bind1(Primitive.Cast, [x], { dtype });
}
function bitcast(x, dtype) {
  return bind1(Primitive.Bitcast, [x], { dtype });
}
function randomBits(k0, k1, shape$1, mode = "xor") {
  return bind1(Primitive.RandomBits, [k0, k1], {
    shape: shape$1,
    mode
  });
}
function sin$1(x) {
  return bind1(Primitive.Sin, [x]);
}
function cos$1(x) {
  return bind1(Primitive.Cos, [x]);
}
function asin$1(x) {
  return bind1(Primitive.Asin, [x]);
}
function atan$1(x) {
  return bind1(Primitive.Atan, [x]);
}
function exp$1(x) {
  return bind1(Primitive.Exp, [x]);
}
function log$1(x) {
  return bind1(Primitive.Log, [x]);
}
function erf$1(x) {
  return bind1(Primitive.Erf, [x]);
}
function erfc$1(x) {
  return bind1(Primitive.Erfc, [x]);
}
function sqrt$1(x) {
  return bind1(Primitive.Sqrt, [x]);
}
function min$1(x, y) {
  return bind1(Primitive.Min, [x, y]);
}
function max$1(x, y) {
  return bind1(Primitive.Max, [x, y]);
}
function reduce(x, op, axis = null, opts) {
  if (!AluGroup.Reduce.has(op))
    throw new TypeError(`Invalid reduce operation: ${op}`);
  axis = normalizeAxis(axis, ndim$1(x));
  const originalShape = getShape(x);
  let result = bind1(Primitive.Reduce, [x], {
    op,
    axis
  });
  if (opts?.keepdims)
    result = result.reshape(originalShape.map((dim, i) => axis.includes(i) ? 1 : dim));
  return result;
}
function dot$1(x, y) {
  return bind1(Primitive.Dot, [x, y]);
}
function conv(x, y, params = {}) {
  if (x.ndim !== y.ndim)
    throw new Error(`conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`);
  const n = x.ndim - 2;
  if (n < 0)
    throw new Error("conv() requires at least 2D inputs");
  return bind1(Primitive.Conv, [x, y], {
    strides: params.strides ?? rep(n, 1),
    padding: params.padding ?? rep(n, [0, 0]),
    lhsDilation: params.lhsDilation ?? rep(n, 1),
    rhsDilation: params.rhsDilation ?? rep(n, 1)
  });
}
function compare(x, y, op) {
  return bind1(Primitive.Compare, [x, y], { op });
}
function greater$1(x, y) {
  return compare(y, x, CompareOp.Less);
}
function less$1(x, y) {
  return compare(x, y, CompareOp.Less);
}
function equal$1(x, y) {
  return compare(x, y, CompareOp.Equal);
}
function notEqual$1(x, y) {
  return compare(x, y, CompareOp.NotEqual);
}
function greaterEqual$1(x, y) {
  return compare(y, x, CompareOp.LessEqual);
}
function lessEqual$1(x, y) {
  return compare(x, y, CompareOp.LessEqual);
}
function where$1(cond, x, y) {
  return bind1(Primitive.Where, [
    cond,
    x,
    y
  ]);
}
function transpose$1(x, perm) {
  perm = perm ? perm.map((a) => checkAxis(a, ndim$1(x))) : range(ndim$1(x)).reverse();
  if (!isPermutation(perm, ndim$1(x)))
    throw new Error(`Invalid transpose permutation for ${ndim$1(x)} axes: ${JSON.stringify(perm)}`);
  return bind1(Primitive.Transpose, [x], { perm });
}
function broadcast(x, shape$1, axis) {
  axis = normalizeAxis(axis, shape$1.length);
  return bind1(Primitive.Broadcast, [x], {
    shape: shape$1,
    axis
  });
}
function reshape$1(x, shape$1) {
  if (typeof shape$1 === "number")
    shape$1 = [shape$1];
  const originalShape = getShape(x);
  const autoIdx = shape$1.indexOf(-1);
  if (autoIdx !== -1) {
    const remaining = prod(originalShape) / -prod(shape$1);
    if (!Number.isInteger(remaining) || remaining < 0)
      throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
    shape$1 = shape$1.toSpliced(autoIdx, 1, remaining);
  }
  if (prod(originalShape) !== prod(shape$1))
    throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
  return bind1(Primitive.Reshape, [x], { shape: shape$1 });
}
function flip$1(x, axis) {
  axis = normalizeAxis(axis, ndim$1(x));
  return bind1(Primitive.Flip, [x], { axis });
}
function shrink(x, slice) {
  const shape$1 = getShape(x);
  if (!Array.isArray(slice) || !slice.every(isNumberPair))
    throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
  if (slice.length !== shape$1.length)
    throw new Error(`Invalid shrink(): expected ${shape$1.length} axes, got ${slice.length}`);
  for (let i = 0;i < shape$1.length; i++) {
    const [start, end] = slice[i];
    if (start > end || start < 0 || end > shape$1[i])
      throw new Error(`Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape$1[i]}`);
  }
  return bind1(Primitive.Shrink, [x], { slice });
}
function pad$1(x, width) {
  const nd = ndim$1(x);
  if (typeof width === "number")
    width = [[width, width]];
  else if (isNumberPair(width))
    width = [width];
  else if (!Array.isArray(width) || !width.every(isNumberPair))
    throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
  if (width.length === 1) {
    const [w0, w1] = width[0];
    width = rep(nd, () => [w0, w1]);
  } else if (width.length !== nd)
    throw new Error(`Invalid pad(): expected ${nd} axes, got ${width.length}`);
  return bind1(Primitive.Pad, [x], { width });
}
function gather(x, indices, axis, outDim) {
  if (indices.length === 0)
    throw new Error("gather() requires at least one index");
  if (!Array.isArray(axis) || axis.length !== indices.length)
    throw new Error(`Invalid gather() axis: expected ${indices.length} axes, got ${JSON.stringify(axis)}`);
  axis = axis.map((a) => checkAxis(a, ndim$1(x)));
  if (new Set(axis).size !== axis.length)
    throw new Error(`Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`);
  outDim = checkAxis(outDim, ndim$1(x) - axis.length + 1);
  return bind1(Primitive.Gather, [x, ...indices], {
    axis,
    outDim
  });
}
function bind1(prim, args, params = {}) {
  const [results] = bind(prim, args, params);
  return results;
}
var traceStack = [];
var dynamicTrace = null;
function newMain(traceType, globalData = null) {
  const level = traceStack.length;
  const main = {
    level,
    traceType,
    globalData
  };
  traceStack.push(main);
  return Object.assign(main, { [Symbol.dispose]() {
    traceStack.pop();
  } });
}
function newDynamic(main) {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main;
  return { [Symbol.dispose]() {
    dynamicTrace = prevDynamicTrace;
  } };
}
var Trace = class {
  constructor(main) {
    this.main = main;
  }
};
function promoteAvals(a, b) {
  const shape$1 = generalBroadcast(a.shape, b.shape);
  const weakType = a.weakType && b.weakType;
  let dtype;
  if (a.weakType === b.weakType)
    dtype = promoteTypes(a.dtype, b.dtype);
  else if (a.weakType)
    dtype = promoteTypes(b.dtype, DType.Uint32);
  else
    dtype = promoteTypes(a.dtype, DType.Uint32);
  return new ShapedArray(shape$1, dtype, weakType);
}
var Tracer = class Tracer2 {
  _trace;
  constructor(trace) {
    this._trace = trace;
  }
  get shape() {
    return this.aval.shape;
  }
  get size() {
    return prod(this.shape);
  }
  get dtype() {
    return this.aval.dtype;
  }
  get weakType() {
    return this.aval.weakType;
  }
  get ndim() {
    return this.shape.length;
  }
  fullLower() {
    return this;
  }
  neg() {
    return neg(this);
  }
  add(other) {
    return add$1(this, other);
  }
  mul(other) {
    return mul(this, other);
  }
  greater(other) {
    return greater$1(this, other);
  }
  less(other) {
    return less$1(this, other);
  }
  equal(other) {
    return equal$1(this, other);
  }
  notEqual(other) {
    return notEqual$1(this, other);
  }
  greaterEqual(other) {
    return greaterEqual$1(this, other);
  }
  lessEqual(other) {
    return lessEqual$1(this, other);
  }
  sum(axis = null, opts) {
    return reduce(this, AluOp.Add, axis, opts);
  }
  prod(axis = null, opts) {
    return reduce(this, AluOp.Mul, axis, opts);
  }
  mean(axis = null, opts) {
    axis = normalizeAxis(axis, this.ndim);
    const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
    if (n === 0)
      throw new Error("mean: cannot compute mean over zero-length axis");
    const result = reduce(this, AluOp.Add, axis, opts);
    return result.mul(1 / n);
  }
  transpose(perm) {
    return transpose$1(this, perm);
  }
  reshape(shape$1) {
    return reshape$1(this, shape$1);
  }
  astype(dtype) {
    if (this.dtype === dtype)
      return this;
    return cast(this, dtype);
  }
  sub(other) {
    return this.add(neg(other));
  }
  div(other) {
    if (isFloatDtype(this.dtype))
      return this.mul(reciprocal$1(other));
    return idiv(this, other);
  }
  diagonal(offset = 0, axis1 = 0, axis2 = 1) {
    if (!Number.isInteger(offset))
      throw new TypeError(`offset must be an integer, got ${offset}`);
    if (offset < 0)
      return this.diagonal(-offset, axis2, axis1);
    axis1 = checkAxis(axis1, this.ndim);
    axis2 = checkAxis(axis2, this.ndim);
    if (axis1 === axis2)
      throw new Error("axis1 and axis2 must not be equal");
    if (offset >= this.shape[axis2])
      throw new Error("offset exceeds axis size");
    let ar = this;
    if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
      const perm = range(ar.ndim).filter((i) => i !== axis1 && i !== axis2).concat(axis1, axis2);
      ar = ar.transpose(perm);
    }
    const [n, m] = ar.shape.slice(-2);
    const diagSize = Math.min(n, m - offset);
    ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
    const npad = diagSize * (m + 1) - n * m;
    if (npad > 0)
      ar = pad$1(ar, [...rep(ar.ndim - 1, [0, 0]), [0, npad]]);
    else if (npad < 0)
      ar = shrink(ar, [...ar.shape.slice(0, -1), n * m + npad].map((x) => [0, x]));
    ar = ar.reshape([
      ...ar.shape.slice(0, -1),
      diagSize,
      m + 1
    ]);
    ar = shrink(ar, [...ar.shape.slice(0, -1).map((x) => [0, x]), [offset, offset + 1]]).reshape(ar.shape.slice(0, -1));
    return ar;
  }
  flatten() {
    return this.reshape(-1);
  }
  ravel() {
    return this.reshape(-1);
  }
  *[Symbol.iterator]() {
    if (this.ndim === 0)
      throw new Error("Cannot iterate over a scalar array");
    for (let i = 0;i < this.shape[0]; i++)
      yield this.ref.slice(i);
    this.dispose();
  }
  slice(...index) {
    const checkBounds = (n, i) => {
      if (i > n || i < -n)
        throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
      return i < 0 ? n + i : i;
    };
    const hasAdvancedIdx = index.some((value) => value instanceof Tracer2);
    const axesForGather = [];
    let outDim = -1;
    if (hasAdvancedIdx) {
      const advancedAxes = [];
      let currentAxisForGather = 0;
      for (let i = 0;i < index.length; i++) {
        const value = index[i];
        if (value instanceof Tracer2) {
          advancedAxes.push(i);
          axesForGather.push(currentAxisForGather++);
        } else if (typeof value === "number")
          advancedAxes.push(i);
        else
          currentAxisForGather++;
      }
      if (advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !== advancedAxes.length - 1)
        outDim = 0;
      else
        outDim = axesForGather[0];
    }
    const slice = [];
    const basicShape = [];
    let needsReshape = false;
    let axis = 0;
    for (const value of index)
      if (value === null) {
        basicShape.push(1);
        needsReshape = true;
      } else if (typeof value === "number") {
        if (axis >= this.shape.length)
          throw new RangeError("Too many indices");
        const i = checkBounds(this.shape[axis++], value);
        slice.push([i, i + 1]);
        needsReshape = true;
      } else if (Array.isArray(value)) {
        if (axis >= this.shape.length)
          throw new RangeError("Too many indices");
        const n = this.shape[axis++];
        if (value.length === 0) {
          basicShape.push(n);
          slice.push([0, n]);
        } else if (value.length === 1) {
          const i = checkBounds(n, value[0]);
          basicShape.push(n - i);
          slice.push([i, n]);
        } else if (value.length === 2) {
          const [i, j] = value.map((v) => checkBounds(n, v));
          if (i > j)
            throw new RangeError(`Slice start at ${i} > end at ${j}`);
          basicShape.push(j - i);
          slice.push([i, j]);
        }
      } else if (value instanceof Tracer2) {
        const n = this.shape[axis++];
        basicShape.push(n);
        slice.push([0, n]);
      } else
        throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
    while (axis < this.shape.length) {
      slice.push([0, this.shape[axis]]);
      basicShape.push(this.shape[axis++]);
    }
    let result = shrink(this, slice);
    result = needsReshape ? reshape$1(result, basicShape) : result;
    if (hasAdvancedIdx)
      result = gather(result, index.filter((a) => a instanceof Tracer2), axesForGather, outDim);
    return result;
  }
};
function ndim$1(x) {
  if (x instanceof Tracer)
    return x.shape.length;
  else
    return 0;
}
function getShape(x) {
  return x instanceof Tracer ? x.shape : [];
}
var ShapedArray = class ShapedArray2 {
  constructor(shape$1, dtype, weakType) {
    this.shape = shape$1;
    this.dtype = dtype;
    this.weakType = weakType;
  }
  static fromAval(aval) {
    return new ShapedArray2(aval.shape, aval.dtype, aval.weakType);
  }
  get ndim() {
    return this.shape.length;
  }
  toString() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }
  equals(other) {
    return this === other || this.constructor === other.constructor && this.ndim === other.ndim && this.shape.every((d, i) => d === other.shape[i]);
  }
};
function getAval(x) {
  if (x instanceof Tracer)
    return x.aval;
  else if (typeof x === "boolean" || typeof x === "number")
    return new ShapedArray([], typeof x === "boolean" ? DType.Bool : DType.Float32, typeof x === "boolean" ? false : true);
  else
    throw new TypeError(`Unknown value: ${x}`);
}
function bind(prim, args, params = {}) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  if (DEBUG >= 5)
    console.info(`processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`);
  return outs.map((out) => out.fullLower());
}
function findTopTrace(xs) {
  let topMain = traceStack[0];
  for (const x of xs)
    if (x instanceof Tracer && x._trace.main.level > topMain.level)
      topMain = x._trace.main;
  if (dynamicTrace && dynamicTrace.level > topMain.level)
    topMain = dynamicTrace;
  return new topMain.traceType(topMain);
}
function fullRaise(trace, val) {
  if (!(val instanceof Tracer))
    return trace.pure(val);
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main))
    return val;
  else if (val._trace.main.level < level)
    return trace.lift(val);
  else if (val._trace.main.level > level)
    throw new Error(`Can't lift Tracer level ${val._trace.main.level} to level ${level}`);
  else
    throw new Error(`Different traces at same level: ${val._trace.constructor}, ${trace.constructor}.`);
}
var TreeMismatchError = class extends TypeError {
  constructor(where$2, left, right) {
    super(`Mismatched tree structures in ${where$2}: ${left} != ${right}`);
  }
};
function flattenFun(f, inTree) {
  const store = { value: undefined };
  const flatFun = (...argsFlat) => {
    const pytreeArgs = unflatten(inTree, argsFlat);
    const out = f(...pytreeArgs);
    const [outFlat, outTree] = flatten(out);
    store.value = outTree;
    return outFlat;
  };
  return [flatFun, store];
}
var UseAfterFreeError = class extends ReferenceError {
  constructor(tracer) {
    super(`Referenced tracer ${tracer.toString()} freed, please use .ref move semantics`);
  }
};
var JitProgram = class {
  constructor(backend, steps, inputs, outputs) {
    this.backend = backend;
    this.steps = steps;
    this.inputs = inputs;
    this.outputs = outputs;
  }
  pprint() {
    const steps = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const inputsNice = step.inputs.map((id, i) => `${i}: %${id}`).join(", ");
          const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
          return PPrint.pp(`execute (${inputsNice}) -> ${outputsNice}, kernel`).concat(step.kernel.pprint().indent(2));
        }
        case "const":
          return PPrint.pp(`%${step.output} = const <Slot ${step.slot}>`);
        case "malloc":
          return PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
      }
    });
    const display = PPrint.prototype.concat(PPrint.pp(`device = ${this.backend.type}`), PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"), PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"), PPrint.pp("steps ="), PPrint.prototype.concat(...steps).indent(2));
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }
  toString() {
    return this.pprint().toString();
  }
  execute(inputs) {
    const scope = /* @__PURE__ */ new Map;
    if (inputs.length !== this.inputs.length)
      throw new TypeError(`Expected ${this.inputs.length} inputs, got ${inputs.length}`);
    for (const [i, id] of this.inputs.entries())
      scope.set(id, inputs[i]);
    const pending = [];
    for (const step of this.steps)
      switch (step.type) {
        case "execute": {
          const inputs$1 = step.inputs.map((id) => scope.get(id));
          const outputs = step.outputs.map((id) => scope.get(id));
          if (inputs$1.some((s) => s === undefined) || outputs.some((s) => s === undefined))
            throw new Error(`internal: JitProgram scope undefined`);
          pending.push(new PendingExecute(this.backend, step.kernel, inputs$1, outputs));
          break;
        }
        case "const":
          scope.set(step.output, step.slot);
          break;
        case "malloc": {
          const slot = this.backend.malloc(step.size);
          scope.set(step.output, slot);
          break;
        }
        case "incref": {
          const slot = scope.get(step.input);
          this.backend.incRef(slot);
          break;
        }
        case "free": {
          const slot = scope.get(step.input);
          this.backend.decRef(slot);
          scope.delete(step.input);
          break;
        }
        default:
      }
    return {
      outputs: this.outputs.map((id) => scope.get(id)),
      pending
    };
  }
};
var JitProgramBuilder = class {
  backend;
  #nextId;
  steps;
  constructor(backend, nargs) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }
  pushConst(slot) {
    const id = this.#nextId++;
    this.steps.push({
      type: "const",
      slot,
      output: id
    });
    return id;
  }
  pushLit(lit) {
    const kernel = new Kernel(0, prod(lit.aval.shape), AluExp.const(lit.dtype, lit.value));
    return this.pushKernel(kernel, []);
  }
  pushKernel(kernel, inputs) {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size: kernel.bytes,
      output: id
    });
    this.steps.push({
      type: "execute",
      kernel,
      inputs,
      outputs: [id]
    });
    return id;
  }
  pushIncref(id) {
    this.steps.push({
      type: "incref",
      input: id
    });
  }
  insertFreeSteps(outputIds) {
    const ids = this.steps.filter((s) => s.type === "malloc").map((s) => s.output);
    for (const id of ids) {
      if (outputIds.includes(id))
        continue;
      const lastUsage = this.steps.findLastIndex((s) => s.type === "execute" && (s.outputs.includes(id) || s.inputs.includes(id)) || s.type === "malloc" && s.output === id);
      this.steps.splice(lastUsage + 1, 0, {
        type: "free",
        input: id
      });
    }
  }
  pushFree(id) {
    this.steps.push({
      type: "free",
      input: id
    });
  }
};
var jitCompileCache = /* @__PURE__ */ new Map;
function jitCompile(backend, jaxpr, consts) {
  if (jaxpr.inBinders.length < consts.length)
    throw new TypeError(`Jaxpr has ${jaxpr.inBinders.length} inputs, but ${consts.length} consts were provided`);
  for (let i = 0;i < consts.length; i++)
    if (consts[i].device !== backend.type)
      throw new TypeError(`Const ${i} has device ${consts[i].device}, but expected ${backend.type}`);
  const cacheKey = backend.type + FpHash.hash(jaxpr, ...consts.map((c) => c.id));
  const cached = jitCompileCache.get(cacheKey);
  if (cached)
    return cached;
  if (DEBUG >= 1)
    console.info(`=========== JIT Compile ===========
` + jaxpr.toString());
  jaxpr = jaxpr.flatten().simplify();
  const nargs = jaxpr.inBinders.length - consts.length;
  const builder = new JitProgramBuilder(backend, nargs);
  const blackNodes = splitGraphDataflow(backend, jaxpr);
  const ctx = /* @__PURE__ */ new Map;
  for (let i = 0;i < consts.length; i++) {
    const v = jaxpr.inBinders[i];
    const slot = consts[i]._realizeSource();
    ctx.set(v, {
      type: "imm",
      arg: builder.pushConst(slot)
    });
  }
  for (let i = 0;i < nargs; i++) {
    const v = jaxpr.inBinders[consts.length + i];
    ctx.set(v, {
      type: "imm",
      arg: i
    });
  }
  for (let i = 0;i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    const inputExps = [];
    const inputAvals = [];
    const inputArgs = [];
    for (const input of eqn.inputs)
      if (input instanceof Var) {
        const jitValue = ctx.get(input);
        if (jitValue.type === "exp") {
          const gidMap = /* @__PURE__ */ new Map;
          for (const [gid, jitId] of jitValue.args.entries()) {
            let newGid = inputArgs.indexOf(jitId);
            if (newGid === -1) {
              newGid = inputArgs.length;
              inputArgs.push(jitId);
            }
            gidMap.set(gid, newGid);
          }
          inputExps.push(jitValue.exp.reindexGids(gidMap));
        } else if (jitValue.type === "imm") {
          let gid = inputArgs.indexOf(jitValue.arg);
          if (gid === -1) {
            gid = inputArgs.length;
            inputArgs.push(jitValue.arg);
          }
          const st = ShapeTracker.fromShape(input.aval.shape);
          const indices = unravelAlu(st.shape, AluVar.gidx);
          inputExps.push(AluExp.globalView(input.aval.dtype, gid, st, indices));
        }
        inputAvals.push(input.aval);
      } else if (input instanceof Lit) {
        inputExps.push(AluExp.const(input.dtype, input.value));
        inputAvals.push(input.aval);
      } else
        throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
    const nargs$1 = inputArgs.length;
    const rule = jitRules[eqn.primitive];
    if (!rule)
      throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
    const kernel = rule(nargs$1, inputExps, inputAvals, eqn.params);
    const outVar = eqn.outBinders[0];
    if (kernel.reduction || blackNodes.has(outVar)) {
      const outId = builder.pushKernel(kernel, inputArgs);
      ctx.set(outVar, {
        type: "imm",
        arg: outId
      });
    } else
      ctx.set(outVar, {
        type: "exp",
        exp: kernel.exp,
        args: inputArgs
      });
  }
  const outputIds = [];
  for (const out of jaxpr.outs)
    if (out instanceof Var) {
      const jitValue = ctx.get(out);
      if (jitValue.type !== "imm")
        throw new Error("internal: Expected imm, since outs are black nodes");
      outputIds.push(jitValue.arg);
    } else if (out instanceof Lit)
      outputIds.push(builder.pushLit(out));
  const outputNeedsRef = new Set([...range(nargs), ...builder.steps.filter((s) => s.type === "const").map((s) => s.output)]);
  for (const outputId of outputIds)
    if (outputNeedsRef.has(outputId))
      builder.pushIncref(outputId);
    else
      outputNeedsRef.add(outputId);
  builder.insertFreeSteps(outputIds);
  const jp = new JitProgram(backend, builder.steps, range(0, nargs), outputIds);
  if (DEBUG >= 4)
    console.info(jp.toString());
  jitCompileCache.set(cacheKey, jp);
  return jp;
}
function reshapeViews(exp$2, mapping, reduceAxis = false) {
  return exp$2.rewrite((exp$3) => {
    if (exp$3.op === AluOp.GlobalView) {
      const [gid, st] = exp$3.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices = reduceAxis ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(AluVar.ridx) : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp$3.dtype, gid, newSt, indices);
      }
    } else if (exp$3.op === AluOp.GlobalIndex)
      throw new Error("internal: reshapeViews() called with GlobalIndex op");
  });
}
function broadcastedJit(fn, opts) {
  return (nargs, exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);
    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length)
      newDtype = avals.filter((_, i) => !skipCastIdx.includes(i)).reduce(promoteAvals).dtype;
    exps = exps.map((exp$3, i) => {
      exp$3 = reshapeViews(exp$3, (st) => {
        if (!deepEqual(st.shape, newShape))
          return st.broadcast(newShape, range(newShape.length - st.shape.length));
      });
      if (exp$3.dtype !== newDtype && !skipCastIdx.includes(i))
        exp$3 = AluExp.cast(newDtype, exp$3);
      return exp$3;
    });
    const exp$2 = fn(exps, params);
    return new Kernel(nargs, prod(newShape), exp$2);
  };
}
function unopJit(fn) {
  return (nargs, [a], [as], params) => {
    return new Kernel(nargs, prod(as.shape), fn(a, params));
  };
}
function reshapeJit(fn) {
  return (nargs, [a], [as], params) => {
    a = reshapeViews(a, (st) => fn(st, params));
    const newShape = fn(ShapeTracker.fromShape(as.shape), params).shape;
    return new Kernel(nargs, prod(newShape), a);
  };
}
var jitRules = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.StopGradient]: unopJit((a) => a),
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.RandomBits]: (nargs, keys, keyShapes, { shape: shape$1, mode }) => {
    const mapping = (st) => {
      if (!deepEqual(st.shape, shape$1))
        return st.broadcast(shape$1, range(shape$1.length - st.shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.cast(DType.Uint32, AluVar.gidx);
    const exp$2 = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return new Kernel(nargs, prod(shape$1), exp$2);
  },
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Erf]: unopJit(AluExp.erf),
  [Primitive.Erfc]: unopJit(AluExp.erfc),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.Reduce](nargs, [a], [as], { op, axis }) {
    const keptAxes = [];
    const shiftedAxes = [];
    const newShape = [];
    for (let i = 0;i < as.shape.length; i++)
      if (axis.includes(i))
        shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(as.shape[i]);
      }
    const size$1 = prod(newShape);
    const reductionSize = prod(shiftedAxes.map((ax) => as.shape[ax]));
    newShape.push(reductionSize);
    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSize);
    return new Kernel(nargs, size$1, a, reduction);
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) => pool(st, window, strides)),
  [Primitive.PoolTranspose](nargs, [a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(ShapeTracker.fromShape(as.shape), inShape, window, strides);
    const size$1 = prod(inShape);
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]);
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(a.dtype, AluOp.Add, stX.shape[stX.shape.length - 1]);
    return new Kernel(nargs, size$1, a, reduction);
  },
  [Primitive.Dot](nargs, [a, b], [as, bs]) {
    const k1 = jitRules[Primitive.Mul](nargs, [a, b], [as, bs], {});
    const c = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce](nargs, [c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1]
    });
  },
  [Primitive.Conv](nargs, [a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(ShapeTracker.fromShape(as.shape), ShapeTracker.fromShape(bs.shape), params);
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot](nargs, [a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(([cond, a, b]) => AluExp.where(cond, a, b), { skipCastIdx: [0] }),
  [Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
  [Primitive.Broadcast]: reshapeJit((st, { shape: shape$1, axis }) => st.broadcast(shape$1, axis)),
  [Primitive.Reshape]: reshapeJit((st, { shape: shape$1 }) => st.reshape(shape$1)),
  [Primitive.Flip]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis)
      arg[ax] = true;
    return st.flip(arg);
  }),
  [Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
  [Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
  [Primitive.Gather](nargs, [x, ...indices], [xs, ...indicesShapes], { axis, outDim }) {
    const axisSet = new Set(axis);
    const indexShape = indicesShapes.map((c) => c.shape).reduce(generalBroadcast);
    const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    idxNonaxis.splice(outDim, indexShape.length);
    const src = [...idxNonaxis];
    for (let i = 0;i < xs.shape.length; i++)
      if (axisSet.has(i))
        src.splice(i, 0, null);
    for (const [i, iexp] of indices.entries())
      src[axis[i]] = AluExp.cast(DType.Int32, reshapeViews(iexp, (st) => st.broadcast(finalShape, [...range(outDim + indexShape.length - st.shape.length), ...range(outDim + indexShape.length, finalShape.length)])));
    const [index, valid] = ShapeTracker.fromShape(xs.shape).toAluExp(src);
    if (!valid.resolve())
      throw new Error("internal: expected full validity mask in Gather");
    return new Kernel(nargs, prod(finalShape), x.substitute({ gidx: index }));
  },
  [Primitive.JitCall]() {
    throw new Error("internal: JitCall should have been flattened before JIT compilation");
  }
};
function splitGraphDataflow(backend, jaxpr) {
  const varToEqn = /* @__PURE__ */ new Map;
  for (let i = 0;i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders)
      if (v instanceof Var)
        varToEqn.set(v, i);
  }
  const blackNodes = /* @__PURE__ */ new Set;
  const p1NextBlack = /* @__PURE__ */ new Map;
  for (const v of jaxpr.outs)
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose
  ];
  const heterogeneousViewPrimitives = [Primitive.Gather, Primitive.RandomBits];
  for (let i = jaxpr.eqns.length - 1;i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive) || heterogeneousViewPrimitives.includes(eqn.primitive) || eqn.outBinders.some((v) => blackNodes.has(v))) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = /* @__PURE__ */ new Set;
    for (let j = i + 1;j < jaxpr.eqns.length; j++)
      for (const v of jaxpr.eqns[j].inputs)
        if (v instanceof Var && eqn.outBinders.includes(v))
          for (const o of jaxpr.eqns[j].outBinders) {
            const u = p1NextBlack.get(o);
            if (u)
              reach.add(u);
          }
    if (reach.size === 1) {
      const b = reach.values().next().value;
      for (const v of eqn.outBinders)
        p1NextBlack.set(v, b);
    } else if (reach.size > 1)
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
  }
  const p2Deps = /* @__PURE__ */ new Map;
  for (const v of jaxpr.inBinders)
    p2Deps.set(v, new Set([v]));
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    const deps = [];
    if (eqn.outBinders.some((v) => blackNodes.has(v)))
      continue;
    for (const input of eqn.inputs)
      if (input instanceof Var)
        if (blackNodes.has(input))
          deps.push(new Set([input]));
        else
          deps.push(p2Deps.get(input));
      else
        deps.push(/* @__PURE__ */ new Set);
    const depCounter = /* @__PURE__ */ new Map;
    for (const depSet of deps)
      for (const dep of depSet)
        depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
    if (depCounter.size > backend.maxArgs) {
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0;i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToEqn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i])
            if (depCounter.get(dep) === 1)
              uniqueDeps++;
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1)
        throw new Error(`internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`);
      const assocVar = eqn.inputs[assocInput];
      p2idx = varToEqn.get(assocVar);
      for (const out of jaxpr.eqns[p2idx].outBinders)
        blackNodes.add(out);
    } else {
      const s = new Set(depCounter.keys());
      for (const out of eqn.outBinders)
        p2Deps.set(out, s);
    }
  }
  return blackNodes;
}
var JsArray = globalThis.Array;
var inlineArrayLimit = 128;
var fudgeArray = pureArray;
var PendingExecute = class {
  prepared = null;
  submitted = false;
  #promise = null;
  #rc = 1;
  constructor(backend, kernel, inputs, outputs) {
    this.backend = backend;
    this.kernel = kernel;
    this.inputs = inputs;
    this.outputs = outputs;
    for (const slot of inputs)
      this.backend.incRef(slot);
    for (const slot of outputs)
      this.backend.incRef(slot);
  }
  updateRc(delta) {
    if (this.#rc <= 0)
      throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      for (const slot of this.inputs)
        this.backend.decRef(slot);
      for (const slot of this.outputs)
        this.backend.decRef(slot);
    }
  }
  async prepare() {
    if (this.prepared)
      return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      this.prepared = await this.backend.prepare(this.kernel);
    })();
    await this.#promise;
  }
  prepareSync() {
    if (this.prepared)
      return;
    this.prepared = this.backend.prepareSync(this.kernel);
  }
  submit() {
    if (this.submitted)
      return;
    if (this.#rc <= 0)
      throw new Error("internal: PendingExecute used rc<=0");
    if (!this.prepared)
      throw new Error("internal: Not prepared yet");
    this.submitted = true;
    this.backend.dispatch(this.prepared, this.inputs, this.outputs);
    for (const slot of this.inputs)
      this.backend.decRef(slot);
    for (const slot of this.outputs)
      this.backend.decRef(slot);
  }
};
var Array$1 = class Array$12 extends Tracer {
  static #nextId = 1001;
  id;
  #dtype;
  #weakType;
  #source;
  #st;
  #backend;
  #committed;
  #rc;
  #pendingSet;
  constructor(args) {
    super(baseArrayTrace);
    this.id = Array$12.#nextId++;
    this.#dtype = args.dtype;
    this.#weakType = args.weakType;
    this.#source = args.source;
    this.#st = args.st;
    this.#backend = args.backend;
    this.#committed = args.committed;
    this.#rc = 1;
    this.#pendingSet = new Set(args.pending);
    if (this.#pendingSet.size === 0)
      this.#pendingSet = null;
    else if (this.#source instanceof AluExp)
      throw new Error("internal: AluExp source cannot have pending executes");
  }
  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
  }
  toString() {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }
  get device() {
    return this.#backend.type;
  }
  #check() {
    if (this.#rc <= 0)
      throw new UseAfterFreeError(this);
  }
  #newArrayFrom(args) {
    return new Array$12({
      source: args.source ?? this.#source,
      st: args.st ?? this.#st,
      dtype: args.dtype ?? this.#dtype,
      weakType: this.#weakType,
      backend: args.backend ?? this.#backend,
      committed: args.committed ?? this.#committed,
      pending: args.pending ?? this.#pending ?? undefined
    });
  }
  get ref() {
    this.#check();
    this.#rc++;
    return this;
  }
  dispose() {
    this.#check();
    if (--this.#rc === 0) {
      for (const exe of this.#pending)
        exe.updateRc(-1);
      if (typeof this.#source === "number")
        this.#backend.decRef(this.#source);
    }
  }
  get #pending() {
    if (!this.#pendingSet)
      return [];
    for (const p of this.#pendingSet)
      if (p.submitted)
        this.#pendingSet.delete(p);
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
      return [];
    } else
      return [...this.#pendingSet];
  }
  [Symbol.toPrimitive]() {
    if (this.ndim === 0)
      return this.dataSync()[0];
    else
      throw new Error(`Cannot convert non-scalar array to primitive: ${this.toString()}`);
  }
  #reshape(st) {
    this.#check();
    const pending = this.#pending;
    for (const exe of pending)
      exe.updateRc(1);
    if (typeof this.#source === "number")
      this.#backend.incRef(this.#source);
    const ar = this.#newArrayFrom({
      st,
      pending
    });
    this.dispose();
    return ar;
  }
  #gather(indices, axis, outDim) {
    this.#check();
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length)
      throw new TypeError("Gather axis must not have duplicates");
    if (indices.some((a) => a.#committed && a.#backend !== this.#backend))
      throw new TypeError(`Gather indices must have the same backend: ${this.#backend.type}`);
    indices = indices.map((ar) => ar._putSync(this.#backend));
    indices = Array$12.#broadcastArrays(indices);
    const indexShape = indices[0].shape;
    const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const idxAxis = idxNonaxis.splice(outDim, indexShape.length);
    const inputs = [];
    const src = [...idxNonaxis];
    for (let i = 0;i < this.shape.length; i++)
      if (axisSet.has(i))
        src.splice(i, 0, null);
    for (const [i, ar] of indices.entries())
      if (ar.#source instanceof AluExp)
        src[axis[i]] = AluExp.cast(DType.Int32, accessorAluExp(ar.#source, ar.#st, idxAxis));
      else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        src[axis[i]] = AluExp.cast(DType.Int32, AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis));
      }
    let exp$2;
    if (this.#source instanceof AluExp)
      exp$2 = accessorAluExp(this.#source, this.#st, src);
    else {
      let gid = inputs.indexOf(this.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(this.#source);
      }
      exp$2 = accessorGlobal(this.#dtype, gid, this.#st, src);
    }
    const kernel = new Kernel(inputs.length, prod(finalShape), exp$2);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
    for (const exe of pending)
      exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
    this.dispose();
    for (const ar of indices)
      ar.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(finalShape),
      pending
    });
  }
  #moveAxesDown(axis) {
    this.#check();
    if (axis.length === 0)
      return this.reshape(this.shape.concat(1));
    const newShape = [];
    const keptAxes = [];
    const shiftedAxes = [];
    for (let i = 0;i < this.#st.shape.length; i++)
      if (axis.includes(i))
        shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(this.#st.shape[i]);
      }
    newShape.push(-1);
    return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
  }
  #transpose(perm) {
    this.#check();
    if (!isPermutation(perm, this.ndim))
      throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    return this.#reshape(this.#st.permute(perm));
  }
  #unary(op, dtypeOutput) {
    const weakType = !dtypeOutput && this.#weakType;
    dtypeOutput ??= this.#dtype;
    this.#check();
    if (this.#source instanceof AluExp) {
      const exp$3 = new AluExp(op, dtypeOutput, [this.#source]);
      this.dispose();
      return this.#newArrayFrom({
        source: exp$3.simplify(),
        dtype: dtypeOutput,
        weakType
      });
    }
    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp$2 = new AluExp(op, dtypeOutput, [AluExp.globalView(this.#dtype, 0, this.#st, indices)]);
    const kernel = new Kernel(1, this.#st.size, exp$2);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending)
      exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, [this.#source], [output]));
    this.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(this.shape),
      dtype: dtypeOutput,
      weakType,
      pending
    });
  }
  #binary(op, other) {
    const custom = (src) => new AluExp(op, src[0].dtype, src);
    return Array$12.#naryCustom(op, custom, [this, other]);
  }
  static #naryCustom(name, custom, arrays, { dtypeOverride, strongTypeOutput, reduceAxis } = {}) {
    const n = arrays.length;
    if (n === 0)
      throw new TypeError(`No inputs for ${name}`);
    for (const ar of arrays)
      ar.#check();
    let castDtype;
    let castWeakType = true;
    for (let i = 0;i < n; i++)
      if (dtypeOverride?.[i]) {
        if (arrays[i].#dtype !== dtypeOverride[i])
          throw new TypeError(`Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`);
      } else if (castDtype === undefined) {
        castDtype = arrays[i].#dtype;
        castWeakType = arrays[i].#weakType;
      } else
        ({ dtype: castDtype, weakType: castWeakType } = promoteAvals(new ShapedArray([], castDtype, castWeakType), new ShapedArray([], arrays[i].#dtype, arrays[i].#weakType)));
    const weakType = castWeakType && !strongTypeOutput;
    const { backend, committed } = Array$12.#computeBackend(name, arrays);
    arrays = arrays.map((ar) => ar._putSync(backend));
    arrays = Array$12.#broadcastArrays(arrays);
    const newShape = [...arrays[0].shape];
    if (arrays.every((ar) => ar.#source instanceof AluExp) && !reduceAxis) {
      const sources = arrays.map((ar, i) => {
        if (!dtypeOverride?.[i])
          return AluExp.cast(castDtype, ar.#source);
        else
          return ar.#source;
      });
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        const exp$4 = custom(sources);
        arrays.forEach((ar) => ar.dispose());
        return new Array$12({
          source: exp$4.simplify(),
          st: arrays[0].#st,
          dtype: exp$4.dtype,
          weakType,
          backend,
          committed
        });
      }
      const exp$3 = custom(arrays.map((ar, i) => {
        const src$1 = sources[i];
        if (ar.#st.contiguous)
          return src$1;
        return accessorAluExp(src$1, ar.#st, unravelAlu(newShape, AluVar.idx));
      }));
      const st = ShapeTracker.fromShape(newShape);
      arrays.forEach((ar) => ar.dispose());
      return new Array$12({
        source: exp$3.simplify(),
        st,
        dtype: exp$3.dtype,
        weakType,
        backend,
        committed
      });
    }
    let indices;
    if (!reduceAxis)
      indices = unravelAlu(newShape, AluVar.gidx);
    else {
      const contractedShape = newShape.slice(0, -1);
      indices = [...unravelAlu(contractedShape, AluVar.gidx), AluVar.ridx];
    }
    const inputs = [];
    const src = [];
    for (const [i, ar] of arrays.entries()) {
      let nextSrc;
      if (ar.#source instanceof AluExp)
        nextSrc = accessorAluExp(ar.#source, ar.#st, indices);
      else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        nextSrc = AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
      }
      if (!dtypeOverride?.[i])
        nextSrc = AluExp.cast(castDtype, nextSrc);
      src.push(nextSrc);
    }
    const exp$2 = custom(src);
    let re = undefined;
    if (reduceAxis) {
      const [axisSize] = newShape.splice(-1, 1);
      re = new Reduction(exp$2.dtype, AluOp.Add, axisSize);
    }
    const kernel = new Kernel(inputs.length, prod(newShape), exp$2, re);
    const output = backend.malloc(kernel.bytes);
    const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
    for (const exe of pending)
      exe.updateRc(1);
    pending.add(new PendingExecute(backend, kernel, inputs, [output]));
    arrays.forEach((ar) => ar.dispose());
    return new Array$12({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      dtype: kernel.dtype,
      weakType,
      backend,
      committed,
      pending
    });
  }
  #reduce(op) {
    const shape$1 = this.shape;
    const reduction = new Reduction(this.#dtype, op, shape$1[shape$1.length - 1]);
    const newShape = shape$1.slice(0, -1);
    const newSize = prod(newShape);
    const indices = [...unravelAlu(newShape, AluVar.gidx), AluVar.ridx];
    let exp$2;
    const inputs = [];
    if (this.#source instanceof AluExp)
      exp$2 = accessorAluExp(this.#source, this.#st, indices);
    else {
      inputs.push(this.#source);
      exp$2 = accessorGlobal(this.#dtype, 0, this.#st, indices);
    }
    const kernel = new Kernel(inputs.length, newSize, exp$2, reduction);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending)
      exe.updateRc(1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
    this.dispose();
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      pending
    });
  }
  #realize() {
    this.#check();
    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    if (this.#source instanceof AluExp) {
      const exp$2 = accessorAluExp(this.#source, this.#st, indices);
      const kernel = new Kernel(0, this.#st.size, exp$2);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(this.#backend, kernel, [], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = new Set([pendingItem]);
    } else {
      if (this.#st.contiguous)
        return;
      const exp$2 = accessorGlobal(this.#dtype, 0, this.#st, indices);
      const kernel = new Kernel(1, this.#st.size, exp$2);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(this.#backend, kernel, [this.#source], [output]);
      this.#backend.decRef(this.#source);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= /* @__PURE__ */ new Set;
      this.#pendingSet.add(pendingItem);
    }
  }
  #dataInline() {
    this.#check();
    if (!(this.#source instanceof AluExp))
      throw new Error("internal: #dataInline called on non-AluExp source");
    const ar = this.#newArrayFrom({ backend: getBackend("cpu") });
    this.dispose();
    return ar.dataSync();
  }
  static #broadcastArrays(arrays) {
    if (arrays.length === 0)
      throw new Error("Need at least one array to broadcast");
    if (arrays.length === 1)
      return arrays;
    const newShape = arrays.map((a) => a.shape).reduce(generalBroadcast);
    return arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape))
        return ar;
      return ar.#reshape(ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)));
    });
  }
  static #computeBackend(name, arrays) {
    const committed = arrays.filter((ar) => ar.#committed);
    if (committed.length > 0) {
      const backend = committed[0].#backend;
      for (const ar of committed)
        if (ar.#backend !== backend)
          throw new Error(`Device mismatch in ${name} between committed arrays on (${backend.type}, ${ar.#backend.type}), please move to the same device with devicePut()`);
      return {
        backend,
        committed: true
      };
    } else {
      const backend = arrays.length > 0 ? arrays[0].#backend : getBackend();
      return {
        backend,
        committed: false
      };
    }
  }
  async data() {
    if (this.#source instanceof AluExp && this.size < inlineArrayLimit && this.device !== "cpu")
      return this.#dataInline();
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending)
        p.submit();
    }
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = await this.#backend.read(this.#source, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }
  async blockUntilReady() {
    this.#check();
    if (this.#source instanceof AluExp)
      return this;
    const pending = this.#pending;
    if (pending) {
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending)
        p.submit();
    }
    await this.#backend.read(this.#source, 0, 0);
    return this;
  }
  dataSync() {
    if (this.#source instanceof AluExp && this.size < inlineArrayLimit && this.device !== "cpu")
      return this.#dataInline();
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = this.#backend.readSync(this.#source, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
  }
  item() {
    if (this.size !== 1)
      throw new Error(`item() can only be called on arrays of size 1`);
    return this.dataSync()[0];
  }
  static _implRules() {
    return {
      [Primitive.Add]([x, y]) {
        return [x.#binary(AluOp.Add, y)];
      },
      [Primitive.Mul]([x, y]) {
        return [x.#binary(AluOp.Mul, y)];
      },
      [Primitive.Idiv]([x, y]) {
        return [x.#binary(AluOp.Idiv, y)];
      },
      [Primitive.Neg]([x]) {
        return [zerosLike$1(x.ref).#binary(AluOp.Sub, x)];
      },
      [Primitive.Reciprocal]([x]) {
        return [x.#unary(AluOp.Reciprocal)];
      },
      [Primitive.StopGradient]([x]) {
        return [x];
      },
      [Primitive.Cast]([x], { dtype }) {
        return [x.#unary(AluOp.Cast, dtype)];
      },
      [Primitive.Bitcast]([x], { dtype }) {
        if (x.dtype === DType.Bool || dtype === DType.Bool)
          throw new TypeError("Bitcast to/from bool is not allowed");
        if (x.dtype === dtype)
          return [x];
        if (byteWidth(x.dtype) !== byteWidth(dtype))
          throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
        if (x.#source instanceof AluExp)
          return [x.#unary(AluOp.Bitcast, dtype)];
        else {
          x.#backend.incRef(x.#source);
          const pending = x.#pending;
          for (const exe of pending)
            exe.updateRc(1);
          const y = x.#newArrayFrom({
            dtype,
            weakType: false,
            pending
          });
          x.dispose();
          return [y];
        }
      },
      [Primitive.RandomBits]([k0, k1], { shape: shape$1, mode }) {
        const keyShape = generalBroadcast(k0.shape, k1.shape);
        if (!deepEqual(generalBroadcast(keyShape, shape$1), shape$1))
          throw new TypeError(`Keys of shapes ${k0.shape} and ${k1.shape} cannot be broadcast to shape ${shape$1}`);
        const c0 = zeros(shape$1, {
          dtype: DType.Uint32,
          device: k0.device
        });
        const c1 = arange(0, prod(shape$1), 1, {
          dtype: DType.Uint32,
          device: k0.device
        }).reshape(shape$1);
        const custom = ([k0$1, k1$1, c0$1, c1$1]) => AluExp.threefry2x32(k0$1, k1$1, c0$1, c1$1, mode);
        return [Array$12.#naryCustom("random_bits", custom, [
          k0,
          k1,
          c0,
          c1
        ])];
      },
      [Primitive.Sin]([x]) {
        return [x.#unary(AluOp.Sin)];
      },
      [Primitive.Cos]([x]) {
        return [x.#unary(AluOp.Cos)];
      },
      [Primitive.Asin]([x]) {
        return [x.#unary(AluOp.Asin)];
      },
      [Primitive.Atan]([x]) {
        return [x.#unary(AluOp.Atan)];
      },
      [Primitive.Exp]([x]) {
        return [x.#unary(AluOp.Exp)];
      },
      [Primitive.Log]([x]) {
        return [x.#unary(AluOp.Log)];
      },
      [Primitive.Erf]([x]) {
        return [x.#unary(AluOp.Erf)];
      },
      [Primitive.Erfc]([x]) {
        return [x.#unary(AluOp.Erfc)];
      },
      [Primitive.Sqrt]([x]) {
        return [x.#unary(AluOp.Sqrt)];
      },
      [Primitive.Min]([x, y]) {
        return [x.#binary(AluOp.Min, y)];
      },
      [Primitive.Max]([x, y]) {
        return [x.#binary(AluOp.Max, y)];
      },
      [Primitive.Reduce]([x], { op, axis }) {
        if (axis.length === 0)
          return [x];
        return [x.#moveAxesDown(axis).#reduce(op)];
      },
      [Primitive.Pool]([x], { window, strides }) {
        const st = pool(x.#st, window, strides);
        return [x.#reshape(st)];
      },
      [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
        const n = inShape.length;
        let st = poolTranspose(x.#st, inShape, window, strides);
        st = st.reshape([...st.shape.slice(0, n), prod(st.shape.slice(n))]);
        return [x.#reshape(st).#reduce(AluOp.Add)];
      },
      [Primitive.Dot]([x, y]) {
        return [Array$12.#naryCustom("dot", ([x$1, y$1]) => AluExp.mul(x$1, y$1), [x, y], { reduceAxis: true })];
      },
      [Primitive.Conv]([x, y], params) {
        checkConvShape(x.shape, y.shape, params);
        const [stX, stY] = prepareConv(x.#st, y.#st, params);
        return [Array$12.#naryCustom("conv", ([x$1, y$1]) => AluExp.mul(x$1, y$1), [x.#reshape(stX), y.#reshape(stY)], { reduceAxis: true })];
      },
      [Primitive.Compare]([x, y], { op }) {
        const custom = ([x$1, y$1]) => aluCompare(x$1, y$1, op);
        return [Array$12.#naryCustom("compare", custom, [x, y], { strongTypeOutput: true })];
      },
      [Primitive.Where]([cond, x, y]) {
        const custom = ([cond$1, x$1, y$1]) => AluExp.where(cond$1, x$1, y$1);
        return [Array$12.#naryCustom("where", custom, [
          cond,
          x,
          y
        ], { dtypeOverride: [DType.Bool] })];
      },
      [Primitive.Transpose]([x], { perm }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast]([x], { shape: shape$1, axis }) {
        return [x.#reshape(x.#st.broadcast(shape$1, axis))];
      },
      [Primitive.Reshape]([x], { shape: shape$1 }) {
        return [x.#reshape(x.#st.reshape(shape$1))];
      },
      [Primitive.Flip]([x], { axis }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis)
          arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      [Primitive.Shrink]([x], { slice }) {
        return [x.#reshape(x.#st.shrink(slice))];
      },
      [Primitive.Pad]([x], { width }) {
        return [x.#reshape(x.#st.pad(width))];
      },
      [Primitive.Gather]([x, ...indices], { axis, outDim }) {
        return [x.#gather(indices, axis, outDim)];
      },
      [Primitive.JitCall](args, { jaxpr, numConsts }) {
        if (jaxpr.inBinders.length !== args.length)
          throw new Error(`jit_call expects ${jaxpr.inBinders.length} args, got ${args.length}`);
        const { backend, committed } = Array$12.#computeBackend("jit_call", args);
        args = args.map((ar) => ar._putSync(backend));
        const consts = args.slice(0, numConsts);
        const tracers = args.slice(numConsts);
        const jp = jitCompile(backend, jaxpr, consts);
        const { outputs, pending } = jp.execute(tracers.map((x) => x._realizeSource()));
        for (const exe of pending)
          exe.updateRc(+outputs.length - 1);
        const prevPending = [...new Set(args.flatMap((x) => x.#pending))];
        for (const exe of prevPending)
          exe.updateRc(+outputs.length);
        pending.splice(0, 0, ...prevPending);
        args.forEach((x) => x.dispose());
        return outputs.map((source, i) => {
          return new Array$12({
            source,
            st: ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
            dtype: jaxpr.outs[i].aval.dtype,
            weakType: jaxpr.outs[i].aval.weakType,
            backend,
            committed,
            pending
          });
        });
      }
    };
  }
  _realizeSource() {
    this.#realize();
    return this.#source;
  }
  async _put(backend) {
    if (this.#backend === backend)
      return this;
    if (this.#source instanceof AluExp) {
      const ar = this.#newArrayFrom({
        backend,
        committed: true
      });
      this.dispose();
      return ar;
    } else {
      const data = await this.data();
      return arrayFromData(data, this.shape, {
        dtype: this.#dtype,
        device: backend.type
      }, this.#weakType);
    }
  }
  _putSync(backend) {
    if (this.#backend === backend)
      return this;
    if (this.#source instanceof AluExp) {
      const ar = this.#newArrayFrom({
        backend,
        committed: true
      });
      this.dispose();
      return ar;
    } else {
      const data = this.dataSync();
      return arrayFromData(data, this.shape, {
        dtype: this.#dtype,
        device: backend.type
      }, this.#weakType);
    }
  }
};
function array(values, { shape: shape$1, dtype, device } = {}) {
  if (values instanceof Tracer) {
    if (shape$1 && !deepEqual(values.shape, shape$1))
      values = values.reshape(shape$1);
    if (dtype && values.dtype !== dtype)
      values = values.astype(dtype);
    return values;
  } else if (ArrayBuffer.isView(values))
    return arrayFromData(values, shape$1 ?? [values.length], {
      dtype,
      device
    });
  else {
    if (!shape$1) {
      shape$1 = [];
      let cur = values;
      while (JsArray.isArray(cur)) {
        shape$1.push(cur.length);
        cur = cur[0];
      }
    }
    const size$1 = prod(shape$1);
    const flat = recursiveFlatten(values);
    if (flat.length !== size$1)
      throw new Error(`Jagged shape: ${JSON.stringify(shape$1)} vs ${flat.length}`);
    if (size$1 === 0)
      return zeros(shape$1, {
        dtype,
        device
      });
    if (size$1 === 1)
      return full(shape$1, flat[0], {
        dtype,
        device
      });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? DType.Bool;
      const data = new Int32Array(flat.map((x) => x ? 1 : 0));
      return arrayFromData(data, shape$1, {
        dtype,
        device
      });
    } else {
      const weakType = dtype == undefined;
      dtype = dtype ?? DType.Float32;
      const data = dtypedJsArray(dtype, flat);
      return arrayFromData(data, shape$1, {
        dtype,
        device
      }, weakType);
    }
  }
}
function arrayFromData(data, shape$1, { dtype, device }, weakType = false) {
  if (data instanceof Float32Array) {
    if (dtype && dtype !== DType.Float32)
      throw new Error("Float32Array must have float32 type");
    dtype ??= DType.Float32;
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== DType.Int32 && dtype !== DType.Bool)
      throw new Error("Int32Array must have int32 or bool type");
    dtype ??= DType.Int32;
  } else if (data instanceof Uint32Array) {
    if (dtype && dtype !== DType.Uint32)
      throw new Error("Uint32Array must have uint32 type");
    dtype ??= DType.Uint32;
  } else if (data instanceof Float16Array) {
    if (dtype && dtype !== DType.Float16)
      throw new Error("Float16Array must have float16 type");
    dtype ??= DType.Float16;
  } else if (data instanceof Float64Array) {
    if (dtype && dtype !== DType.Float64)
      throw new Error("Float64Array must have float64 type");
    dtype ??= DType.Float64;
  } else
    throw new Error("Unsupported data array type: " + data.constructor.name);
  if (data.length < inlineArrayLimit) {
    let allEqual = true;
    for (let i = 1;i < data.length; i++)
      if (data[i] !== data[0]) {
        allEqual = false;
        break;
      }
    if (allEqual) {
      const sa = new ShapedArray(shape$1, dtype, weakType);
      return fullInternal(sa, data[0], device);
    }
  }
  const backend = getBackend(device);
  const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const slot = backend.malloc(data.byteLength, buf);
  return new Array$1({
    source: slot,
    st: ShapeTracker.fromShape(shape$1),
    dtype,
    weakType,
    backend,
    committed: device != null
  });
}
function dataToJs(dtype, data, shape$1) {
  if (shape$1.length === 0)
    return dtype === DType.Bool ? Boolean(data[0]) : data[0];
  const [first, ...rest] = shape$1;
  const restSize = prod(rest);
  const ret = [];
  for (let i = 0;i < first; i++) {
    const subarray = data.slice(i * restSize, (i + 1) * restSize);
    ret.push(dataToJs(dtype, subarray, rest));
  }
  return ret;
}
function pureArray(x) {
  if (x instanceof Tracer)
    return x;
  else
    return array(x);
}
var EvalTrace = class extends Trace {
  pure = (x) => pureArray(x);
  lift = (x) => x;
  processPrimitive(primitive, tracers, params) {
    return implRules[primitive](tracers, params);
  }
};
var baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));
var implRules = Array$1._implRules();
function fullInternal(aval, fillValue, device) {
  return new Array$1({
    source: AluExp.const(aval.dtype, fillValue),
    st: ShapeTracker.fromShape(aval.shape),
    dtype: aval.dtype,
    weakType: aval.weakType,
    backend: getBackend(device),
    committed: device != null
  });
}
function zerosLike$1(val, dtype) {
  return fullLike(val, 0, dtype);
}
function onesLike$1(val, dtype) {
  return fullLike(val, 1, dtype);
}
function fullLike(val, fillValue, dtype) {
  const aval = getAval(val);
  if (val instanceof Tracer)
    val.dispose();
  if (fillValue instanceof Tracer)
    throw new Error("numpy.fullLike() with array argument not implemented yet");
  const sa = new ShapedArray(aval.shape, dtype ?? aval.dtype, aval.weakType);
  return fullInternal(sa, fillValue);
}
function zeros(shape$1, { dtype, device } = {}) {
  return full(shape$1, 0, {
    dtype,
    device
  });
}
function ones(shape$1, { dtype, device } = {}) {
  return full(shape$1, 1, {
    dtype,
    device
  });
}
function full(shape$1, fillValue, { dtype, device } = {}) {
  let weakType = dtype == undefined;
  if (typeof fillValue === "number")
    dtype = dtype ?? DType.Float32;
  else if (typeof fillValue === "boolean") {
    dtype = dtype ?? DType.Bool;
    weakType = false;
  } else if (fillValue instanceof Tracer)
    throw new Error("numpy.full() with array argument not implemented yet");
  else
    throw new TypeError(`Invalid type for full: ${fillValue}`);
  return fullInternal(new ShapedArray(shape$1, dtype, weakType), fillValue, device);
}
function eye(numRows, numCols, { dtype, device } = {}) {
  numCols = numCols ?? numRows;
  const weakType = dtype == undefined;
  dtype = dtype ?? DType.Float32;
  if (numCols < numRows) {
    const arr = eye(numCols, numRows, {
      dtype,
      device
    });
    return arr.transpose();
  }
  if (numRows === 0)
    return zeros([0, numCols], {
      dtype,
      device
    });
  const exp$2 = AluExp.cmplt(AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)), AluExp.i32(1));
  return new Array$1({
    source: AluExp.cast(dtype, exp$2),
    st: ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    weakType,
    backend: getBackend(device),
    committed: device != null
  });
}
function identity$1(n, { dtype, device } = {}) {
  return eye(n, n, {
    dtype,
    device
  });
}
function arange(start, stop, step = 1, { dtype, device } = {}) {
  dtype = dtype ?? DType.Int32;
  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  if (step === 0)
    throw new RangeError(`Invalid step for arange: ${step}. Step must be non-zero.`);
  const size$1 = Math.max(0, Math.ceil((stop - start) / step));
  if (size$1 === 0)
    return zeros([0], {
      dtype,
      device
    });
  const exp$2 = AluExp.add(AluExp.const(dtype, start), AluExp.mul(AluExp.cast(dtype, AluVar.idx), AluExp.const(dtype, step)));
  const st = ShapeTracker.fromShape([size$1]);
  return new Array$1({
    source: exp$2,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != null
  });
}
function linspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
  dtype = dtype ?? DType.Float32;
  if (num < 0 || !Number.isInteger(num))
    throw new RangeError(`Invalid num for linspace: ${num}. Must be non-negative integer.`);
  else if (num === 0)
    return zeros([0], {
      dtype,
      device
    });
  else if (num === 1)
    return full([1], start, {
      dtype,
      device
    });
  else if (start === stop)
    return full([num], start, {
      dtype,
      device
    });
  const delta = stop - start;
  const denom = endpoint ? num - 1 : num;
  const exp$2 = AluExp.cast(dtype, AluExp.add(AluExp.f32(start), AluExp.mul(AluExp.f32(delta / denom), AluExp.cast(DType.Float32, AluVar.idx))));
  const st = ShapeTracker.fromShape([num]);
  return new Array$1({
    source: exp$2,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != null
  });
}
function aluCompare(a, b, op) {
  switch (op) {
    case CompareOp.Less:
      return AluExp.cmplt(a, b);
    case CompareOp.Equal:
      return AluExp.cmpne(a, b).not();
    case CompareOp.NotEqual:
      return AluExp.cmpne(a, b);
    case CompareOp.LessEqual:
      return AluExp.add(AluExp.cmplt(a, b), AluExp.cmpne(a, b).not());
  }
}
function _usingCtx() {
  var r = typeof SuppressedError == "function" ? SuppressedError : function(r$1, e$2) {
    var n$1 = Error();
    return n$1.name = "SuppressedError", n$1.error = r$1, n$1.suppressed = e$2, n$1;
  }, e$1 = {}, n = [];
  function using(r$1, e$2) {
    if (e$2 != null) {
      if (Object(e$2) !== e$2)
        throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
      if (r$1)
        var o = e$2[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
      if (o === undefined && (o = e$2[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r$1))
        var t = o;
      if (typeof o != "function")
        throw new TypeError("Object is not disposable.");
      t && (o = function o$1() {
        try {
          t.call(e$2);
        } catch (r$2) {
          return Promise.reject(r$2);
        }
      }), n.push({
        v: e$2,
        d: o,
        a: r$1
      });
    } else
      r$1 && n.push({
        d: e$2,
        a: r$1
      });
    return e$2;
  }
  return {
    e: e$1,
    u: using.bind(null, false),
    a: using.bind(null, true),
    d: function d() {
      var o, t = this.e, s = 0;
      function next() {
        for (;o = n.pop(); )
          try {
            if (!o.a && s === 1)
              return s = 0, n.push(o), Promise.resolve().then(next);
            if (o.d) {
              var r$1 = o.d.call(o.v);
              if (o.a)
                return s |= 2, Promise.resolve(r$1).then(next, err);
            } else
              s |= 1;
          } catch (r$2) {
            return err(r$2);
          }
        if (s === 1)
          return t !== e$1 ? Promise.reject(t) : Promise.resolve();
        if (t !== e$1)
          throw t;
      }
      function err(n$1) {
        return t = t !== e$1 ? new r(n$1, t) : n$1, next();
      }
      return next();
    }
  };
}
var Var = class Var2 {
  static #nextId = 1;
  id;
  aval;
  constructor(aval) {
    this.id = Var2.#nextId++;
    this.aval = aval;
  }
  toString() {
    return `Var(${this.id}):${this.aval.toString()}`;
  }
};
var Lit = class {
  value;
  aval;
  get dtype() {
    return this.aval.dtype;
  }
  constructor(aval, value) {
    if (aval.shape.length !== 0)
      throw new Error(`internal: Lit must be a scalar`);
    this.value = value;
    this.aval = ShapedArray.fromAval(aval);
  }
};
function atomIsLit(atom, literal) {
  return atom instanceof Lit && (literal === undefined || atom.value === literal);
}
var VarPrinter = class {
  names = /* @__PURE__ */ new Map;
  #next = "a";
  #advance() {
    const ret = this.#next;
    let lastNonz = this.#next.length - 1;
    while (lastNonz >= 0 && this.#next[lastNonz] === "z")
      lastNonz--;
    if (lastNonz < 0)
      this.#next = "a".repeat(this.#next.length + 1);
    else {
      let result = this.#next.slice(0, lastNonz);
      result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
      result += "a".repeat(this.#next.length - 1 - lastNonz);
      this.#next = result;
    }
    return ret;
  }
  name(v) {
    if (this.names.has(v))
      return this.names.get(v);
    const name = this.#advance();
    this.names.set(v, name);
    return name;
  }
  nameType(v) {
    return `${this.name(v)}:${v.aval.toString()}`;
  }
};
var JaxprEqn = class {
  constructor(primitive, inputs, params, outBinders) {
    this.primitive = primitive;
    this.inputs = inputs;
    this.params = params;
    this.outBinders = outBinders;
  }
  pprint(usedVars, vp = new VarPrinter) {
    const lhs = PPrint.pp(this.outBinders.map((v) => !usedVars || usedVars.has(v) ? vp.nameType(v) : "_").join(" "));
    let rhs = PPrint.pp(this.primitive);
    const paramsList = Object.entries(this.params).map(([k, v]) => PPrint.pp(`${k}=${v}`));
    if (paramsList.length > 0)
      rhs = rhs.stack(PPrint.pp(" [ ")).stack(PPrint.prototype.concat(...paramsList)).stack(PPrint.pp(" ] "));
    else
      rhs = rhs.stack(PPrint.pp(" "));
    rhs = rhs.stack(PPrint.pp(this.inputs.map((x) => x instanceof Var ? vp.name(x) : String(x.value)).join(" ")));
    return lhs.stack(PPrint.pp(" = ")).stack(rhs);
  }
  toString() {
    return this.pprint().toString();
  }
};
var Jaxpr = class Jaxpr2 {
  #hash;
  constructor(inBinders, eqns, outs) {
    this.inBinders = inBinders;
    this.eqns = eqns;
    this.outs = outs;
  }
  pprint() {
    const vp = new VarPrinter;
    const usedVars = new Set([...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter((x) => x instanceof Var));
    const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
    const eqns = PPrint.prototype.concat(...this.eqns.map((e$1) => e$1.pprint(usedVars, vp)));
    const outs = this.outs.map((x) => x instanceof Var ? vp.name(x) : x.value).join(", ");
    return PPrint.pp(`{ lambda ${inBinders} .`).concat((this.eqns.length ? PPrint.pp("let ").stack(eqns).concat(PPrint.pp(`in ( ${outs} ) }`)) : PPrint.pp(`( ${outs} ) }`)).indent(2));
  }
  toString() {
    return this.pprint().toString();
  }
  getHash() {
    if (this.#hash !== undefined)
      return this.#hash;
    const hasher = new FpHash;
    const varIds = /* @__PURE__ */ new Map;
    const vi = (v) => {
      if (varIds.has(v))
        return varIds.get(v);
      const id = varIds.size + 1;
      varIds.set(v, FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
      return id;
    };
    hasher.update(this.inBinders.length);
    for (const x of this.inBinders)
      hasher.update(vi(x));
    hasher.update(this.eqns.length);
    for (const eqn of this.eqns) {
      hasher.update(eqn.primitive);
      hasher.update(eqn.inputs.length);
      for (const x of eqn.inputs)
        hasher.update(x instanceof Var ? vi(x) : x.value);
      hasher.update(JSON.stringify(eqn.params));
      hasher.update(eqn.outBinders.length);
      for (const x of eqn.outBinders)
        hasher.update(vi(x));
    }
    hasher.update(this.outs.length);
    for (const x of this.outs)
      hasher.update(x instanceof Var ? vi(x) : x.value);
    return this.#hash = hasher.value;
  }
  hash(state) {
    state.update(this.getHash());
  }
  simplify() {
    const context = /* @__PURE__ */ new Map;
    const newEqns = [];
    for (const e$1 of this.eqns) {
      const inputs = e$1.inputs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
      const eqn = new JaxprEqn(e$1.primitive, inputs, e$1.params, e$1.outBinders);
      if (eqn.primitive === Primitive.Add) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 0))
          context.set(c, b);
        else if (atomIsLit(b, 0))
          context.set(c, a);
        else if (atomIsLit(a) && atomIsLit(b))
          context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.dtype === DType.Bool ? Math.min(a.value + b.value, 1) : a.value + b.value));
        else
          newEqns.push(eqn);
      } else if (eqn.primitive === Primitive.Neg) {
        const [a] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a))
          context.set(c, new Lit(a.aval, -a.value));
        else
          newEqns.push(eqn);
      } else if (eqn.primitive === Primitive.Mul) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 1))
          context.set(c, b);
        else if (atomIsLit(b, 1))
          context.set(c, a);
        else if (atomIsLit(a) && atomIsLit(b))
          context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.value * b.value));
        else
          newEqns.push(eqn);
      } else if (eqn.primitive === Primitive.Idiv) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(b, 1))
          context.set(c, a);
        else
          newEqns.push(eqn);
      } else if ((eqn.primitive === Primitive.Broadcast || eqn.primitive === Primitive.Reshape) && deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape) || eqn.primitive === Primitive.Transpose && eqn.params.perm.every((p, i) => p === i) || eqn.primitive === Primitive.Flip && eqn.params.axis.length === 0 || eqn.primitive === Primitive.Shrink && eqn.params.slice.every(([s, e$2], i) => s === 0 && e$2 === eqn.inputs[0].aval.shape[i]) || eqn.primitive === Primitive.Pad && eqn.params.width.every(([w0, w1]) => w0 === 0 && w1 === 0))
        context.set(eqn.outBinders[0], eqn.inputs[0]);
      else
        newEqns.push(eqn);
    }
    const outs = this.outs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
    const usedVars = new Set(outs.filter((x) => x instanceof Var));
    const liveEqns = [];
    for (let i = newEqns.length - 1;i >= 0; i--) {
      const eqn = newEqns[i];
      if (eqn.outBinders.some((v) => usedVars.has(v))) {
        liveEqns.push(eqn);
        for (const v of eqn.inputs)
          if (v instanceof Var)
            usedVars.add(v);
      }
    }
    return new Jaxpr2(this.inBinders, liveEqns.reverse(), outs);
  }
  flatten() {
    if (!this.eqns.some((eqn) => eqn.primitive === Primitive.JitCall))
      return this;
    const newEqns = [];
    const varMap = /* @__PURE__ */ new Map;
    const varMapF = (x) => x instanceof Var ? varMap.get(x) ?? x : x;
    for (const eqn of this.eqns)
      if (eqn.primitive === Primitive.JitCall) {
        const jaxpr = eqn.params.jaxpr.flatten();
        const translation = /* @__PURE__ */ new Map;
        const translationF = (x) => x instanceof Var ? translation.get(x) : x;
        for (const [v, x] of zip(jaxpr.inBinders, eqn.inputs))
          translation.set(v, varMapF(x));
        for (const ieqn of jaxpr.eqns) {
          const inputs = ieqn.inputs.map(translationF);
          const outBinders = [];
          for (const v of ieqn.outBinders) {
            const u = new Var(v.aval);
            outBinders.push(u);
            translation.set(v, u);
          }
          newEqns.push(new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders));
        }
        for (const [v, x] of zip(eqn.outBinders, jaxpr.outs))
          varMap.set(v, translationF(x));
      } else if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x)))
        newEqns.push(new JaxprEqn(eqn.primitive, eqn.inputs.map(varMapF), eqn.params, eqn.outBinders));
      else
        newEqns.push(eqn);
    const newOuts = this.outs.map(varMapF);
    return new Jaxpr2(this.inBinders, newEqns, newOuts);
  }
};
var JaxprType = class {
  constructor(inTypes, outTypes) {
    this.inTypes = inTypes;
    this.outTypes = outTypes;
  }
  toString() {
    const inTypes = this.inTypes.map((aval) => aval.toString()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.toString()).join(", ");
    return `(${inTypes}) -> (${outTypes})`;
  }
};
function typecheckJaxpr(jaxpr) {
  const env = /* @__PURE__ */ new Set;
  for (const v of jaxpr.inBinders) {
    if (env.has(v))
      throw new TypeError(`Duplicate variable binding: ${v}`);
    env.add(v);
  }
  for (const eqn of jaxpr.eqns) {
    const inTypes$1 = eqn.inputs.map((x) => typecheckAtom(env, x));
    const rule = abstractEvalRules[eqn.primitive];
    const outTypes$1 = rule(inTypes$1, eqn.params);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes$1)) {
      if (!outType.equals(outBinder.aval))
        throw new TypeError(`Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`);
      if (env.has(outBinder))
        throw new TypeError(`Duplicate variable binding: ${outBinder}`);
      env.add(outBinder);
    }
  }
  const inTypes = jaxpr.inBinders.map((v) => v.aval);
  const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
  return new JaxprType(inTypes, outTypes);
}
function typecheckAtom(env, x) {
  if (x instanceof Var) {
    if (!env.has(x))
      throw new Error(`Unknown variable: ${x}`);
    return x.aval;
  } else if (x instanceof Lit)
    return x.aval;
  else
    throw new TypeError(`Invalid atom type: ${x}`);
}
function evalJaxpr(jaxpr, args) {
  const env = /* @__PURE__ */ new Map;
  const usageCount = /* @__PURE__ */ new Map;
  for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs))
    if (x instanceof Var)
      usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
  const remainingRefs = /* @__PURE__ */ new Map;
  const read = (x) => {
    if (x instanceof Var) {
      remainingRefs.set(x, (remainingRefs.get(x) ?? 0) - 1);
      return env.get(x);
    } else
      return array(x.value, { dtype: x.dtype });
  };
  const write = (v, val) => {
    if (env.has(v))
      throw new Error(`Variable already bound: ${v}`);
    let refCount = usageCount.get(v) ?? 0;
    if (refCount) {
      env.set(v, val);
      remainingRefs.set(v, refCount);
      while (refCount-- > 1)
        val.ref;
    } else
      val.dispose();
  };
  try {
    for (const [v, arg] of zip(jaxpr.inBinders, args))
      write(v, arg);
    for (const eqn of jaxpr.eqns) {
      const inVals = eqn.inputs.map(read);
      const outVals = bind(eqn.primitive, inVals, eqn.params);
      for (const [v, val] of zip(eqn.outBinders, outVals))
        write(v, val);
    }
    return jaxpr.outs.map(read);
  } catch (error) {
    for (let [v, refCount] of remainingRefs.entries())
      if (refCount > 0) {
        const tracer = env.get(v);
        while (refCount--)
          tracer.dispose();
      }
    throw error;
  }
}
function jaxprAsFun(jaxpr) {
  return (...args) => evalJaxpr(jaxpr, args);
}
var JaxprTracer = class extends Tracer {
  constructor(trace, aval) {
    super(trace);
    this.aval = aval;
  }
  toString() {
    return `JaxprTracer(${this.aval.toString()})`;
  }
  get ref() {
    return this;
  }
  dispose() {}
};
var JaxprTrace = class extends Trace {
  newArg(aval) {
    aval = ShapedArray.fromAval(aval);
    const tracer = this.builder.newTracer(this, aval);
    this.builder.addVar(tracer);
    return tracer;
  }
  getOrMakeConstTracer(val) {
    let tracer = this.builder.constTracers.get(val);
    if (tracer === undefined) {
      tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
      this.builder.addConst(tracer, val instanceof Tracer ? val.ref : array(val));
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;
  processPrimitive(primitive, tracers, params) {
    const avalsIn = tracers.map((t) => t.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const outTracers = avalsOut.map((aval) => this.builder.newTracer(this, aval));
    this.builder.addEqn(new JaxprEqn(primitive, tracers.map((t) => this.builder.getVar(t)), params, outTracers.map((t) => this.builder.addVar(t))));
    return outTracers;
  }
  get builder() {
    return this.main.globalData;
  }
};
var JaxprBuilder = class {
  eqns = [];
  tracerToVar = /* @__PURE__ */ new Map;
  constTracers = /* @__PURE__ */ new Map;
  constVals = /* @__PURE__ */ new Map;
  tracers = [];
  newTracer(trace, aval) {
    const tracer = new JaxprTracer(trace, aval);
    this.tracers.push(tracer);
    return tracer;
  }
  addEqn(eqn) {
    this.eqns.push(eqn);
  }
  addVar(tracer) {
    if (this.tracerToVar.has(tracer))
      throw new Error(`Tracer was added as variable twice: ${tracer}`);
    const v = new Var(tracer.aval);
    this.tracerToVar.set(tracer, v);
    return v;
  }
  getVar(tracer) {
    const v = this.tracerToVar.get(tracer);
    if (v === undefined)
      throw new Error(`Could not find variable for tracer: ${tracer}`);
    return v;
  }
  addConst(tracer, val) {
    const v = this.addVar(tracer);
    this.constTracers.set(val, tracer);
    this.constVals.set(v, val);
    return v;
  }
  build(inTracers, outTracers) {
    let [constVars, consts] = unzip2(this.constVals.entries());
    const t2v = this.getVar.bind(this);
    const inBinders = [...constVars, ...inTracers.map(t2v)];
    const outVars = outTracers.map(t2v);
    let jaxpr = new Jaxpr(inBinders, this.eqns, outVars);
    typecheckJaxpr(jaxpr);
    [jaxpr, consts] = _inlineLiterals(jaxpr, consts);
    return {
      jaxpr,
      consts
    };
  }
};
function _inlineLiterals(jaxpr, consts) {
  const literals = /* @__PURE__ */ new Map;
  const constBinders = [];
  const newConsts = [];
  for (let i = 0;i < consts.length; i++)
    if (ndim$1(consts[i]) === 0 && consts[i] instanceof Array$1) {
      const ar = consts[i];
      literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
    } else {
      constBinders.push(jaxpr.inBinders[i]);
      newConsts.push(consts[i]);
    }
  const newEqns = jaxpr.eqns.map((eqn) => new JaxprEqn(eqn.primitive, eqn.inputs.map((x) => literals.get(x) ?? x), eqn.params, eqn.outBinders));
  const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
  const newJaxpr = new Jaxpr([...constBinders, ...jaxpr.inBinders.slice(consts.length)], newEqns, newOuts);
  typecheckJaxpr(newJaxpr);
  return [newJaxpr, newConsts];
}
function binopAbstractEval([x, y]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray))
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  return [promoteAvals(x, y)];
}
function compareAbstractEval([x, y]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray))
    throw new TypeError("compareAbstractEval expects ShapedArray inputs");
  const aval = promoteAvals(x, y);
  return [new ShapedArray(aval.shape, DType.Bool, false)];
}
function vectorizedUnopAbstractEval([x]) {
  return [ShapedArray.fromAval(x)];
}
var abstractEvalRules = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Idiv]: binopAbstractEval,
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Reciprocal]: vectorizedUnopAbstractEval,
  [Primitive.StopGradient]: vectorizedUnopAbstractEval,
  [Primitive.Cast]([x], { dtype }) {
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.Bitcast]([x], { dtype }) {
    if (x.dtype === DType.Bool || dtype === DType.Bool)
      throw new TypeError("Bitcast to/from bool is not allowed");
    if (byteWidth(x.dtype) !== byteWidth(dtype))
      throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.RandomBits]([k0, k1], { shape: shape$1 }) {
    if (k0.dtype !== DType.Uint32 || k1.dtype !== DType.Uint32)
      throw new TypeError(`RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`);
    const keyShape = generalBroadcast(k0.shape, k1.shape);
    if (!deepEqual(generalBroadcast(keyShape, shape$1), shape$1))
      throw new TypeError(`Keys of shapes ${k0.shape} and ${k1.shape} cannot be broadcast to shape ${shape$1}`);
    return [new ShapedArray(shape$1, DType.Uint32, false)];
  },
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.Asin]: vectorizedUnopAbstractEval,
  [Primitive.Atan]: vectorizedUnopAbstractEval,
  [Primitive.Exp]: vectorizedUnopAbstractEval,
  [Primitive.Log]: vectorizedUnopAbstractEval,
  [Primitive.Erf]: vectorizedUnopAbstractEval,
  [Primitive.Erfc]: vectorizedUnopAbstractEval,
  [Primitive.Sqrt]: vectorizedUnopAbstractEval,
  [Primitive.Min]: binopAbstractEval,
  [Primitive.Max]: binopAbstractEval,
  [Primitive.Reduce]([x], { axis }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pool]([x], { window, strides }) {
    const shape$1 = checkPoolShape(x.shape, window, strides);
    return [new ShapedArray(shape$1, x.dtype, x.weakType)];
  },
  [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
    const shape$1 = checkPoolShape(inShape, window, strides);
    if (!deepEqual(shape$1, x.shape))
      throw new TypeError(`PoolTranspose shape mismatch: expected ${JSON.stringify(shape$1)}, got ${JSON.stringify(x.shape)}`);
    return [new ShapedArray(inShape, x.dtype, x.weakType)];
  },
  [Primitive.Dot]([x, y]) {
    if (x.ndim === 0 && y.ndim === 0)
      throw new TypeError("Dot requires at least 1D inputs");
    const { shape: shape$1, dtype, weakType } = promoteAvals(x, y);
    shape$1.splice(-1, 1);
    return [new ShapedArray(shape$1, dtype, weakType)];
  },
  [Primitive.Conv]([lhs, rhs], params) {
    const { dtype, weakType } = promoteAvals(new ShapedArray([], lhs.dtype, lhs.weakType), new ShapedArray([], rhs.dtype, rhs.weakType));
    const shape$1 = checkConvShape(lhs.shape, rhs.shape, params);
    return [new ShapedArray(shape$1, dtype, weakType)];
  },
  [Primitive.Compare]: compareAbstractEval,
  [Primitive.Where]([cond, x, y]) {
    if (cond.dtype !== DType.Bool)
      throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
    const xy = promoteAvals(x, y);
    const shape$1 = generalBroadcast(cond.shape, xy.shape);
    return [new ShapedArray(shape$1, xy.dtype, xy.weakType)];
  },
  [Primitive.Transpose]([x], { perm }) {
    return [new ShapedArray(perm.map((i) => x.shape[i]), x.dtype, x.weakType)];
  },
  [Primitive.Broadcast]([x], { shape: shape$1 }) {
    return [new ShapedArray(shape$1, x.dtype, x.weakType)];
  },
  [Primitive.Reshape]([x], { shape: shape$1 }) {
    return [new ShapedArray(shape$1, x.dtype, x.weakType)];
  },
  [Primitive.Flip]([x], _) {
    return [ShapedArray.fromAval(x)];
  },
  [Primitive.Shrink]([x], { slice }) {
    const newShape = slice.map((s) => s[1] - s[0]);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pad]([x], { width }) {
    const newShape = x.shape.map((dim, i) => dim + width[i][0] + width[i][1]);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Gather]([x, ...indices], { axis, outDim }) {
    for (const a of indices)
      if (a.dtype !== DType.Int32 && a.dtype !== DType.Uint32)
        throw new TypeError(`Gather indices must be Int32 or Uint32, got ${a.dtype}`);
    if (axis.length !== indices.length)
      throw new TypeError(`Gather: ${axis} axes but ${indices.length} indices`);
    if (indices.length === 0)
      throw new TypeError("Gather must have 1+ indices with same shape");
    if (axis.some((a) => a < 0 || a >= x.shape.length))
      throw new TypeError("Gather axis out of bounds");
    if (outDim < 0 || outDim > x.shape.length - axis.length)
      throw new TypeError("Gather outDim out of bounds");
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length)
      throw new TypeError("Gather axes are not unique");
    const gatherShape = indices.reduce((shape$1, a) => generalBroadcast(shape$1, a.shape), []);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    newShape.splice(outDim, 0, ...gatherShape);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.JitCall](args, { jaxpr }) {
    const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
    if (args.length !== inTypes.length)
      throw new TypeError(`jit_call expected ${inTypes.length} arguments, got ${args.length}`);
    for (let i = 0;i < inTypes.length; i++)
      if (!args[i].equals(inTypes[i]))
        throw new TypeError(`jit_call argument ${i} has type ${args[i]}, expected ${inTypes[i]}`);
    return outTypes;
  }
};
function splitIdx(values, argnums) {
  const a = [];
  const b = [];
  for (let i = 0;i < values.length; i++)
    if (argnums.has(i))
      a.push(values[i]);
    else
      b.push(values[i]);
  return [a, b];
}
function joinIdx(n, a, b, argnums) {
  const result = [];
  let ai = 0;
  let bi = 0;
  for (let i = 0;i < n; i++)
    if (argnums.has(i))
      result.push(a[ai++]);
    else
      result.push(b[bi++]);
  return result;
}
function makeJaxpr$1(f, opts) {
  return (...argsIn) => {
    try {
      var _usingCtx$1 = _usingCtx();
      const staticArgnums = new Set(opts?.staticArgnums ?? []);
      const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);
      const [avalsIn, inTree] = flatten(shapedArgs);
      const [fFlat, outTree] = flattenFun((...dynamicArgs) => {
        return f(...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums));
      }, inTree);
      const builder = new JaxprBuilder;
      const main = _usingCtx$1.u(newMain(JaxprTrace, builder));
      _usingCtx$1.u(newDynamic(main));
      const trace = new JaxprTrace(main);
      const tracersIn = avalsIn.map((aval) => trace.newArg(typeof aval === "object" ? aval : pureArray(aval)));
      const outs = fFlat(...tracersIn);
      const tracersOut = outs.map((out) => fullRaise(trace, out));
      const { jaxpr, consts } = builder.build(tracersIn, tracersOut);
      if (outTree.value === undefined)
        throw new Error("outTree was not set in makeJaxpr");
      return {
        jaxpr: jaxpr.simplify(),
        consts,
        treedef: outTree.value
      };
    } catch (_) {
      _usingCtx$1.e = _;
    } finally {
      _usingCtx$1.d();
    }
  };
}
function jit$1(f, opts) {
  const cache = /* @__PURE__ */ new Map;
  const staticArgnums = new Set(opts?.staticArgnums ?? []);
  const result = (...args) => {
    const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);
    const [argsFlat, inTree] = flatten(dynamicArgs);
    const avalsInFlat = argsFlat.map((x) => ShapedArray.fromAval(getAval(x)));
    const avalsIn = unflatten(inTree, avalsInFlat);
    const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
    const cacheKey = JSON.stringify(jaxprArgs);
    const { jaxpr, consts, treedef: outTree } = runWithCache(cache, cacheKey, () => makeJaxpr$1(f, opts)(...jaxprArgs));
    const outs = bind(Primitive.JitCall, [...consts.map((c) => c.ref), ...argsFlat], {
      name: f.name || "closure",
      jaxpr,
      numConsts: consts.length
    });
    return unflatten(outTree, outs);
  };
  result.dispose = () => {
    for (const { consts } of cache.values())
      for (const c of consts)
        c.dispose();
  };
  return result;
}
var JVPTracer = class extends Tracer {
  constructor(trace, primal, tangent) {
    super(trace);
    this.primal = primal;
    this.tangent = tangent;
  }
  get aval() {
    return this.primal.aval;
  }
  toString() {
    return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
  }
  get ref() {
    this.primal.ref, this.tangent.ref;
    return this;
  }
  dispose() {
    this.primal.dispose();
    this.tangent.dispose();
  }
};
var JVPTrace = class extends Trace {
  pure(val) {
    return this.lift(pureArray(val));
  }
  lift(val) {
    return new JVPTracer(this, val, zerosLike$1(val.ref));
  }
  processPrimitive(primitive, tracers, params) {
    const [primalsIn, tangentsIn] = unzip2(tracers.map((x) => [x.primal, x.tangent]));
    const jvpRule = jvpRules[primitive];
    if (jvpRule === undefined)
      throw new Error(`No JVP rule for: ${primitive}`);
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(([x, t]) => new JVPTracer(this, x, t));
  }
};
function linearTangentsJvp(primitive) {
  return (primals, tangents, params) => {
    const ys = bind(primitive, primals, params);
    const dys = bind(primitive, tangents, params);
    return [ys, dys];
  };
}
function bilinearTangentsJvp(primitive) {
  return ([x, y], [dx, dy], params) => {
    const primal = bind1(primitive, [x.ref, y.ref], params);
    const tangent = bind1(primitive, [x, dy], params).add(bind1(primitive, [dx, y], params));
    return [[primal], [tangent]];
  };
}
function zeroTangentsJvp(primitive) {
  return (primals, tangents, params) => {
    for (const t of tangents)
      t.dispose();
    const ys = bind(primitive, primals, params);
    return [ys, ys.map((y) => zerosLike$1(y.ref))];
  };
}
var jvpRules = {
  [Primitive.Add]: linearTangentsJvp(Primitive.Add),
  [Primitive.Mul]: bilinearTangentsJvp(Primitive.Mul),
  [Primitive.Idiv]: zeroTangentsJvp(Primitive.Idiv),
  [Primitive.Neg]: linearTangentsJvp(Primitive.Neg),
  [Primitive.Reciprocal]([x], [dx]) {
    const xRecip = reciprocal$1(x.ref);
    return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
  },
  [Primitive.StopGradient]: zeroTangentsJvp(Primitive.StopGradient),
  [Primitive.Cast]([x], [dx], { dtype }) {
    if (x.dtype === dtype)
      return [[x], [dx]];
    if (isFloatDtype(dtype) && isFloatDtype(x.dtype))
      return [[cast(x, dtype)], [cast(dx, dtype)]];
    else {
      dx.dispose();
      return [[cast(x.ref, dtype)], [zerosLike$1(x)]];
    }
  },
  [Primitive.Bitcast]([x], [dx], { dtype }) {
    if (x.dtype === dtype)
      return [[x], [dx]];
    dx.dispose();
    return [[bitcast(x.ref, dtype)], [zerosLike$1(x)]];
  },
  [Primitive.RandomBits]: zeroTangentsJvp(Primitive.RandomBits),
  [Primitive.Sin]([x], [dx]) {
    return [[sin$1(x.ref)], [cos$1(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos$1(x.ref)], [neg(sin$1(x)).mul(dx)]];
  },
  [Primitive.Asin]([x], [dx]) {
    const denom = sqrt$1(reciprocal$1(cast(1, x.dtype).sub(x.ref.mul(x.ref))));
    return [[asin$1(x)], [denom.mul(dx)]];
  },
  [Primitive.Atan]([x], [dx]) {
    const denom = cast(1, x.dtype).add(x.ref.mul(x.ref));
    return [[atan$1(x)], [dx.div(denom)]];
  },
  [Primitive.Exp]([x], [dx]) {
    const z = exp$1(x);
    return [[z.ref], [z.mul(dx)]];
  },
  [Primitive.Log]([x], [dx]) {
    return [[log$1(x.ref)], [reciprocal$1(x).mul(dx)]];
  },
  [Primitive.Erf]([x], [dx]) {
    const coeff = 2 / Math.sqrt(Math.PI);
    const expTerm = exp$1(neg(x.ref.mul(x.ref)));
    return [[erf$1(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  [Primitive.Erfc]([x], [dx]) {
    const coeff = -2 / Math.sqrt(Math.PI);
    const expTerm = exp$1(neg(x.ref.mul(x.ref)));
    return [[erfc$1(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  [Primitive.Sqrt]([x], [dx]) {
    const z = sqrt$1(x);
    return [[z.ref], [reciprocal$1(z.mul(2)).mul(dx)]];
  },
  [Primitive.Min]([x, y], [dx, dy]) {
    return [[min$1(x.ref, y.ref)], [where$1(less$1(y, x), dy, dx)]];
  },
  [Primitive.Max]([x, y], [dx, dy]) {
    return [[max$1(x.ref, y.ref)], [where$1(less$1(x, y), dy, dx)]];
  },
  [Primitive.Reduce]([x], [dx], { op, axis }) {
    if (op === AluOp.Add)
      return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
    else if (op === AluOp.Mul) {
      const primal = reduce(x.ref, op, axis);
      const tangent = broadcast(primal.ref, x.shape, axis).mul(reciprocal$1(x)).mul(dx).sum(axis);
      return [[primal], [tangent]];
    } else if (op === AluOp.Min || op === AluOp.Max) {
      const primal = reduce(x.ref, op, axis);
      const notMin = notEqual$1(x, broadcast(primal.ref, x.shape, axis));
      const minCount = where$1(notMin.ref, 0, 1).sum(axis);
      const tangent = where$1(notMin, 0, dx).sum(axis).div(minCount);
      return [[primal], [tangent]];
    } else
      throw new Error(`JVP rule not implemented for reduce op: ${op}`);
  },
  [Primitive.Pool]: linearTangentsJvp(Primitive.Pool),
  [Primitive.PoolTranspose]: linearTangentsJvp(Primitive.PoolTranspose),
  [Primitive.Dot]: bilinearTangentsJvp(Primitive.Dot),
  [Primitive.Conv]: bilinearTangentsJvp(Primitive.Conv),
  [Primitive.Compare]: zeroTangentsJvp(Primitive.Compare),
  [Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
    dcond.dispose();
    return [[where$1(cond.ref, x, y)], [where$1(cond, dx, dy)]];
  },
  [Primitive.Transpose]: linearTangentsJvp(Primitive.Transpose),
  [Primitive.Broadcast]: linearTangentsJvp(Primitive.Broadcast),
  [Primitive.Reshape]: linearTangentsJvp(Primitive.Reshape),
  [Primitive.Flip]: linearTangentsJvp(Primitive.Flip),
  [Primitive.Shrink]: linearTangentsJvp(Primitive.Shrink),
  [Primitive.Pad]: linearTangentsJvp(Primitive.Pad),
  [Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
    const indicesRef = indices.map((t) => t.ref);
    return [[gather(x, indices, axis, outDim)], [gather(dx, indicesRef, axis, outDim)]];
  },
  [Primitive.JitCall](primals, tangents, { name, jaxpr }) {
    const { newJaxpr, newConsts } = jvpJaxpr(jaxpr);
    const outs = bind(Primitive.JitCall, [
      ...newConsts.map((c) => c.ref),
      ...primals,
      ...tangents
    ], {
      name: `${name}_jvp`,
      jaxpr: newJaxpr,
      numConsts: newConsts.length
    });
    const n = outs.length / 2;
    if (!Number.isInteger(n))
      throw new Error("internal: JVP Jaxpr output length is not even");
    const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
    return [primalsOut, tangentsOut];
  }
};
var jvpJaxprCache = /* @__PURE__ */ new Map;
function jvpJaxpr(jaxpr) {
  if (jvpJaxprCache.has(jaxpr))
    return jvpJaxprCache.get(jaxpr);
  const inAvals = jaxpr.inBinders.map((v) => v.aval);
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr$1((primals, tangents) => jvpFlat(jaxprAsFun(jaxpr), primals, tangents))(inAvals, inAvals);
  const result = {
    newJaxpr,
    newConsts
  };
  jvpJaxprCache.set(jaxpr, result);
  return result;
}
function jvpFlat(f, primals, tangents) {
  try {
    var _usingCtx$1 = _usingCtx();
    const main = _usingCtx$1.u(newMain(JVPTrace));
    const trace = new JVPTrace(main);
    const tracersIn = zip(primals, tangents).map(([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t)));
    const outs = f(...tracersIn);
    const tracersOut = outs.map((out) => fullRaise(trace, out));
    return unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
  } catch (_) {
    _usingCtx$1.e = _;
  } finally {
    _usingCtx$1.d();
  }
}
function mappedAval(batchDim, aval) {
  const shape$1 = [...aval.shape];
  shape$1.splice(batchDim, 1);
  return new ShapedArray(shape$1, aval.dtype, aval.weakType);
}
function moveaxis$1(x, src, dst) {
  const t = pureArray(x);
  src = checkAxis(src, t.ndim);
  dst = checkAxis(dst, t.ndim);
  if (src === dst)
    return t;
  const perm = range(t.ndim);
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose$1(t, perm);
}
function moveBatchAxis(axisSize, src, dst, x) {
  if (src === null) {
    const targetShape = [...x.shape];
    targetShape.splice(dst, 0, axisSize);
    return broadcast(x, targetShape, [dst]);
  } else if (src === dst)
    return x;
  else
    return moveaxis$1(x, src, dst);
}
var BatchTracer = class extends Tracer {
  constructor(trace, val, batchDim) {
    super(trace);
    this.val = val;
    this.batchDim = batchDim;
  }
  get aval() {
    if (this.batchDim === null)
      return this.val.aval;
    else
      return mappedAval(this.batchDim, this.val.aval);
  }
  toString() {
    return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
  }
  get ref() {
    this.val.ref;
    return this;
  }
  dispose() {
    this.val.dispose();
  }
  fullLower() {
    if (this.batchDim === null)
      return this.val.fullLower();
    else
      return this;
  }
};
var BatchTrace = class extends Trace {
  pure(val) {
    return this.lift(pureArray(val));
  }
  lift(val) {
    return new BatchTracer(this, val, null);
  }
  processPrimitive(primitive, tracers, params) {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === undefined)
      throw new Error(`No vmap rule for: ${primitive}`);
    if (bdimsIn.every((d) => d === null)) {
      const valOuts$1 = bind(primitive, valsIn, params);
      return valOuts$1.map((x) => new BatchTracer(this, x, null));
    }
    const [valOuts, bdimOuts] = vmapRule(this.axisSize, valsIn, bdimsIn, params);
    return zip(valOuts, bdimOuts).map(([x, bd]) => new BatchTracer(this, x, bd));
  }
  get axisSize() {
    return this.main.globalData;
  }
};
function broadcastBatcher(op) {
  return (axisSize, args, dims) => {
    if (args.length === 0)
      throw new Error("Empty list in broadcastBatcher");
    const nd = Math.max(...args.map((x, i) => ndim$1(x) + (dims[i] === null ? 1 : 0)));
    const firstIdx = dims.findIndex((d) => d !== null);
    const firstBdim = dims[firstIdx] - args[firstIdx].ndim;
    if (zip(args, dims).every(([x, d]) => d === null && ndim$1(x) < -firstBdim || d !== null && d - x.ndim === firstBdim))
      return [[op(...args)], [nd + firstBdim]];
    args = args.map((x, i) => {
      if (dims[i] === null)
        return x;
      x = moveBatchAxis(axisSize, dims[i], 0, x);
      if (x.ndim < nd)
        x = x.reshape([
          x.shape[0],
          ...rep(nd - x.ndim, 1),
          ...x.shape.slice(1)
        ]);
      return x;
    });
    return [[op(...args)], [0]];
  };
}
function unopBatcher(op) {
  return (axisSize, [x], [xBdim], params) => {
    return [[op(x, params)], [xBdim]];
  };
}
var vmapRules = {
  [Primitive.Add]: broadcastBatcher(add$1),
  [Primitive.Mul]: broadcastBatcher(mul),
  [Primitive.Idiv]: broadcastBatcher(idiv),
  [Primitive.Neg]: unopBatcher(neg),
  [Primitive.Reciprocal]: unopBatcher(reciprocal$1),
  [Primitive.StopGradient]: unopBatcher(stopGradient),
  [Primitive.Cast]: unopBatcher((x, { dtype }) => cast(x, dtype)),
  [Primitive.Bitcast]: unopBatcher((x, { dtype }) => bitcast(x, dtype)),
  [Primitive.Sin]: unopBatcher(sin$1),
  [Primitive.Cos]: unopBatcher(cos$1),
  [Primitive.Asin]: unopBatcher(asin$1),
  [Primitive.Atan]: unopBatcher(atan$1),
  [Primitive.Exp]: unopBatcher(exp$1),
  [Primitive.Log]: unopBatcher(log$1),
  [Primitive.Erf]: unopBatcher(erf$1),
  [Primitive.Erfc]: unopBatcher(erfc$1),
  [Primitive.Sqrt]: unopBatcher(sqrt$1),
  [Primitive.Min]: broadcastBatcher(min$1),
  [Primitive.Max]: broadcastBatcher(max$1),
  [Primitive.Reduce](axisSize, [x], [xBdim], { op, axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduce(x, op, newAxis)], [outBdim]];
  },
  [Primitive.Dot](axisSize, [x, y], [xBdim, yBdim]) {
    x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
    y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
    const z = dot$1(x, y);
    return [[z], [z.ndim - 1]];
  },
  [Primitive.Compare](axisSize, args, dims, { op }) {
    return broadcastBatcher((x, y) => compare(x, y, op))(axisSize, args, dims, {});
  },
  [Primitive.Where]: broadcastBatcher(where$1),
  [Primitive.Transpose](axisSize, [x], [xBdim], { perm }) {
    assertNonNull(xBdim);
    const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
    newPerm.splice(xBdim, 0, xBdim);
    return [[transpose$1(x, newPerm)], [xBdim]];
  },
  [Primitive.Broadcast](axisSize, [x], [xBdim], { shape: shape$1, axis }) {
    assertNonNull(xBdim);
    const newShape = shape$1.toSpliced(xBdim, 0, axisSize);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[broadcast(x, newShape, newAxis)], [xBdim]];
  },
  [Primitive.Reshape](axisSize, [x], [xBdim], { shape: shape$1 }) {
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    return [[reshape$1(x, [axisSize, ...shape$1])], [0]];
  },
  [Primitive.Flip](axisSize, [x], [xBdim], { axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[flip$1(x, newAxis)], [xBdim]];
  },
  [Primitive.Shrink](axisSize, [x], [xBdim], { slice }) {
    assertNonNull(xBdim);
    const newSlice = slice.toSpliced(xBdim, 0, [0, axisSize]);
    return [[shrink(x, newSlice)], [xBdim]];
  },
  [Primitive.Pad](axisSize, [x], [xBdim], { width }) {
    assertNonNull(xBdim);
    const newWidth = width.toSpliced(xBdim, 0, [0, 0]);
    return [[pad$1(x, newWidth)], [xBdim]];
  },
  [Primitive.Gather](axisSize, [x, ...indices], [xBdim, ...indicesBdim], { axis, outDim }) {
    if (indicesBdim.every((d) => d === null)) {
      assertNonNull(xBdim);
      const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
      let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
      let newOutDim = outDim;
      if (newOutDim < newBdim)
        newBdim += axis.length;
      else
        newOutDim += 1;
      return [[gather(x, indices, newAxis, newOutDim)], [newBdim]];
    }
    const nd = Math.max(...indices.map((m, i) => ndim$1(m) + (indicesBdim[i] === null ? 1 : 0)));
    indices = indices.map((m, i) => {
      if (indicesBdim[i] === null)
        return m;
      m = moveBatchAxis(axisSize, indicesBdim[i], 0, m);
      if (m.ndim < nd)
        m = m.reshape([
          m.shape[0],
          ...rep(nd - m.ndim, 1),
          ...m.shape.slice(1)
        ]);
      return m;
    });
    if (xBdim === null)
      return [[gather(x, indices, axis, outDim)], [outDim]];
    else {
      x = moveBatchAxis(axisSize, xBdim, 0, x);
      const newAxis = [0, ...axis.map((ax) => ax + 1)];
      const extraBatchIndex = arange(axisSize).reshape([-1, ...rep(nd - 1, 1)]);
      indices.splice(0, 0, extraBatchIndex);
      return [[gather(x, indices, newAxis, outDim)], [outDim]];
    }
  },
  [Primitive.JitCall](axisSize, args, dims, { name, jaxpr }) {
    const { newJaxpr, newConsts } = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind(Primitive.JitCall, [...newConsts.map((c) => c.ref), ...args], {
      name: `${name}_vmap`,
      jaxpr: newJaxpr,
      numConsts: newConsts.length
    });
    return [outs, rep(outs.length, 0)];
  }
};
var vmapJaxprCache = /* @__PURE__ */ new Map;
function vmapJaxpr(jaxpr, axisSize, dims) {
  const cacheKey = JSON.stringify([axisSize, dims]);
  const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult)
    return prevResult;
  const inAvals = jaxpr.inBinders.map((v, i) => {
    if (dims[i] === null)
      return v.aval;
    const shape$1 = [...v.aval.shape];
    shape$1.splice(dims[i], 0, axisSize);
    return new ShapedArray(shape$1, v.aval.dtype, v.aval.weakType);
  });
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr$1((args) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(inAvals);
  const result = {
    newJaxpr,
    newConsts
  };
  if (!vmapJaxprCache.has(jaxpr))
    vmapJaxprCache.set(jaxpr, /* @__PURE__ */ new Map);
  vmapJaxprCache.get(jaxpr).set(cacheKey, result);
  return result;
}
function vmapFlat(f, inAxes, args) {
  let axisSize = undefined;
  for (let i = 0;i < args.length; i++)
    if (inAxes[i] !== null) {
      const arg = args[i];
      if (!(arg instanceof Tracer))
        throw new TypeError("vmap requires Tracer argument for mapped axes");
      const size$1 = arg.shape[inAxes[i]];
      if (axisSize === undefined)
        axisSize = size$1;
      else if (axisSize !== size$1)
        throw new TypeError("vmap requires all mapped axes to have the same size");
    }
  if (axisSize === undefined)
    throw new TypeError("vmap requires at least one mapped axis");
  let valsOut, bdimsOut;
  try {
    var _usingCtx$1 = _usingCtx();
    const main = _usingCtx$1.u(newMain(BatchTrace, axisSize));
    const trace = new BatchTrace(main);
    const tracersIn = args.map((x, i) => inAxes[i] === null ? pureArray(x) : new BatchTracer(trace, pureArray(x), inAxes[i]));
    const outs = f(...tracersIn);
    const tracersOut = outs.map((out) => fullRaise(trace, out));
    [valsOut, bdimsOut] = unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
  } catch (_) {
    _usingCtx$1.e = _;
  } finally {
    _usingCtx$1.d();
  }
  return zip(valsOut, bdimsOut).map(([valOut, bdim]) => moveBatchAxis(axisSize, bdim, 0, valOut));
}
function vmap$1(f, inAxes = 0) {
  return (...args) => {
    const [argsFlat, inTree] = flatten(args);
    let inAxesFlat = [];
    if (typeof inAxes === "number")
      inAxesFlat = rep(argsFlat.length, inAxes);
    else
      for (let i = 0;i < args.length; i++)
        if (inAxes[i] == null)
          inAxesFlat.push(...rep(inTree.childTreedefs[i].size, null));
        else if (typeof inAxes[i] === "number")
          inAxesFlat.push(...rep(inTree.childTreedefs[i].size, inAxes[i]));
        else {
          const [axesFlat, axesTreeDef] = flatten(inAxes[i]);
          if (!inTree.childTreedefs[i].equals(axesTreeDef))
            throw new TreeMismatchError("vmap", inTree.childTreedefs[i], axesTreeDef);
          inAxesFlat.push(...axesFlat);
        }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === undefined)
      throw new Error("outTree was not set in vmap");
    return unflatten(outTree.value, outsFlat);
  };
}
var PartialVal = class PartialVal2 {
  constructor(val, aval) {
    this.val = val;
    this.aval = aval;
  }
  static known(val) {
    return new PartialVal2(val, ShapedArray.fromAval(val.aval));
  }
  static unknown(aval) {
    return new PartialVal2(null, ShapedArray.fromAval(aval));
  }
  get isKnown() {
    return this.val !== null;
  }
  toString() {
    return this.val ? this.val.toString() : this.aval.toString();
  }
};
var PartialEvalTracer = class extends Tracer {
  #rc;
  constructor(trace, pval, recipe) {
    super(trace);
    this.pval = pval;
    this.recipe = recipe;
    this.#rc = 1;
  }
  get aval() {
    return this.pval.aval;
  }
  toString() {
    if (!this.recipe)
      return `PartialEvalTracer(${this.pval.toString()})`;
    else
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
  }
  get ref() {
    if (this.#rc <= 0)
      throw new UseAfterFreeError(this);
    this.#rc++;
    return this;
  }
  dispose() {
    if (this.#rc <= 0)
      throw new UseAfterFreeError(this);
    if (--this.#rc === 0) {
      if (this.pval.isKnown)
        this.pval.val.dispose();
      else if (this.recipe) {
        if (this.recipe.type === "Const")
          this.recipe.val.dispose();
        else if (this.recipe.type === "JaxprEqn")
          this.recipe.tracersIn.forEach((t) => t.dispose());
      }
    }
  }
  fullLower() {
    if (this.pval.isKnown) {
      const val = this.pval.val.ref;
      this.dispose();
      return val;
    }
    return this;
  }
};
var PartialEvalTrace = class extends Trace {
  newArg(pval) {
    if (pval.isKnown)
      return new PartialEvalTracer(this, pval, null);
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }
  pure(val) {
    return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
  }
  lift = this.pure;
  instantiateConst(tracer) {
    if (!tracer.pval.isKnown)
      return tracer;
    else {
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      const val = tracer.pval.val.ref;
      tracer.dispose();
      return new PartialEvalTracer(this, pval, {
        type: "Const",
        val
      });
    }
  }
  processPrimitive(primitive, tracers, params) {
    if (tracers.every((t) => t.pval.isKnown))
      return bind(primitive, tracers.map((t) => t.fullLower()), params);
    if (primitive === Primitive.JitCall) {
      const { name, jaxpr, numConsts } = params;
      return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: []
    };
    const tracersOut = avalsOut.map((aval, i) => {
      if (i > 0)
        tracersIn.forEach((t) => t.ref);
      return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
    });
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }
  #partialEvalJaxpr(name, jaxpr, numConsts, tracers) {
    jaxpr = jaxpr.flatten();
    const inUnknowns = tracers.map((t) => !t.pval.isKnown);
    const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(jaxpr, inUnknowns);
    const [knownTracers, unknownTracers] = partitionList(inUnknowns, tracers);
    const outs1Res = bind(Primitive.JitCall, knownTracers.map((t) => t.ref.fullLower()), {
      name: `${name}_peval`,
      jaxpr: jaxpr1,
      numConsts: 0
    });
    const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
    const res = outs1Res.slice(jaxpr1.outs.length - numRes);
    const resTracers = res.map((x) => this.instantiateConst(fullRaise(this, x)));
    const recipe = {
      type: "JaxprEqn",
      prim: Primitive.JitCall,
      tracersIn: resTracers.concat(unknownTracers),
      params: {
        name: `${name}_resid`,
        jaxpr: jaxpr2,
        numConsts: 0
      },
      avalsOut: jaxpr2.outs.map((x) => x.aval),
      tracerRefsOut: []
    };
    const outs2 = jaxpr2.outs.map((x, i$1) => {
      if (i$1 > 0)
        recipe.tracersIn.forEach((t) => t.ref);
      return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
    });
    recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));
    let i = 0;
    let j = 0;
    return outUnknowns.map((unk) => unk ? outs2[j++] : outs1[i++]);
  }
};
function partialEvalJaxpr(jaxpr, inUnknowns, instantiate) {
  jaxpr = jaxpr.flatten();
  const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
  const knownVars = new Set(knownIns);
  const residuals = /* @__PURE__ */ new Set;
  const eqns1 = [];
  const eqns2 = [];
  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive === Primitive.JitCall)
      throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
    const hasUnknowns = eqn.inputs.some((x) => x instanceof Var && !knownVars.has(x));
    if (hasUnknowns) {
      for (const x of eqn.inputs)
        if (x instanceof Var && knownVars.has(x))
          residuals.add(x);
      eqns2.push(eqn);
    } else {
      eqns1.push(eqn);
      for (const v of eqn.outBinders)
        knownVars.add(v);
    }
  }
  const outUnknowns = jaxpr.outs.map((x) => x instanceof Var && !knownVars.has(x));
  if (instantiate !== undefined)
    for (let i = 0;i < jaxpr.outs.length; i++) {
      const x = jaxpr.outs[i];
      if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
        residuals.add(x);
        outUnknowns[i] = true;
      }
    }
  const residualsL = Array.from(residuals);
  const [ins1, ins2] = partitionList(inUnknowns, jaxpr.inBinders);
  const [outs1, outs2] = partitionList(outUnknowns, jaxpr.outs);
  const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
  const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
  return {
    jaxpr1,
    jaxpr2,
    outUnknowns,
    numRes: residualsL.length
  };
}
var UndefPrimal = class {
  aval;
  constructor(aval) {
    this.aval = ShapedArray.fromAval(aval);
  }
};
function evalJaxprTransposed(jaxpr, args, cotangents) {
  const knownPrimals = /* @__PURE__ */ new Map;
  for (let i = 0;i < jaxpr.inBinders.length; i++)
    if (!(args[i] instanceof UndefPrimal))
      knownPrimals.set(jaxpr.inBinders[i], args[i]);
  const ctStore = /* @__PURE__ */ new Map;
  const readCotangent = (v) => {
    const ct = ctStore.get(v);
    if (ct) {
      ctStore.delete(v);
      return ct;
    } else
      return zeros(v.aval.shape, { dtype: v.aval.dtype });
  };
  const writeCotangent = (v, ct) => {
    if (ct !== null) {
      const oldCt = ctStore.get(v);
      if (oldCt)
        ctStore.set(v, add$1(oldCt, ct));
      else
        ctStore.set(v, ct);
    }
  };
  for (let i = 0;i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var)
      writeCotangent(v, cotangents[i]);
  }
  for (let i = jaxpr.eqns.length - 1;i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    const primalsIn = eqn.inputs.map((v) => v instanceof Lit ? array(v.value, { dtype: v.dtype }) : knownPrimals.has(v) ? knownPrimals.get(v).ref : new UndefPrimal(v.aval));
    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule)
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
    for (let j = 0;j < eqn.inputs.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownPrimals.has(v))
        writeCotangent(v, cotangentsIn[j]);
      else if (cotangentsIn[j] !== null)
        throw new Error("internal: cotangent should be null");
    }
  }
  for (const t of knownPrimals.values())
    t.dispose();
  const results = [];
  for (let i = 0;i < jaxpr.inBinders.length; i++)
    if (args[i] instanceof UndefPrimal)
      results.push(readCotangent(jaxpr.inBinders[i]));
  return results;
}
function unbroadcast(x, target) {
  const shape$1 = target.aval.shape;
  const extraDims = x.ndim > shape$1.length ? range(x.ndim - shape$1.length) : [];
  if (x.ndim < shape$1.length)
    throw new Error(`unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape$1})`);
  const unsqueeze = [];
  const keptReduceDims = [];
  for (let i = 0;i < shape$1.length; i++) {
    const indexFromEnd = shape$1.length - i;
    const indexInX = x.ndim - indexFromEnd;
    const xLen = x.shape[indexInX];
    if (xLen > 1 && shape$1[i] === 1) {
      unsqueeze.push(i);
      keptReduceDims.push(indexInX);
    } else if (shape$1[i] !== xLen)
      throw new Error("internal: unbroadcast shape mismatch");
  }
  const reductionDims = [...extraDims, ...keptReduceDims];
  if (reductionDims.length === 0)
    return x;
  let result = x.sum(reductionDims);
  if (!deepEqual(result.shape, shape$1))
    result = broadcast(result, shape$1, unsqueeze);
  return result;
}
var NonlinearError = class extends TypeError {
  constructor(primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
};
var transposeRules = {
  [Primitive.Mul]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    return [x instanceof UndefPrimal ? unbroadcast(mul(ct, y), x) : null, y instanceof UndefPrimal ? unbroadcast(mul(x, ct), y) : null];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    if (x instanceof UndefPrimal && y instanceof UndefPrimal)
      return [unbroadcast(ct.ref, x), unbroadcast(ct, y)];
    return x instanceof UndefPrimal ? (y.dispose(), [unbroadcast(ct, x), null]) : (x.dispose(), [null, unbroadcast(ct, y)]);
  },
  [Primitive.Reduce]([ct], [x], { op, axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reduce);
    if (op === AluOp.Add)
      return [broadcast(ct, x.aval.shape, axis)];
    else
      throw new NonlinearError(Primitive.Reduce);
  },
  [Primitive.Pool]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Pool);
    return bind(Primitive.PoolTranspose, [ct], {
      inShape: x.aval.shape,
      window,
      strides
    });
  },
  [Primitive.PoolTranspose]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.PoolTranspose);
    return bind(Primitive.Pool, [ct], {
      window,
      strides
    });
  },
  [Primitive.Dot]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Dot);
    const axisSize = generalBroadcast(x.aval.shape, y.aval.shape).slice(-1)[0];
    ct = broadcast(ct, ct.shape.concat(axisSize), [-1]);
    return [x instanceof UndefPrimal ? unbroadcast(mul(ct, y), x) : null, y instanceof UndefPrimal ? unbroadcast(mul(x, ct), y) : null];
  },
  [Primitive.Conv]([ct], [lhs, rhs], params) {
    if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Conv);
    const rev01 = [
      1,
      0,
      ...range(2, ct.ndim)
    ];
    if (lhs instanceof UndefPrimal) {
      let kernel = rhs;
      kernel = transpose$1(kernel, rev01);
      kernel = flip$1(kernel, range(2, kernel.ndim));
      const result = conv(ct, kernel, {
        strides: params.lhsDilation,
        padding: params.padding.map(([pl, _pr], i) => {
          const dilatedKernel = (kernel.shape[i + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + 2] - 1) * params.strides[i] + 1;
          const padBefore = dilatedKernel - 1 - pl;
          const dilatedLhs = (lhs.aval.shape[i + 2] - 1) * params.lhsDilation[i] + 1;
          const padAfter = dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
          return [padBefore, padAfter];
        }),
        lhsDilation: params.strides,
        rhsDilation: params.rhsDilation
      });
      return [result, null];
    } else {
      const newLhs = transpose$1(lhs, rev01);
      const newRhs = transpose$1(ct, rev01);
      let result = conv(newLhs, newRhs, {
        strides: params.rhsDilation,
        padding: params.padding.map(([pl, _pr], i) => {
          const dilatedLhs = (lhs.aval.shape[i + 2] - 1) * params.lhsDilation[i] + 1;
          const dilatedKernel = (rhs.aval.shape[i + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + 2] - 1) * params.strides[i] + 1;
          const padFromLhs = dilatedCt - dilatedLhs;
          const padFromRhs = dilatedKernel - pl - 1;
          return [pl, padFromLhs + padFromRhs];
        }),
        lhsDilation: params.lhsDilation,
        rhsDilation: params.strides
      });
      result = transpose$1(result, rev01);
      return [null, result];
    }
  },
  [Primitive.Where]([ct], [cond, x, y]) {
    const cts = [
      null,
      null,
      null
    ];
    if (cond instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal)
      cts[1] = unbroadcast(where$1(cond.ref, ct.ref, 0), x);
    else
      x.dispose();
    if (y instanceof UndefPrimal)
      cts[2] = unbroadcast(where$1(cond.ref, 0, ct.ref), y);
    else
      y.dispose();
    ct.dispose();
    cond.dispose();
    return cts;
  },
  [Primitive.Transpose]([ct], [x], { perm }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    return [transpose$1(ct, invertPermutation(perm))];
  },
  [Primitive.Broadcast]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduce(ct, AluOp.Add, axis)];
  },
  [Primitive.Reshape]([ct], [x], _) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reshape);
    return [reshape$1(ct, x.aval.shape)];
  },
  [Primitive.Flip]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Flip);
    return [flip$1(ct, axis)];
  },
  [Primitive.Shrink]([ct], [x], { slice }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Shrink);
    const width = slice.map(([s, e$1], i) => [s, x.aval.shape[i] - e$1]);
    return [pad$1(ct, width)];
  },
  [Primitive.Pad]([ct], [x], { width }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Pad);
    const slice = width.map(([s, _e], i) => [s, s + x.aval.shape[i]]);
    return [shrink(ct, slice)];
  },
  [Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Gather);
    if (indices.some((i) => i instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Gather);
    throw new Error("Gather transpose rule is not yet implemented, requires complex Scatter sum operation");
  },
  [Primitive.JitCall](cts, args, { name, jaxpr }) {
    const undefPrimals = args.map((x) => x instanceof UndefPrimal);
    const { newJaxpr, newConsts } = transposeJaxpr(jaxpr, undefPrimals);
    const residuals = args.filter((x, i$1) => !undefPrimals[i$1]);
    const outs = bind(Primitive.JitCall, [
      ...newConsts.map((c) => c.ref),
      ...residuals,
      ...cts
    ], {
      name: `${name}_t`,
      jaxpr: newJaxpr,
      numConsts: newConsts.length
    });
    let i = 0;
    return undefPrimals.map((isUndef) => isUndef ? outs[i++] : null);
  }
};
var transposeJaxprCache = /* @__PURE__ */ new Map;
function transposeJaxpr(jaxpr, undefPrimals) {
  const cacheKey = JSON.stringify(undefPrimals);
  const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult)
    return prevResult;
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr$1((forwardIn, cotangents) => {
    const args = [];
    let forwardInIdx = 0;
    for (let i = 0;i < undefPrimals.length; i++)
      if (undefPrimals[i])
        args.push(new UndefPrimal(inTypes[i]));
      else
        args.push(forwardIn[forwardInIdx++]);
    return evalJaxprTransposed(jaxpr, args, cotangents);
  })(forwardInTypes, outTypes);
  typecheckJaxpr(newJaxpr);
  const result = {
    newJaxpr,
    newConsts
  };
  if (!transposeJaxprCache.has(jaxpr))
    transposeJaxprCache.set(jaxpr, /* @__PURE__ */ new Map);
  transposeJaxprCache.get(jaxpr).set(cacheKey, result);
  return result;
}
var lax_exports = {};
__export(lax_exports, {
  conv: () => conv$1,
  convGeneralDilated: () => convGeneralDilated,
  convWithGeneralPadding: () => convWithGeneralPadding,
  erf: () => erf2,
  erfc: () => erfc2,
  reduceWindow: () => reduceWindow,
  stopGradient: () => stopGradient$1
});
function padtypeToPads(inShape, filterShape, strides, dilation, padding) {
  const padType = padding.toUpperCase();
  switch (padType) {
    case "VALID":
      return rep(inShape.length, [0, 0]);
    case "SAME":
    case "SAME_LOWER": {
      const outShape = inShape.map((size$1, i) => Math.ceil(size$1 / strides[i]));
      const padSizes = zipn(outShape, strides, filterShape, dilation, inShape).map(([o, s, k, d, i]) => Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i));
      if (padType === "SAME")
        return padSizes.map((size$1) => [size$1 >> 1, size$1 - (size$1 >> 1)]);
      else
        return padSizes.map((size$1) => [size$1 - (size$1 >> 1), size$1 >> 1]);
    }
    default:
      throw new Error(`Unknown padding type: ${padType}`);
  }
}
function convGeneralDilated(lhs, rhs, windowStrides, padding, { lhsDilation, rhsDilation } = {}) {
  if (lhs.ndim < 2)
    throw new Error("lhs must have at least 2 dimensions");
  if (rhs.ndim < 2)
    throw new Error("rhs must have at least 2 dimensions");
  if (typeof padding === "string") {
    if (lhsDilation?.some((d) => d !== 1))
      throw new Error("String padding is not supported for transposed convolutions");
    padding = padtypeToPads(lhs.shape.slice(2), rhs.shape.slice(2), windowStrides, rhsDilation ?? rep(rhs.ndim - 2, 1), padding);
  }
  return conv(lhs, rhs, {
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation
  });
}
function convWithGeneralPadding(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
  return convGeneralDilated(lhs, rhs, windowStrides, padding, {
    lhsDilation,
    rhsDilation
  });
}
function conv$1(lhs, rhs, windowStrides, padding) {
  return convGeneralDilated(lhs, rhs, windowStrides, padding);
}
function reduceWindow(operand, computation, windowDimensions, windowStrides) {
  if (operand.ndim < windowDimensions.length)
    throw new Error(`Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`);
  if (!windowStrides)
    windowStrides = rep(windowDimensions.length, 1);
  for (let i = 0;i < operand.ndim; i++)
    computation = vmap$1(computation, 0);
  return computation(bind1(Primitive.Pool, [operand], {
    window: windowDimensions,
    strides: windowStrides
  }));
}
function erf2(x) {
  return erf$1(x);
}
function erfc2(x) {
  return erfc$1(x);
}
function stopGradient$1(x) {
  return stopGradient(x);
}
var numpy_exports = {};
__export(numpy_exports, {
  Array: () => Array$1,
  DType: () => DType,
  abs: () => abs,
  absolute: () => absolute,
  acos: () => acos,
  acosh: () => acosh,
  add: () => add,
  allclose: () => allclose,
  arange: () => arange,
  arccos: () => arccos,
  arccosh: () => arccosh,
  arcsinh: () => arcsinh,
  arctan: () => arctan,
  arctan2: () => arctan2,
  arctanh: () => arctanh,
  argmax: () => argmax,
  argmin: () => argmin,
  array: () => array,
  asin: () => asin,
  asinh: () => asinh,
  astype: () => astype,
  atan: () => atan,
  atan2: () => atan2,
  atanh: () => atanh,
  bool: () => bool,
  broadcastArrays: () => broadcastArrays,
  broadcastShapes: () => broadcastShapes,
  broadcastTo: () => broadcastTo,
  cbrt: () => cbrt,
  clip: () => clip,
  columnStack: () => columnStack,
  concatenate: () => concatenate,
  cos: () => cos,
  cosh: () => cosh,
  deg2rad: () => deg2rad,
  degrees: () => degrees,
  diag: () => diag,
  diagonal: () => diagonal,
  divide: () => divide,
  dot: () => dot,
  dstack: () => dstack,
  e: () => e,
  equal: () => equal,
  eulerGamma: () => eulerGamma,
  exp: () => exp,
  exp2: () => exp2,
  expm1: () => expm1,
  eye: () => eye,
  flip: () => flip,
  fliplr: () => fliplr,
  flipud: () => flipud,
  float16: () => float16,
  float32: () => float32,
  float64: () => float64,
  full: () => full,
  fullLike: () => fullLike$1,
  greater: () => greater,
  greaterEqual: () => greaterEqual,
  hamming: () => hamming,
  hann: () => hann,
  heaviside: () => heaviside,
  hstack: () => hstack,
  hypot: () => hypot,
  identity: () => identity$1,
  inf: () => inf,
  inner: () => inner,
  int32: () => int32,
  isfinite: () => isfinite,
  isinf: () => isinf,
  isnan: () => isnan,
  isneginf: () => isneginf,
  isposinf: () => isposinf,
  less: () => less,
  lessEqual: () => lessEqual,
  linspace: () => linspace,
  log: () => log,
  log10: () => log10,
  log1p: () => log1p,
  log2: () => log2,
  matmul: () => matmul,
  max: () => max,
  maximum: () => maximum,
  mean: () => mean,
  meshgrid: () => meshgrid,
  min: () => min,
  minimum: () => minimum,
  moveaxis: () => moveaxis,
  multiply: () => multiply,
  nan: () => nan,
  ndim: () => ndim,
  negative: () => negative,
  notEqual: () => notEqual,
  ones: () => ones,
  onesLike: () => onesLike,
  outer: () => outer,
  pad: () => pad,
  permuteDims: () => permuteDims,
  pi: () => pi,
  pow: () => pow,
  power: () => power,
  prod: () => prod$1,
  promoteTypes: () => promoteTypes,
  rad2deg: () => rad2deg,
  radians: () => radians,
  ravel: () => ravel,
  reciprocal: () => reciprocal,
  repeat: () => repeat,
  reshape: () => reshape,
  shape: () => shape,
  sign: () => sign,
  sin: () => sin,
  sinh: () => sinh,
  size: () => size,
  sqrt: () => sqrt,
  square: () => square,
  stack: () => stack,
  std: () => std,
  subtract: () => subtract,
  sum: () => sum,
  tan: () => tan,
  tanh: () => tanh,
  tile: () => tile,
  transpose: () => transpose,
  tri: () => tri,
  tril: () => tril,
  triu: () => triu,
  trueDivide: () => trueDivide,
  trunc: () => trunc,
  uint32: () => uint32,
  var_: () => var_,
  vdot: () => vdot,
  vecdot: () => vecdot,
  vstack: () => vstack,
  where: () => where,
  zeros: () => zeros,
  zerosLike: () => zerosLike
});
var float32 = DType.Float32;
var int32 = DType.Int32;
var uint32 = DType.Uint32;
var bool = DType.Bool;
var float16 = DType.Float16;
var float64 = DType.Float64;
var e = Math.E;
var eulerGamma = 0.5772156649015329;
var inf = Number.POSITIVE_INFINITY;
var nan = NaN;
var pi = Math.PI;
var add = add$1;
var multiply = mul;
var negative = neg;
var reciprocal = reciprocal$1;
var sin = sin$1;
var cos = cos$1;
var asin = asin$1;
var atan = atan$1;
var exp = exp$1;
var log = log$1;
var sqrt = sqrt$1;
var minimum = min$1;
var maximum = max$1;
var greater = greater$1;
var less = less$1;
var equal = equal$1;
var notEqual = notEqual$1;
var greaterEqual = greaterEqual$1;
var lessEqual = lessEqual$1;
var where = where$1;
var transpose = transpose$1;
var reshape = reshape$1;
var moveaxis = moveaxis$1;
var pad = pad$1;
var ndim = ndim$1;
var shape = getShape;
var zerosLike = zerosLike$1;
var onesLike = onesLike$1;
var fullLike$1 = fullLike;
function size(a, axis) {
  const s = shape(a);
  return axis === undefined ? prod(s) : s[axis];
}
function astype(a, dtype) {
  return fudgeArray(a).astype(dtype);
}
function sum(a, axis = null, opts) {
  return reduce(a, AluOp.Add, axis, opts);
}
function prod$1(a, axis = null, opts) {
  return reduce(a, AluOp.Mul, axis, opts);
}
function min(a, axis = null, opts) {
  return reduce(a, AluOp.Min, axis, opts);
}
function max(a, axis = null, opts) {
  return reduce(a, AluOp.Max, axis, opts);
}
function mean(a, axis = null, opts) {
  return fudgeArray(a).mean(axis, opts);
}
function argmin(a, axis, opts) {
  a = fudgeArray(a);
  if (axis === undefined) {
    a = a.ravel();
    axis = 0;
  } else
    axis = checkAxis(axis, a.ndim);
  const shape$1 = a.shape;
  const isMax = equal(a, min(a.ref, axis, { keepdims: true }));
  const length = array(shape$1[axis], {
    dtype: int32,
    device: a.device
  });
  const idx = isMax.astype(DType.Int32).mul(arange(shape$1[axis], 0, -1, {
    dtype: int32,
    device: a.device
  }).reshape([shape$1[axis], ...rep(shape$1.length - axis - 1, 1)]));
  return length.sub(max(idx, axis, opts));
}
function argmax(a, axis, opts) {
  a = fudgeArray(a);
  if (axis === undefined) {
    a = a.ravel();
    axis = 0;
  } else
    axis = checkAxis(axis, a.ndim);
  const shape$1 = a.shape;
  const isMax = equal(a, max(a.ref, axis, { keepdims: true }));
  const length = array(shape$1[axis], {
    dtype: int32,
    device: a.device
  });
  const idx = isMax.astype(DType.Int32).mul(arange(shape$1[axis], 0, -1, {
    dtype: int32,
    device: a.device
  }).reshape([shape$1[axis], ...rep(shape$1.length - axis - 1, 1)]));
  return length.sub(max(idx, axis, opts));
}
function flip(x, axis = null) {
  const nd = ndim(x);
  axis = normalizeAxis(axis, nd);
  return flip$1(x, axis);
}
function concatenate(xs, axis = 0) {
  if (xs.length === 0)
    throw new Error("Need at least one array to concatenate");
  const shapes = xs.map(shape);
  axis = checkAxis(axis, shapes[0].length);
  for (let i = 1;i < shapes.length; i++)
    if (shapes[i].length !== shapes[0].length || !shapes[i].every((d, j) => j === axis || d === shapes[0][j]))
      throw new Error(`Cannot concatenate arrays with shapes ${JSON.stringify(shapes)} along axis ${axis}`);
  const makePadAxis = (start, end) => shapes[0].map((_, i) => i === axis ? [start, end] : [0, 0]);
  let result = xs[0];
  for (let i = 1;i < xs.length; i++) {
    const len1 = result.shape[axis];
    const len2 = shapes[i][axis];
    result = pad(result, makePadAxis(0, len2)).add(pad(xs[i], makePadAxis(len1, 0)));
  }
  return result;
}
function stack(xs, axis = 0) {
  if (xs.length === 0)
    throw new Error("Need at least one array to stack");
  const shapes = xs.map((x) => shape(x));
  if (!shapes.every((s) => deepEqual(s, shapes[0])))
    throw new Error(`Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`);
  axis = checkAxis(axis, shapes[0].length + 1);
  const newShape = shapes[0].toSpliced(axis, 0, 1);
  const newArrays = xs.map((x) => fudgeArray(x).reshape(newShape));
  return concatenate(newArrays, axis);
}
function hstack(xs) {
  if (xs.length === 0)
    throw new Error("Need at least one array to hstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0]))
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0)
    return stack(xs);
  else if (nds[0] === 1)
    return concatenate(xs);
  else
    return concatenate(xs, 1);
}
function vstack(xs) {
  if (xs.length === 0)
    throw new Error("Need at least one array to vstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0]))
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0)
    return stack(xs).reshape([-1, 1]);
  else if (nds[0] === 1)
    return stack(xs);
  else
    return concatenate(xs);
}
function dstack(xs) {
  if (xs.length === 0)
    throw new Error("Need at least one array to dstack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0]))
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0)
    return stack(xs).reshape([
      1,
      1,
      -1
    ]);
  else if (nds[0] === 1) {
    const ret = stack(xs, -1);
    return ret.reshape([1, ...ret.shape]);
  } else if (nds[0] === 2)
    return stack(xs, -1);
  else
    return concatenate(xs, 2);
}
function columnStack(xs) {
  if (xs.length === 0)
    throw new Error("Need at least one array to columnStack");
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0]))
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  if (nds[0] === 0)
    return stack(xs).reshape([1, -1]);
  else if (nds[0] === 1)
    return stack(xs, -1);
  else
    return concatenate(xs, 1);
}
function flipud(x) {
  return flip(x, 0);
}
function fliplr(x) {
  return flip(x, 1);
}
var permuteDims = transpose;
function ravel(a) {
  return fudgeArray(a).ravel();
}
function repeat(a, repeats, axis) {
  if (!Number.isInteger(repeats) || repeats < 0)
    throw new Error(`repeat: repeats must be a non-negative integer, got ${repeats}`);
  a = fudgeArray(a);
  if (axis === undefined) {
    a = ravel(a);
    axis = 0;
  }
  axis = checkAxis(axis, a.ndim);
  if (repeats === 1)
    return a;
  const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
  const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
  return broadcast(a, broadcastedShape, [axis + 1]).reshape(finalShape);
}
function tile(a, reps) {
  a = fudgeArray(a);
  if (typeof reps === "number")
    reps = [reps];
  if (!reps.every((r) => Number.isInteger(r) && r >= 0))
    throw new Error(`tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`);
  const ndiff = reps.length - a.ndim;
  if (ndiff > 0)
    a = a.reshape([...rep(ndiff, 1), ...a.shape]);
  if (ndiff < 0)
    reps = [...rep(-ndiff, 1), ...reps];
  const broadcastedShape = [];
  const broadcastAxes = [];
  for (let i = 0;i < a.ndim; i++) {
    if (reps[i] > 1) {
      broadcastedShape.push(reps[i]);
      broadcastAxes.push(broadcastedShape.length - 1);
    }
    broadcastedShape.push(a.shape[i]);
  }
  const finalShape = a.shape.map((d, i) => reps[i] * d);
  return broadcast(a, broadcastedShape, broadcastAxes).reshape(finalShape);
}
function broadcastTo(a, shape$1) {
  const nd = ndim(a);
  if (shape$1.length < nd)
    throw new Error(`broadcastTo: target shape ${JSON.stringify(shape$1)} has fewer dimensions than input array: ${nd}`);
  return broadcast(a, shape$1, range(shape$1.length - nd));
}
function broadcastShapes(...shapes) {
  if (shapes.length === 0)
    return [];
  return shapes.reduce(generalBroadcast);
}
function broadcastArrays(...arrays) {
  const shapes = arrays.map((a) => shape(a));
  const outShape = broadcastShapes(...shapes);
  return arrays.map((a) => broadcastTo(a, outShape));
}
function diagonal(a, offset, axis1, axis2) {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}
function diag(v, k = 0) {
  const a = fudgeArray(v);
  if (!Number.isInteger(k))
    throw new TypeError(`k must be an integer, got ${k}`);
  if (a.ndim === 1) {
    const n = a.shape[0];
    const ret = where(eye(n).equal(1), a.ref, zerosLike(a));
    if (k > 0)
      return pad(ret, [[0, k], [k, 0]]);
    else if (k < 0)
      return pad(ret, [[-k, 0], [0, -k]]);
    else
      return ret;
  } else if (a.ndim === 2)
    return diagonal(a, k);
  else
    throw new TypeError("numpy.diag only supports 1D and 2D arrays");
}
function allclose(actual, expected, options) {
  const { rtol = 0.00001, atol = 0.0000001 } = options ?? {};
  const x = array(actual);
  const y = array(expected);
  if (!deepEqual(x.shape, y.shape))
    return false;
  const xData = x.dataSync();
  const yData = y.dataSync();
  for (let i = 0;i < xData.length; i++)
    if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i]))
      return false;
  return true;
}
function matmul(x, y) {
  if (ndim(x) === 0 || ndim(y) === 0)
    throw new TypeError("matmul: x and y must be at least 1D");
  x = x, y = y;
  if (y.ndim === 1)
    return dot$1(x, y);
  x = x.reshape(x.shape.toSpliced(-1, 0, 1));
  y = y.reshape(y.shape.toSpliced(-2, 0, 1)).transpose([
    ...range(y.shape.length - 1),
    y.shape.length,
    y.shape.length - 1
  ]);
  return dot$1(x, y);
}
function dot(x, y) {
  if (ndim(x) === 0 || ndim(y) === 0)
    return multiply(x, y);
  x = x, y = y;
  if (y.ndim === 1)
    return dot$1(x, y);
  x = x.reshape(x.shape.toSpliced(-1, 0, ...rep(y.ndim - 1, 1)));
  y = y.transpose([
    ...range(y.shape.length - 2),
    y.shape.length - 1,
    y.shape.length - 2
  ]);
  return dot$1(x, y);
}
function inner(x, y) {
  x = reshape(x, shape(x).toSpliced(-1, 0, ...rep(ndim(y) - 1, 1)));
  return dot$1(x, y);
}
function outer(x, y) {
  x = ravel(x);
  y = ravel(y);
  return multiply(x.reshape([x.shape[0], 1]), y);
}
function vecdot(x, y, { axis } = {}) {
  const xaxis = checkAxis(axis ?? -1, ndim(x));
  const yaxis = checkAxis(axis ?? -1, ndim(y));
  if (shape(x)[xaxis] !== shape(y)[yaxis])
    throw new Error(`vecdot: shapes ${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`);
  x = moveaxis(x, xaxis, -1);
  y = moveaxis(y, yaxis, -1);
  return dot$1(x, y);
}
function vdot(x, y) {
  return dot$1(ravel(x), ravel(y));
}
function meshgrid(xs, { indexing } = {}) {
  indexing ??= "xy";
  for (const x of xs)
    if (x.ndim !== 1)
      throw new TypeError(`meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`);
  if (xs.length <= 1)
    return xs;
  if (indexing === "xy") {
    const [a, b, ...rest] = xs;
    const [rb, ra, ...rrest] = meshgrid([
      b,
      a,
      ...rest
    ], { indexing: "ij" });
    return [
      ra,
      rb,
      ...rrest
    ];
  }
  const shape$1 = xs.map((x) => x.shape[0]);
  return xs.map((x, i) => broadcast(x, shape$1, [...range(i), ...range(i + 1, xs.length)]));
}
function tri(n, m, k = 0, { dtype, device } = {}) {
  m ??= n;
  dtype ??= DType.Float32;
  if (!Number.isInteger(n) || n < 0)
    throw new TypeError(`tri: n must be a non-negative integer, got ${n}`);
  if (!Number.isInteger(m) || m < 0)
    throw new TypeError(`tri: m must be a non-negative integer, got ${m}`);
  if (!Number.isInteger(k))
    throw new TypeError(`tri: k must be an integer, got ${k}`);
  const rows = arange(k, n + k, 1, {
    dtype: DType.Int32,
    device
  });
  const cols = arange(0, m, 1, {
    dtype: DType.Int32,
    device
  });
  return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}
function tril(a, k = 0) {
  if (ndim(a) < 2)
    throw new TypeError(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(tri(n, m, k, { dtype: bool }), a.ref, zerosLike(a));
}
function triu(a, k = 0) {
  if (ndim(a) < 2)
    throw new TypeError(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(tri(n, m, k - 1, { dtype: bool }), zerosLike(a.ref), a);
}
function clip(a, min$2, max$2) {
  a = fudgeArray(a);
  if (max$2 !== undefined)
    a = minimum(a, max$2);
  if (min$2 !== undefined)
    a = maximum(a, min$2);
  return a;
}
function absolute(x) {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(-1), x);
}
var abs = absolute;
function sign(x) {
  x = fudgeArray(x);
  return where(notEqual(x.ref, 0), where(less(x.ref, 0), -1, 1), 0);
}
function hamming(M) {
  return cos(linspace(0, 2 * Math.PI, M)).mul(-0.46).add(0.54);
}
function hann(M) {
  return cos(linspace(0, 2 * Math.PI, M)).mul(-0.5).add(0.5);
}
var heaviside = jit$1(function heaviside$1(x1, x2) {
  return where(less(x1.ref, 0), 0, where(equal(x1, 0), x2, 1));
});
function square(x) {
  x = fudgeArray(x);
  return x.ref.mul(x);
}
function tan(x) {
  x = fudgeArray(x);
  return sin(x.ref).div(cos(x));
}
function acos(x) {
  return subtract(pi / 2, asin(x));
}
var hypot = jit$1(function hypot$1(x1, x2) {
  return sqrt(square(x1).add(square(x2)));
});
var atan2 = jit$1(function atan2$1(y, x) {
  const r = sqrt(square(x.ref).add(square(y.ref)));
  const xNeg = less(x.ref, 0);
  const numer = where(xNeg.ref, r.ref.sub(x.ref), y.ref);
  const denom = where(xNeg, y, r.add(x));
  return atan(numer.div(denom)).mul(2);
});
var arccos = acos;
var arctan = atan;
var arctan2 = atan2;
function subtract(x, y) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  return x.sub(y);
}
function trueDivide(x, y) {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (!isFloatDtype(x.dtype) || !isFloatDtype(y.dtype))
    throw new TypeError(`trueDivide: x and y must be floating-point arrays, got ${x.dtype} and ${y.dtype}`);
  return x.div(y);
}
var divide = trueDivide;
function trunc(x) {
  return idiv(x, 1);
}
function exp2(p) {
  return exp(multiply(p, Math.LN2));
}
function log2(x) {
  return log(x).mul(Math.LOG2E);
}
function log10(x) {
  return log(x).mul(Math.LOG10E);
}
function expm1(x) {
  return exp(x).sub(1);
}
function log1p(x) {
  return log(add(1, x));
}
function deg2rad(x) {
  return multiply(x, pi / 180);
}
var radians = deg2rad;
function rad2deg(x) {
  return multiply(x, 180 / pi);
}
var degrees = rad2deg;
var power = jit$1(function power$1(x1, x2) {
  return exp(log(x1).mul(x2));
});
var pow = power;
var cbrt = jit$1(function cbrt$1(x) {
  const sgn = where(less(x.ref, 0), -1, 1);
  return sgn.ref.mul(exp(log(x.mul(sgn)).mul(1 / 3)));
});
var sinh = jit$1(function sinh$1(x) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.sub(emx).mul(0.5);
});
var cosh = jit$1(function cosh$1(x) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.add(emx).mul(0.5);
});
var tanh = jit$1(function tanh$1(x) {
  const negsgn = where(less(x.ref, 0), 1, -1);
  const en2x = exp(x.mul(negsgn.ref).mul(2));
  return en2x.ref.sub(1).div(en2x.add(1)).mul(negsgn);
});
var arcsinh = jit$1(function arcsinh$1(x) {
  return log(x.ref.add(sqrt(square(x).add(1))));
});
var arccosh = jit$1(function arccosh$1(x) {
  return log(x.ref.add(sqrt(square(x).sub(1))));
});
var arctanh = jit$1(function arctanh$1(x) {
  return log(add(1, x.ref).div(subtract(1, x))).mul(0.5);
});
var asinh = arcsinh;
var acosh = arccosh;
var atanh = arctanh;
function var_(x, axis = null, opts) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  if (n === 0)
    throw new Error("var: cannot compute variance over zero-length axis");
  const mu = opts?.mean !== undefined ? opts.mean : mean(x.ref, axis, { keepdims: true });
  return square(x.sub(mu)).sum(axis, { keepdims: opts?.keepdims }).mul(1 / (n - (opts?.correction ?? 0)));
}
function std(x, axis = null, opts) {
  return sqrt(var_(x, axis, opts));
}
function isinf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.ref.equal(Infinity).add(x.equal(-Infinity)) : fullLike$1(x, false);
}
function isnan(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.ref.notEqual(x) : fullLike$1(x, false);
}
function isneginf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike$1(x, false);
}
function isposinf(x) {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike$1(x, false);
}
var isfinite = jit$1(function isfinite$1(x) {
  if (!isFloatDtype(x.dtype))
    return fullLike$1(x, true);
  return isnan(x.ref).add(isinf(x)).notEqual(true);
});
var nn_exports = {};
__export(nn_exports, {
  celu: () => celu,
  elu: () => elu,
  gelu: () => gelu,
  glu: () => glu,
  identity: () => identity,
  leakyRelu: () => leakyRelu,
  logSigmoid: () => logSigmoid,
  logSoftmax: () => logSoftmax,
  logmeanexp: () => logmeanexp,
  logsumexp: () => logsumexp,
  mish: () => mish,
  oneHot: () => oneHot,
  relu: () => relu,
  relu6: () => relu6,
  sigmoid: () => sigmoid,
  silu: () => silu,
  softSign: () => softSign,
  softmax: () => softmax,
  softplus: () => softplus,
  squareplus: () => squareplus,
  standardize: () => standardize,
  swish: () => swish
});
function relu(x) {
  return maximum(x, 0);
}
function relu6(x) {
  return clip(x, 0, 6);
}
function sigmoid(x) {
  return reciprocal(exp(negative(x)).add(1));
}
function softplus(x) {
  return log(exp(x).add(1));
}
function softSign(x) {
  x = fudgeArray(x);
  return x.ref.div(absolute(x).add(1));
}
var silu = jit$1(function silu$1(x) {
  return x.ref.mul(sigmoid(x));
});
var swish = silu;
function logSigmoid(x) {
  return negative(softplus(negative(x)));
}
var identity = fudgeArray;
function leakyRelu(x, negativeSlope = 0.01) {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(negativeSlope), x);
}
function elu(x, alpha = 1) {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref).sub(1).mul(alpha), x);
}
function celu(x, alpha = 1) {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref.div(alpha)).sub(1).mul(alpha), x);
}
var gelu = jit$1(function gelu$1(x, opts) {
  if (opts?.approximate ?? true) {
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    return x.ref.mul(0.5).mul(tanh(x.ref.mul(x.ref.mul(x).mul(0.044715).add(1)).mul(SQRT_2_OVER_PI)).add(1));
  } else
    return x.ref.mul(0.5).mul(erfc$1(negative(x.ref.mul(Math.SQRT1_2))));
}, { staticArgnums: [1] });
function glu(x, axis = -1) {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const size$1 = x.shape[axis];
  if (size$1 % 2 !== 0)
    throw new Error(`glu: axis ${axis} of shape (${x.shape}) does not have even length`);
  const slice = x.shape.map((a$1) => [0, a$1]);
  const a = shrink(x.ref, slice.toSpliced(axis, 1, [0, size$1 / 2]));
  const b = shrink(x, slice.toSpliced(axis, 1, [size$1 / 2, size$1]));
  return a.mul(sigmoid(b));
}
function squareplus(x, b = 4) {
  x = fudgeArray(x);
  return x.ref.add(sqrt(square(x).add(b))).mul(0.5);
}
function mish(x) {
  x = fudgeArray(x);
  return x.ref.mul(tanh(softplus(x)));
}
function softmax(x, axis = -1) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0)
    return onesLike(x);
  const xMax = max(x.ref, axis, { keepdims: true });
  const unnormalized = exp(x.sub(stopGradient(xMax)));
  return unnormalized.ref.div(unnormalized.sum(axis, { keepdims: true }));
}
function logSoftmax(x, axis = -1) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0)
    return zerosLike(x);
  const xMax = max(x.ref, axis, { keepdims: true });
  const shifted = x.sub(stopGradient(xMax));
  const shiftedLogsumexp = log(exp(shifted.ref).sum(axis, { keepdims: true }));
  return shifted.sub(shiftedLogsumexp);
}
function logsumexp(x, axis = null) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0)
    return x;
  const xMax = stopGradient(max(x.ref, axis));
  const xMaxDims = broadcast(xMax.ref, x.shape, axis);
  const shifted = x.sub(xMaxDims);
  return xMax.add(log(exp(shifted).sum(axis)));
}
function logmeanexp(x, axis = null) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0)
    return x;
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  return logsumexp(x, axis).sub(Math.log(n));
}
function standardize(x, axis = -1, opts = {}) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0)
    return x;
  const mu = opts.mean !== undefined ? fudgeArray(opts.mean) : x.ref.mean(axis, { keepdims: true });
  const sigma2 = opts.variance !== undefined ? fudgeArray(opts.variance) : square(x.ref).mean(axis, { keepdims: true }).sub(square(mu.ref));
  return x.sub(mu).div(sqrt(sigma2.add(opts.epsilon ?? 0.00001)));
}
function oneHot(x, numClasses) {
  if (isFloatDtype(x.dtype))
    throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
  return eye(numClasses, undefined, { device: x.device }).slice(x);
}
var random_exports = {};
__export(random_exports, {
  bernoulli: () => bernoulli,
  bits: () => bits,
  exponential: () => exponential,
  key: () => key,
  normal: () => normal,
  split: () => split,
  uniform: () => uniform
});
function validateKeyShape(key$1) {
  if (key$1.ndim === 0)
    throw new Error("Key must have at least one dimension.");
  if (key$1.shape[key$1.shape.length - 1] !== 2)
    throw new Error(`Invalid key shape: ${key$1.shape}. Expected last dimension to be 2.`);
  return key$1.shape.slice(0, -1);
}
function key(seed) {
  seed = seed >>> 0;
  return array([0, seed], { dtype: DType.Uint32 });
}
function split(key$1, num = 2) {
  const shape$1 = typeof num === "number" ? [num] : num;
  for (const len of shape$1)
    if (len <= 0 || !Number.isInteger(len))
      throw new Error(`Invalid split length: ${len}. Must be a positive integer.`);
  const keyShape = validateKeyShape(key$1);
  const k0 = key$1.ref.slice(...keyShape.map(() => null), 0);
  const k1 = key$1.slice(...keyShape.map(() => null), 1);
  return stack([randomBits(k0.ref, k1.ref, shape$1, 0), randomBits(k0, k1, shape$1, 1)], -1);
}
function bits(key$1, shape$1 = []) {
  const keyShape = validateKeyShape(key$1);
  return randomBits(key$1.ref.slice(...keyShape.map(() => null), 0), key$1.slice(...keyShape.map(() => null), 1), shape$1);
}
var uniform = jit$1(function uniform$1(key$1, shape$1 = [], { minval = 0, maxval = 1 } = {}) {
  if (minval >= maxval)
    throw new Error(`Invalid range: [${minval}, ${maxval}).`);
  const mantissa = bits(key$1, shape$1).div(array(512, {
    dtype: DType.Uint32,
    device: key$1.device
  }));
  const float12 = mantissa.add(array(1065353216, {
    dtype: DType.Uint32,
    device: key$1.device
  }));
  const rand = bitcast(float12, DType.Float32).sub(1);
  if (minval === 0 && maxval === 1)
    return rand;
  else
    return rand.mul(maxval - minval).add(minval);
}, { staticArgnums: [1, 2] });
function bernoulli(key$1, p = 0.5, shape$1 = []) {
  p = fudgeArray(p);
  return uniform(key$1, shape$1).less(p);
}
var exponential = jit$1(function exponential$1(key$1, shape$1 = []) {
  const u = uniform(key$1, shape$1);
  return negative(log1p(negative(u)));
}, { staticArgnums: [1] });
var normal = jit$1(function normal$1(key$1, shape$1 = []) {
  const [k1, k2] = split(key$1, 2);
  const u1 = uniform(k1, shape$1);
  const u2 = uniform(k2, shape$1);
  const radius = sqrt(log1p(negative(u1)).mul(-2));
  const theta = u2.mul(2 * Math.PI);
  return radius.mul(cos(theta));
}, { staticArgnums: [1] });
var scipy_special_exports = {};
__export(scipy_special_exports, {
  erf: () => erf2,
  erfc: () => erfc2,
  logSoftmax: () => logSoftmax,
  logit: () => logit,
  logsumexp: () => logsumexp,
  softmax: () => softmax
});
var logit = jit$1(function logit$1(x) {
  return log(x.ref.div(subtract(1, x)));
});
Symbol.dispose ??= Symbol.for("Symbol.dispose");
Symbol.asyncDispose ??= Symbol.for("Symbol.asyncDispose");

// main.ts
var byId = (id) => {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element: ${id}`);
  }
  return el;
};
var ui = {
  status: byId("status-text"),
  round: byId("round-text"),
  scores: byId("scores-text"),
  board: byId("board-text"),
  turnPill: byId("turn-pill"),
  checkpointUrl: byId("checkpoint-url"),
  reloadModel: byId("reload-model"),
  newGame: byId("new-game"),
  aiFirst: byId("ai-first"),
  mctsSims: byId("mcts-sims"),
  actionSelect: byId("action-select"),
  playMove: byId("play-move"),
  randomMove: byId("random-move"),
  log: byId("log")
};
var game = null;
var model = null;
var humanPlayer = 0;
var actionSpace = 0;
var config = {
  cpuct: 1.5,
  maxDepth: 200
};
function setStatus(text) {
  ui.status.textContent = text;
}
function appendLog(line) {
  ui.log.textContent += `${line}
`;
  ui.log.scrollTop = ui.log.scrollHeight;
}
function flattenArray(arr) {
  if (!Array.isArray(arr)) {
    return [arr];
  }
  return arr.flat(Infinity);
}
async function readArray(array2) {
  if (array2 == null) {
    return new Float32Array;
  }
  if (typeof array2.dataSync === "function") {
    return array2.dataSync();
  }
  if (typeof array2.toArray === "function") {
    const out = array2.toArray();
    const resolved = out instanceof Promise ? await out : out;
    return Float32Array.from(flattenArray(resolved));
  }
  if (array2.data) {
    return array2.data;
  }
  throw new Error("Unable to read array data from jax-js array");
}
function decodeFloat16(uint16) {
  const s = (uint16 & 32768) >> 15;
  const e2 = (uint16 & 31744) >> 10;
  const f = uint16 & 1023;
  if (e2 === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  }
  if (e2 === 31) {
    return f ? NaN : s ? -Infinity : Infinity;
  }
  return (s ? -1 : 1) * Math.pow(2, e2 - 15) * (1 + f / Math.pow(2, 10));
}
function float16ToFloat32(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0;i < out.length; i += 1) {
    out[i] = decodeFloat16(view.getUint16(i * 2, true));
  }
  return out;
}
function makeJaxArray(tensor) {
  const data = tensor.data ?? tensor;
  const shape2 = tensor.shape ?? tensor.s;
  if (!shape2) {
    throw new Error("Tensor missing shape metadata");
  }
  return numpy_exports.array(data).reshape(shape2);
}
function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const base = 8 + headerLen;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__")
      continue;
    const [start, end] = info.data_offsets;
    const bytes = new Uint8Array(buffer.slice(base + start, base + end));
    let data;
    const dtype = info.dtype;
    if (dtype === "F32") {
      data = new Float32Array(bytes.buffer);
    } else if (dtype === "F16") {
      data = float16ToFloat32(bytes);
    } else if (dtype === "I64") {
      data = new BigInt64Array(bytes.buffer);
    } else if (dtype === "I32") {
      data = new Int32Array(bytes.buffer);
    } else {
      throw new Error(`Unsupported dtype ${dtype} for tensor ${name}`);
    }
    tensors[name] = {
      data,
      shape: info.shape,
      dtype
    };
  }
  return tensors;
}
function prepareLinear(tensors, name, inSize) {
  const t = tensors[name];
  if (!t) {
    throw new Error(`Missing tensor: ${name}`);
  }
  const shape2 = t.shape;
  if (!shape2 || shape2.length !== 2) {
    throw new Error(`Tensor ${name} expected 2D shape`);
  }
  const w = makeJaxArray(t);
  let outSize;
  let wT;
  if (shape2[1] === inSize) {
    outSize = shape2[0];
    wT = numpy_exports.transpose(w);
  } else if (shape2[0] === inSize) {
    outSize = shape2[1];
    wT = w;
  } else {
    throw new Error(`Tensor ${name} shape ${shape2} does not match input ${inSize}`);
  }
  return { wT, outSize };
}
function prepareBias(tensors, name) {
  const t = tensors[name];
  if (!t) {
    return null;
  }
  return makeJaxArray(t);
}
function relu2(x) {
  return numpy_exports.maximum(x, 0);
}
function linear(x, wT, b) {
  let y = numpy_exports.dot(x, wT.ref);
  if (b) {
    y = y.add(b.ref);
  }
  return y;
}
function buildModel(tensors, obsSize) {
  const trunk0 = prepareLinear(tensors, "trunk.layers.0.weight", obsSize);
  const trunk0b = prepareBias(tensors, "trunk.layers.0.bias");
  const trunk1 = prepareLinear(tensors, "trunk.layers.2.weight", trunk0.outSize);
  const trunk1b = prepareBias(tensors, "trunk.layers.2.bias");
  const policy0 = prepareLinear(tensors, "policy_head.layers.0.weight", trunk1.outSize);
  const policy0b = prepareBias(tensors, "policy_head.layers.0.bias");
  const policy1 = prepareLinear(tensors, "policy_head.layers.2.weight", policy0.outSize);
  const policy1b = prepareBias(tensors, "policy_head.layers.2.bias");
  const value0 = prepareLinear(tensors, "value_head.layers.0.weight", trunk1.outSize);
  const value0b = prepareBias(tensors, "value_head.layers.0.bias");
  const value1 = prepareLinear(tensors, "value_head.layers.2.weight", value0.outSize);
  const value1b = prepareBias(tensors, "value_head.layers.2.bias");
  return {
    obsSize,
    hiddenSize: trunk0.outSize,
    async predict(obs) {
      const x = numpy_exports.array(obs).reshape([1, obsSize]);
      let h = relu2(linear(x, trunk0.wT, trunk0b));
      h = relu2(linear(h, trunk1.wT, trunk1b));
      let p = relu2(linear(h.ref, policy0.wT, policy0b));
      p = linear(p, policy1.wT, policy1b);
      let v = relu2(linear(h, value0.wT, value0b));
      v = linear(v, value1.wT, value1b);
      v = numpy_exports.tanh(v);
      const pFlat = numpy_exports.reshape(p, [actionSpace]);
      const vFlat = numpy_exports.reshape(v, [1]);
      const policyArray = await readArray(pFlat.ref);
      const valueArray = await readArray(vFlat.ref);
      pFlat.dispose();
      vFlat.dispose();
      return {
        policy: policyArray,
        value: valueArray[0] ?? valueArray
      };
    }
  };
}
async function loadModel() {
  setStatus("Loading model…");
  const url = ui.checkpointUrl.value.trim();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch checkpoint: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const tensors = parseSafetensors(buffer);
  actionSpace = action_space_size();
  const obsSize = observation_size();
  const modelObj = buildModel(tensors, obsSize);
  const policyShape = (tensors["policy_head.layers.2.weight"].shape ?? [])[0];
  if (policyShape && policyShape !== actionSpace) {
    console.warn("Policy head size mismatch", policyShape, actionSpace);
  }
  model = modelObj;
  setStatus("Model ready");
}
function softmaxLegal(logits, legalIds) {
  let maxLogit = -Infinity;
  for (const id of legalIds) {
    const v = logits[id];
    if (v > maxLogit)
      maxLogit = v;
  }
  let sum2 = 0;
  const result = [];
  for (const id of legalIds) {
    const value = Math.exp(logits[id] - maxLogit);
    sum2 += value;
    result.push([id, value]);
  }
  if (sum2 <= 0) {
    return result.map(([id]) => [id, 1 / legalIds.length]);
  }
  return result.map(([id, value]) => [id, value / sum2]);
}
function selectChild(node, cpuct) {
  let bestIdx = 0;
  let bestScore = -Infinity;
  const parentN = Math.max(1, node.visitCount);
  node.children.forEach((edge, idx) => {
    const q = edge.visitCount > 0 ? edge.valueSum / edge.visitCount : 0;
    const u = cpuct * edge.prior * (Math.sqrt(parentN) / (1 + edge.visitCount));
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

class MctsNode {
  state;
  toPlay;
  isTerminal;
  children;
  visitCount;
  constructor(state) {
    this.state = state;
    this.toPlay = state.current_player();
    this.isTerminal = state.is_game_over();
    this.children = [];
    this.visitCount = 0;
  }
}
async function evaluateNode(node) {
  if (!model) {
    throw new Error("Model not loaded");
  }
  if (node.isTerminal) {
    return 0;
  }
  if (node.children.length > 0) {
    return 0;
  }
  const obs = node.state.encode_observation(node.toPlay);
  const { policy, value } = await model.predict(obs);
  const legalIds = node.state.legal_action_ids();
  const priors = softmaxLegal(policy, legalIds);
  node.children = priors.map(([id, prior]) => ({
    actionId: id,
    prior,
    visitCount: 0,
    valueSum: 0,
    reward: 0,
    child: null
  }));
  return value;
}
async function runSimulation(root) {
  const path = [];
  let node = root;
  for (let depth = 0;depth < config.maxDepth; depth += 1) {
    if (node.isTerminal || node.children.length === 0) {
      break;
    }
    const idx = selectChild(node, config.cpuct);
    const edge = node.children[idx];
    path.push({ node, edge });
    if (!edge.child) {
      const childState = node.state.clone_handle();
      const applyResult = childState.apply_action_id(edge.actionId);
      edge.reward = applyResult.reward;
      edge.child = new MctsNode(childState);
      node = edge.child;
      break;
    }
    node = edge.child;
  }
  let leafValue = 0;
  if (!node.isTerminal) {
    leafValue = await evaluateNode(node);
  }
  let value = leafValue;
  for (let i = path.length - 1;i >= 0; i -= 1) {
    const { node: parent, edge } = path[i];
    const child = edge.child;
    if (parent.toPlay !== child.toPlay) {
      value = -value;
    }
    value = Math.max(-1, Math.min(1, edge.reward + value));
    edge.visitCount += 1;
    edge.valueSum += value;
    parent.visitCount += 1;
  }
}
async function selectAction(rootState, numSimulations) {
  const root = new MctsNode(rootState);
  await evaluateNode(root);
  for (let i = 0;i < numSimulations; i += 1) {
    await runSimulation(root);
  }
  let bestId = root.children[0]?.actionId ?? 0;
  let bestCount = -Infinity;
  for (const edge of root.children) {
    if (edge.visitCount > bestCount) {
      bestCount = edge.visitCount;
      bestId = edge.actionId;
    }
  }
  return bestId;
}
function updateActionList() {
  if (!game)
    return;
  const ids = game.legal_action_ids();
  const labels = game.legal_action_strings();
  ui.actionSelect.innerHTML = "";
  ids.forEach((id, idx) => {
    const option = document.createElement("option");
    const label = labels[idx] ?? `Action ${id}`;
    option.value = String(id);
    option.textContent = `${idx}: ${label}`;
    ui.actionSelect.appendChild(option);
  });
  ui.playMove.disabled = ids.length === 0;
  ui.randomMove.disabled = ids.length === 0;
}
function updateBoard() {
  if (!game)
    return;
  ui.board.textContent = game.render_text(humanPlayer);
  ui.turnPill.textContent = game.current_player() === humanPlayer ? "Your turn" : "AI thinking";
  ui.round.textContent = `${game.round() + 1}`;
  const scores = Array.from(game.scores());
  ui.scores.textContent = scores.map((s, idx) => `P${idx}: ${s}`).join(" · ");
  updateActionList();
}
function setControlsEnabled(enabled) {
  ui.playMove.disabled = !enabled;
  ui.randomMove.disabled = !enabled;
  ui.actionSelect.disabled = !enabled;
}
async function maybeRunAi() {
  if (!game)
    return;
  while (!game.is_game_over() && game.current_player() !== humanPlayer) {
    setStatus("AI thinking…");
    ui.turnPill.textContent = "AI thinking";
    setControlsEnabled(false);
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const sims = Number.parseInt(ui.mctsSims.value, 10) || 400;
    const rootState = game.clone_handle();
    const actionId = await selectAction(rootState, sims);
    const actionLabel = game.action_id_to_string(actionId);
    const result = game.apply_action_id(actionId);
    appendLog(`AI: ${actionLabel}`);
    updateBoard();
    if (result.game_over) {
      announceGameOver(result);
      return;
    }
  }
  setStatus("Your turn");
  setControlsEnabled(true);
}
function announceGameOver(result) {
  const scores = result.scores ?? [];
  const humanScore = scores[humanPlayer];
  const aiScore = scores[1 - humanPlayer];
  setStatus("Game over");
  ui.turnPill.textContent = "Game over";
  appendLog(`Game over. You ${humanScore >= aiScore ? "win" : "lose"}!`);
}
async function handleHumanMove(actionId) {
  if (!game)
    return;
  const label = game.action_id_to_string(actionId);
  const result = game.apply_action_id(actionId);
  appendLog(`You: ${label}`);
  updateBoard();
  if (result.game_over) {
    announceGameOver(result);
    return;
  }
  await maybeRunAi();
}
function newGame() {
  const seed = BigInt(Date.now());
  game = new_game_state(seed);
  humanPlayer = ui.aiFirst.checked ? 1 : 0;
  ui.log.textContent = "";
  updateBoard();
  setStatus("Your turn");
  if (game.current_player() !== humanPlayer) {
    maybeRunAi();
  }
}
ui.playMove.addEventListener("click", () => {
  const actionId = Number.parseInt(ui.actionSelect.value, 10);
  if (!Number.isNaN(actionId)) {
    handleHumanMove(actionId);
  }
});
ui.randomMove.addEventListener("click", () => {
  if (!game)
    return;
  const ids = game.legal_action_ids();
  if (ids.length === 0)
    return;
  const randomId = ids[Math.floor(Math.random() * ids.length)];
  handleHumanMove(randomId);
});
ui.newGame.addEventListener("click", () => newGame());
ui.reloadModel.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (err) {
    console.error(err);
    setStatus(`Model load failed: ${err?.message ?? err}`);
  }
});
async function boot() {
  setStatus("Loading WASM…");
  await init2();
  try {
    await loadModel();
  } catch (err) {
    console.error(err);
    setStatus(`Model load failed: ${err?.message ?? err}`);
    return;
  }
  newGame();
}
boot();
