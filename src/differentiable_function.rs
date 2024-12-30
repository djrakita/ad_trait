use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use crate::{AD};
use crate::forward_ad::adfn::adfn;
use crate::forward_ad::ForwardADTrait;
use crate::reverse_ad::adr::{adr, GlobalComputationGraph};
use crate::simd::f64xn::f64xn;
use rand::distributions::Distribution;

pub trait DifferentiableFunctionClass {
    type FunctionType<T: AD> : DifferentiableFunctionTrait<T>;
}
impl DifferentiableFunctionClass for () {
    type FunctionType<T: AD> = ();
}

pub trait DifferentiableFunctionTrait<T: AD> {
    fn call(&self, inputs: &[T], freeze: bool) -> Vec<T>;
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
}
impl<T: AD> DifferentiableFunctionTrait<T> for () {
    fn call(&self, _inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![]
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        0
    }
}

pub struct DifferentiableFunctionClassZero;
impl DifferentiableFunctionClass for DifferentiableFunctionClassZero {
    type FunctionType<T: AD> = DifferentiableFunctionZero;
}

pub struct DifferentiableFunctionZero {
    num_inputs: usize,
    num_outputs: usize
}
impl DifferentiableFunctionZero {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self { num_inputs, num_outputs }
    }
}
impl<T: AD> DifferentiableFunctionTrait<T> for DifferentiableFunctionZero {
    fn call(&self, _inputs: &[T], _frozen_freeze: bool) -> Vec<T> {
        vec![T::zero(); self.num_outputs]
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

pub trait DerivativeMethodClass {
    type DerivativeMethod : DerivativeMethodTrait;
}
impl DerivativeMethodClass for () {
    type DerivativeMethod = ();
}

pub trait DerivativeMethodTrait: Clone {
    type T: AD;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>);
}
impl DerivativeMethodTrait for () {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, _inputs: &[f64], _function: &D) -> (Vec<f64>, DMatrix<f64>) {
        panic!("derivative should not actually be called on ()");
    }
}

pub struct DerivativeMethodClassFiniteDifferencing;
impl DerivativeMethodClass for DerivativeMethodClassFiniteDifferencing {
    type DerivativeMethod = FiniteDifferencing;
}

#[derive(Clone)]
pub struct FiniteDifferencing { }
impl FiniteDifferencing {
    pub fn new() -> Self {
        Self {}
    }
}
impl DerivativeMethodTrait for FiniteDifferencing {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);

        let h = 0.0000001;

        let x0 = inputs.to_vec();
        // let f0 = D::call(&x0, args);
        let f0 = function.call(&x0, false);

        for col_idx in 0..num_inputs {
            let mut xh = x0.clone();
            xh[col_idx] += h;
            // let fh = D::call(&xh, args);
            let fh = function.call(&xh, true);
            for row_idx in 0..num_outputs {
                out_derivative[(row_idx, col_idx)] = (fh[row_idx] - f0[row_idx]) / h;
            }
        }

        (f0, out_derivative)
    }
}

pub struct DerivativeMethodClassReverseAD;
impl DerivativeMethodClass for DerivativeMethodClassReverseAD {
    type DerivativeMethod = ReverseAD;
}

#[derive(Clone)]
pub struct ReverseAD { }
impl ReverseAD {
    pub fn new() -> Self {
        Self {}
    }
}
impl DerivativeMethodTrait for ReverseAD {
    type T = adr;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);

        GlobalComputationGraph::get().reset();

        let mut inputs_ad = vec![];
        for input in inputs.iter() {
            inputs_ad.push(adr::new_variable(*input, false));
        }

        let f = function.call(&inputs_ad, false);
        assert_eq!(f.len(), num_outputs);
        let out_value = f.iter().map(|x| x.value()).collect();

        for row_idx in 0..num_outputs {
            if f[row_idx].is_constant() {
                for col_idx in 0..num_inputs {
                    out_derivative[(row_idx, col_idx)] = 0.0;
                }
            } else {
                let grad_output = f[row_idx].get_backwards_mode_grad();
                for col_idx in 0..num_inputs {
                    let d = grad_output.wrt(&inputs_ad[col_idx]);
                    out_derivative[(row_idx, col_idx)] = d;
                }
            }
        }

        (out_value, out_derivative)
    }
}

pub struct DerivativeMethodClassForwardAD;
impl DerivativeMethodClass for DerivativeMethodClassForwardAD {
    type DerivativeMethod = ForwardAD;
}

#[derive(Clone)]
pub struct ForwardAD { }
impl ForwardAD {
    pub fn new() -> Self {
        Self {}
    }
}
impl DerivativeMethodTrait for ForwardAD {
    type T = adfn<1>;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        for col_idx in 0..num_inputs {
            let mut inputs_ad = vec![];
            for (i, input) in inputs.iter().enumerate() {
                if i == col_idx {
                    inputs_ad.push(adfn::new(*input, [1.0]))
                } else {
                    inputs_ad.push(adfn::new(*input, [0.0]))
                }
            }

            // let f = D::call(&inputs_ad, args);
            let freeze = if col_idx == 0 { false } else { true };
            let f = function.call(&inputs_ad, freeze);
            assert_eq!(f.len(), num_outputs, "{}", format!("does not match {}, {}", f.len(), num_outputs));
            for (row_idx, res) in f.iter().enumerate() {
                if out_value.len() < num_outputs {
                    out_value.push(res.value);
                }
                if res.tangent[0].is_nan() {
                    out_derivative[(row_idx, col_idx)] = res.tangent[0];
                } else {
                    out_derivative[(row_idx, col_idx)] = res.tangent[0];
                }
            }
        }

        (out_value, out_derivative)
    }
}

pub struct DerivativeMethodClassForwardADMulti<A: AD + ForwardADTrait>(PhantomData<A>);
impl<A: AD + ForwardADTrait> DerivativeMethodClass for DerivativeMethodClassForwardADMulti<A> {
    type DerivativeMethod = ForwardADMulti<A>;
}

#[derive(Clone)]
pub struct ForwardADMulti<A: AD + ForwardADTrait> {
    phantom_data: PhantomData<A>
}
impl<A: AD + ForwardADTrait> ForwardADMulti<A> {
    pub fn new() -> Self {
        Self { phantom_data: PhantomData::default() }
    }
}
impl<A: AD + ForwardADTrait> DerivativeMethodTrait for ForwardADMulti<A> {
    type T = A;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        let mut curr_idx = 0;

        let mut freeze = false;
        let k = Self::T::tangent_size();
        'l1: loop {
            let mut inputs_ad = vec![];
            for input in inputs.iter() {
                // inputs_ad.push(adf::new(*input, [0.0; K]))
                inputs_ad.push(Self::T::constant(*input));
            }

            'l2: for i in 0..k {
                if curr_idx + i >= num_inputs { break 'l2; }
                // inputs_ad[curr_idx+i].tangent[i] = 1.0;
                inputs_ad[curr_idx+i].set_tangent_value(i, 1.0);
            }

            let f = function.call(&inputs_ad, freeze);
            freeze = true;
            assert_eq!(f.len(), num_outputs);

            for (row_idx, res) in f.iter().enumerate() {
                if out_value.len() < num_outputs {
                    out_value.push(res.value());
                }
                let curr_tangent = res.tangent_as_vec();
                'l3: for i in 0..k {
                    if curr_idx + i >= num_inputs { break 'l3; }
                    // out_derivative[(row_idx, curr_idx+i)] = res.tangent[i];
                    if curr_tangent[i].is_nan() {
                        out_derivative[(row_idx, curr_idx+i)] = curr_tangent[i];
                    } else {
                        out_derivative[(row_idx, curr_idx+i)] = curr_tangent[i];
                    }
                }
            }

            curr_idx += k;
            if curr_idx >= num_inputs { break 'l1; }
        }

        return (out_value, out_derivative)
    }
}

pub struct DerivativeMethodClassFiniteDifferencingMulti<const K: usize>;
impl<const K: usize> DerivativeMethodClass for DerivativeMethodClassFiniteDifferencingMulti<K> {
    type DerivativeMethod = FiniteDifferencingMulti2<K>;
}

#[derive(Clone)]
pub struct FiniteDifferencingMulti2<const K: usize>;
impl<const K: usize> FiniteDifferencingMulti2<K> {
    pub fn new() -> Self {
        Self {}
    }
}
impl<const K: usize> DerivativeMethodTrait for FiniteDifferencingMulti2<K> {
    type T = f64xn<K>;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        let h = 0.0000001;

        let mut curr_idx = 0;
        let mut first_loop = true;

        'l1: loop {
            let mut inputs_ad = vec![];
            for input in inputs.iter() {
                inputs_ad.push(f64xn::<K>::splat(*input));
            }

            if first_loop {
                'l2: for i in 0..K {
                    if curr_idx + i >= num_inputs { break 'l2; }
                    if i+1 >= K { break 'l2; }
                    inputs_ad[curr_idx + i].value[i + 1] += h;
                }
            } else {
                'l2: for i in 0..K {
                    if curr_idx + i >= num_inputs { break 'l2; }
                    if i >= K { break 'l2; }
                    inputs_ad[curr_idx + i].value[i] += h;
                }
            }

            // let f = D::call(&inputs_ad, args);
            let f = function.call(&inputs_ad, false);
            assert_eq!(f.len(), num_outputs);

            if first_loop {
                for res in f.iter() {
                    out_value.push(res.value[0]);
                }
            }

            for (row_idx, res) in f.iter().enumerate() {
                if first_loop {
                    'l3: for i in 0..K {
                        if curr_idx + i >= num_inputs { break 'l3; }
                        if i + 1 >= K { break 'l3; }
                        out_derivative[(row_idx, curr_idx+i)] = (res.value[i+1] - out_value[row_idx]) / h;
                    }
                } else {
                    'l3: for i in 0..K {
                        if curr_idx + i >= num_inputs { break 'l3; }
                        if i >= K { break 'l3; }
                        out_derivative[(row_idx, curr_idx + i)] = (res.value[i] - out_value[row_idx]) / h;
                    }
                }
            }

            if first_loop {
                first_loop = false;
                curr_idx += K-1;
            } else {
                curr_idx += K;
            }

            if curr_idx >= num_inputs {
                break 'l1;
            }
        }

        return (out_value, out_derivative)
    }
}

#[derive(Clone)]
pub struct WASP {
    cache: Arc<RwLock<WASPCache>>,
    num_f_calls: Arc<RwLock<usize>>,
    d_theta: f64,
    d_ell: f64
}
impl WASP {
    pub fn new(n: usize, m: usize, orthonormal_delta_x: bool, d_theta: f64, d_ell: f64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(WASPCache::new(n, m, orthonormal_delta_x))),
            num_f_calls: Arc::new(RwLock::new(0)),
            d_theta,
            d_ell,
        }
    }
    pub fn new_default(n: usize, m: usize) -> Self {
        Self::new(n, m, true, 0.3, 0.3)
    }
    pub fn num_f_calls(&self) -> usize {
        return self.num_f_calls.read().unwrap().clone()
    }
}
impl DerivativeMethodTrait for WASP {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut num_f_calls = 0;
        let f_k = function.call(inputs, false);
        let f_k_dv = DVector::from_column_slice(&f_k);
        num_f_calls += 1;
        let epsilon = 0.000001;

        let mut cache = self.cache.write().unwrap();
        let n = inputs.len();

        let x = DVector::<f64>::from_column_slice(inputs);

        loop {
            let i = cache.i.clone();

            let delta_x_i = cache.delta_x.column(i);

            let x_k_plus_delta_x_i = &x + epsilon*&delta_x_i;
            let f_k_plus_delta_x_i = DVector::<f64>::from_column_slice(&function.call(x_k_plus_delta_x_i.as_slice(), true));
            num_f_calls += 1;
            let delta_f_i = (&f_k_plus_delta_x_i - &f_k_dv) / epsilon;
            let delta_f_i_hat = cache.delta_f_t.row(i);
            let delta_f_i_hat = DVector::from_column_slice(delta_f_i_hat.transpose().as_slice());
            let return_result = close_enough(&delta_f_i, &delta_f_i_hat, self.d_theta, self.d_ell);

            cache.delta_f_t.set_row(i, &delta_f_i.transpose());
            let c_1_mat = &cache.c_1[i];
            let c_2_mat = &cache.c_2[i];
            let delta_f_t = &cache.delta_f_t;

            let d_t_star = c_1_mat*delta_f_t + c_2_mat*delta_f_i.transpose();
            let d_star = d_t_star.transpose();

            let tmp = &d_star * &cache.delta_x;
            cache.delta_f_t = tmp.transpose();

            let mut new_i = i + 1;
            if new_i >= n { new_i = 0; }
            cache.i = new_i;

            if return_result {
                *self.num_f_calls.write().unwrap() = num_f_calls;
                return (f_k, d_star);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct WASPCache {
    pub i: usize,
    pub delta_f_t: DMatrix<f64>,
    pub delta_x: DMatrix<f64>,
    pub c_1: Vec<DMatrix<f64>>,
    pub c_2: Vec<DVector<f64>>
}
impl WASPCache {
    pub fn new(n: usize, m: usize, orthonormal_delta_x: bool) -> Self {
        let delta_f_t = DMatrix::<f64>::identity(n, m);
        let delta_x = get_tangent_matrix(n, orthonormal_delta_x);
        let mut c_1 = vec![];
        let mut c_2 = vec![];

        let a_mat = 2.0 * &delta_x * &delta_x.transpose();
        let a_inv_mat = a_mat.try_inverse().unwrap();

        for i in 0..n {
            let delta_x_i = DVector::<f64>::from_column_slice(delta_x.column(i).as_slice());
            let s_i = (delta_x_i.transpose() * &a_inv_mat * &delta_x_i)[(0,0)];
            let s_i_inv = 1.0 / s_i;
            let c_1_mat = &a_inv_mat * (DMatrix::<f64>::identity(n, n) - s_i_inv * &delta_x_i * delta_x_i.transpose() * &a_inv_mat) * 2.0 * &delta_x;
            let c_2_mat = s_i_inv * &a_inv_mat * delta_x_i;
            c_1.push(c_1_mat);
            c_2.push(c_2_mat);
        }

        return Self {
            i: 0,
            delta_f_t,
            delta_x,
            c_1,
            c_2,
        }
    }
}

#[derive(Clone)]
pub struct WASP2 {
    cache: Arc<RwLock<WASPCache2>>,
    num_f_calls: Arc<RwLock<usize>>,
    d_theta: f64,
    d_ell: f64
}
impl WASP2 {
    pub fn new(n: usize, m: usize, alpha:f64, orthonormal_delta_x: bool, d_theta: f64, d_ell: f64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(WASPCache2::new(n, m, alpha, orthonormal_delta_x))),
            num_f_calls: Arc::new(RwLock::new(0)),
            d_theta,
            d_ell,
        }
    }
    pub fn new_default(n: usize, m: usize) -> Self {
        Self::new(n, m, 0.98, true, 0.3, 0.3)
    }
    pub fn num_f_calls(&self) -> usize {
        return self.num_f_calls.read().unwrap().clone()
    }
}
impl DerivativeMethodTrait for WASP2 {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut num_f_calls = 0;
        let f_k = function.call(inputs, false);
        let f_k_dv = DVector::from_column_slice(&f_k);
        num_f_calls += 1;
        let epsilon = 0.000001;

        let mut cache = self.cache.write().unwrap();
        let n = inputs.len();

        let x = DVector::<f64>::from_column_slice(inputs);

        loop {
            let i = cache.i.clone();

            let delta_x_i = cache.delta_x.column(i);

            let x_k_plus_delta_x_i = &x + epsilon*&delta_x_i;
            let f_k_plus_delta_x_i = DVector::<f64>::from_column_slice(&function.call(x_k_plus_delta_x_i.as_slice(), true));
            num_f_calls += 1;
            let delta_f_i = (&f_k_plus_delta_x_i - &f_k_dv) / epsilon;
            let delta_f_i_hat = &cache.curr_d * &delta_x_i;
            // let delta_f_i_hat = cache.delta_f_t.row(i);
            // let delta_f_i_hat = DVector::from_column_slice(delta_f_i_hat.transpose().as_slice());
            let return_result = close_enough(&delta_f_i, &delta_f_i_hat, self.d_theta, self.d_ell);

            cache.delta_f_t.set_row(i, &delta_f_i.transpose());
            let c_1_mat = &cache.c_1[i];
            let c_2_mat = &cache.c_2[i];
            let delta_f_t = &cache.delta_f_t;

            let d_t_star = c_1_mat*delta_f_t + c_2_mat*delta_f_i.transpose();
            let d_star = d_t_star.transpose();
            cache.curr_d = d_star.clone();

            let mut new_i = i + 1;
            if new_i >= n { new_i = 0; }
            cache.i = new_i;

            if return_result {
                *self.num_f_calls.write().unwrap() = num_f_calls;
                return (f_k, d_star);
            }
        }
    }
}

pub struct WASPCache2 {
    pub i: usize,
    pub curr_d: DMatrix<f64>,
    pub delta_f_t: DMatrix<f64>,
    pub delta_x: DMatrix<f64>,
    pub c_1: Vec<DMatrix<f64>>,
    pub c_2: Vec<DVector<f64>>
}
impl WASPCache2 {
    pub fn new(n: usize, m: usize, alpha: f64, orthonormal_delta_x: bool) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0);

        let curr_d = DMatrix::<f64>::identity(n, n);
        let delta_f_t = DMatrix::<f64>::identity(n, m);
        let delta_x = get_tangent_matrix(n, orthonormal_delta_x);
        let mut c_1 = vec![];
        let mut c_2 = vec![];

        for i in 0..n {
            let delta_x_i = DVector::<f64>::from_column_slice(delta_x.column(i).as_slice());
            let mut w_i = DMatrix::<f64>::zeros(n, n);
            for j in 0..n {
                let exponent = math_mod(i as i32 - j as i32, n as i32) as f64 / (n as i32 - 1) as f64;
                w_i[(j, j)] = alpha * (1.0 - alpha).powf(exponent);
            }
            let w_i_2 = &w_i * &w_i;

            let a_i = 2.0 * &delta_x * &w_i_2 * &delta_x.transpose();
            let a_i_inv = a_i.clone().try_inverse().unwrap();

            let s_i = (delta_x_i.transpose() * &a_i_inv * &delta_x_i)[(0,0)];
            let s_i_inv = 1.0 / s_i;
            let c_1_mat = &a_i_inv * (DMatrix::<f64>::identity(n, n) - s_i_inv * &delta_x_i * delta_x_i.transpose() * &a_i_inv) * 2.0 * &delta_x * &w_i_2;
            let c_2_mat = s_i_inv * &a_i_inv * delta_x_i;
            c_1.push(c_1_mat);
            c_2.push(c_2_mat);
        }

        return Self {
            i: 0,
            curr_d,
            delta_f_t,
            delta_x,
            c_1,
            c_2,
        }
    }
}

pub fn math_mod(a: i32, b: i32) -> i32 {
    return ((a % b) + b) % b;
}

pub (crate) fn get_tangent_matrix(n: usize, orthogonal: bool) -> DMatrix<f64> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);

    let t = DMatrix::<f64>::from_fn(n, n, |_, _| uniform.sample(&mut rng));

    return if orthogonal {
        let svd = t.svd(true, true);
        let delta_x = svd.u.as_ref().unwrap() * svd.v_t.as_ref().unwrap();
        delta_x
    } else {
        t
    }
}

pub (crate) fn close_enough(a: &DVector<f64>, b: &DVector<f64>, d_theta: f64, d_ell: f64) -> bool {
    let a_n = a.norm();
    let b_n = b.norm();

    let tmp = ((a.dot(&b) / ( a_n*b_n )) - 1.0).abs();
    if tmp > d_theta { return false; }

    let tmp1 = if b_n != 0.0 {
        ((a_n / b_n) - 1.0).abs()
    } else {
        f64::MAX
    };
    let tmp2 = if a_n != 0.0 {
        ((b_n / a_n) - 1.0).abs()
    } else {
        f64::MAX
    };

    if f64::min(tmp1, tmp2) > d_ell { return false; }

    return true;
}

/*

pub fn math_modulus(a: i64, b: i64) -> usize {
    (((a % b) + b) % b) as usize
}

pub fn get_tangent_matrix(n: usize, orthonormalize: bool) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(n, n);
    let mut rng = rand::thread_rng();

    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = rng.gen_range(-1.0..=1.0);
        }
    }

    if orthonormalize {
        let svd = out.svd(true, true);
        out = svd.u.as_ref().unwrap()*svd.v_t.as_ref().unwrap();
    }

    return out;
}

pub fn wasp_projection<D: DifferentiableFunctionTrait<f64> + ?Sized>(f: &D, f_x_k: &DVector<f64>, x_k: &[f64], cache: &WASPCache) -> DMatrix<f64> {
    let epsilon = 0.00001;
    let x_k = DVector::from_column_slice(x_k);
    let i = cache.i.lock().unwrap();
    let c_1_mat = &cache.c_1_mats[*i];
    let c_2_mat = &cache.c_2_mats[*i];
    let delta_x_i = DVector::from_column_slice(cache.delta_x_mat.column(*i).as_slice());
    let f_x_k_delta = DVector::from_column_slice(&f.call((&x_k + epsilon*&delta_x_i).as_slice(), true));
    let delta_f_i = (f_x_k_delta - f_x_k) / epsilon;
    let mut delta_f_hat_t = cache.delta_f_mat_t.lock().unwrap();
    delta_f_hat_t.set_row(*i, &delta_f_i.transpose());
    return c_1_mat*&*delta_f_hat_t + c_2_mat*&delta_f_i.transpose();
}

pub fn wasp_projection2<D: DifferentiableFunctionTrait<f64> + ?Sized>(f: &D, f_x_k: &DVector<f64>, x_k: &[f64], cache: &WASPCache2) -> DMatrix<f64> {
    let epsilon = 0.00001;
    let x_k = DVector::from_column_slice(x_k);
    let i = cache.i.lock().unwrap();
    let c_1_mat = &cache.c_1_mats[*i];
    let c_2_mat = &cache.c_2_mats[*i];
    let delta_x_i = DVector::from_column_slice(cache.delta_x_mat.column(*i).as_slice());
    let f_x_k_delta = DVector::from_column_slice(&f.call((&x_k + epsilon*&delta_x_i).as_slice(), true));
    let delta_f_i = (f_x_k_delta - f_x_k) / epsilon;
    let mut delta_f_hat_t = cache.delta_f_mat_t.lock().unwrap();
    delta_f_hat_t.set_row(*i, &delta_f_i.transpose());
    return c_1_mat*&*delta_f_hat_t + c_2_mat*&delta_f_i.transpose();
}

pub fn close_enough(d_a_t_mat: &DMatrix<f64>, d_b_t_mat: &DMatrix<f64>, l: usize, m: usize, d_theta: f64) -> bool {
    let mut numbers: Vec<usize> = (0..m).collect();

    let mut rng = thread_rng();
    numbers.shuffle(&mut rng);

    let js: Vec<usize> = numbers.into_iter().take(l).collect();

    // println!("{}", d_a_t_mat);
    // println!("{}", d_b_t_mat);
    // println!("---");

    for j in js {
        let d_a = DVector::from_column_slice(d_a_t_mat.column(j).as_slice());
        let d_b = DVector::from_column_slice(d_b_t_mat.column(j).as_slice());

        let d_a_n = d_a.norm();
        let d_b_n = d_b.norm();

        let dot = d_a.dot(&d_b);
        let angle = (dot / (d_a_n * d_b_n)).acos();
        // println!("{:?}", angle);

        if angle > d_theta { return false; }
    }

    return true;
}

#[inline(always)]
pub fn close_enough2(a: &DVector<f64>, b: &DVector<f64>, d_theta: f64, d_l: f64) -> bool {
    let an = a.norm();
    let bn = b.norm();
    let d = a.dot(b);

    if (d / (an * bn) - 1.0).abs() > d_theta { return false; }
    if (an / bn - 1.0).abs() > d_l { return false; }

    return true;
}

pub fn derivative_angular_distance(d_a_t_mat: &DMatrix<f64>, d_b_t_mat: &DMatrix<f64>) -> f64 {
    let m = d_a_t_mat.ncols();

    let mut max_angle = f64::MIN;

    for j in 0..m {
        let d_a = DVector::from_column_slice(d_a_t_mat.column(j).as_slice());
        let d_b = DVector::from_column_slice(d_b_t_mat.column(j).as_slice());

        let d_a_n = d_a.norm();
        let d_b_n = d_b.norm();

        let dot = d_a.dot(&d_b);
        let angle = (dot / (d_a_n * d_b_n)).acos();
        if angle > max_angle { max_angle = angle; }
    }

    return max_angle;
}

#[derive(Clone)]
pub struct WASPCache {
    pub delta_f_mat_t: Arc<Mutex<DMatrix<f64>>>,
    pub delta_x_mat: DMatrix<f64>,
    pub c_1_mats: Vec<DMatrix<f64>>,
    pub c_2_mats: Vec<DVector<f64>>,
    pub i: Arc<Mutex<usize>>
}
impl WASPCache {
    pub fn new(n: usize, m: usize, alpha: f64, orthonormalize: bool) -> Self {
        let delta_x_mat = get_tangent_matrix(n, orthonormalize);
        let mut c_1_mats = vec![];
        let mut c_2_mats = vec![];

        for i in 0..n {
            let delta_x_i = DVector::from_column_slice(delta_x_mat.column(i).as_slice());
            let mut w_i_mat = DMatrix::zeros(n, n);
            for j in 0..n {
                let exp = math_modulus(i as i64 - j as i64, n as i64) as f64 / ((n - 1) as f64);
                w_i_mat[(j,j)] = alpha*(1.0 - alpha).pow(  exp );
            }
            let w_i_mat_2 = &w_i_mat * & w_i_mat;
            let a_i_mat = 2.0*(&delta_x_mat * &w_i_mat_2 * &delta_x_mat.transpose());
            let a_i_mat_inv = a_i_mat.clone().try_inverse().unwrap();
            let s_i = (&delta_x_i.transpose() * &a_i_mat_inv * &delta_x_i)[0];
            let s_i_inv = 1.0 / s_i;
            let c_1_mat = &a_i_mat_inv*(DMatrix::identity(n, n) - s_i_inv*&delta_x_i*&delta_x_i.transpose()*&a_i_mat_inv)*2.0*&delta_x_mat*&w_i_mat_2;
            let c_2_mat = s_i_inv*&a_i_mat_inv*&delta_x_i;
            c_1_mats.push(c_1_mat);
            c_2_mats.push(c_2_mat);
        }

        Self {
            delta_f_mat_t: Arc::new(Mutex::new(DMatrix::zeros(n, m))),
            delta_x_mat,
            c_1_mats,
            c_2_mats,
            i: Arc::new(Mutex::new(0)),
        }
    }
}

#[derive(Clone)]
pub struct WASPCache2 {
    pub delta_f_mat_t: Arc<Mutex<DMatrix<f64>>>,
    pub delta_x_mat: DMatrix<f64>,
    pub c_1_mats: Vec<DMatrix<f64>>,
    pub c_2_mats: Vec<DVector<f64>>,
    pub i: Arc<Mutex<usize>>
}
impl WASPCache2 {
    pub fn new(n: usize, m: usize, orthonormalize: bool) -> Self {
        let delta_x_mat = get_tangent_matrix(n, orthonormalize);
        let mut c_1_mats = vec![];
        let mut c_2_mats = vec![];

        for i in 0..n {
            let delta_x_i = DVector::from_column_slice(delta_x_mat.column(i).as_slice());
            let a_i_mat = 2.0*(&delta_x_mat * &delta_x_mat.transpose());
            let a_i_mat_inv = a_i_mat.clone().try_inverse().unwrap();
            let s_i = (&delta_x_i.transpose() * &a_i_mat_inv * &delta_x_i)[0];
            let s_i_inv = 1.0 / s_i;
            let c_1_mat = &a_i_mat_inv*(DMatrix::identity(n, n) - s_i_inv*&delta_x_i*&delta_x_i.transpose()*&a_i_mat_inv)*2.0*&delta_x_mat;
            let c_2_mat = s_i_inv*&a_i_mat_inv*&delta_x_i;
            c_1_mats.push(c_1_mat);
            c_2_mats.push(c_2_mat);
        }

        Self {
            delta_f_mat_t: Arc::new(Mutex::new(DMatrix::zeros(n, m))),
            delta_x_mat,
            c_1_mats,
            c_2_mats,
            i: Arc::new(Mutex::new(0)),
        }
    }
}

pub struct DerivativeMethodClassWASP;
impl DerivativeMethodClass for DerivativeMethodClassWASP {
    type DerivativeMethod = WASP;
}

#[derive(Clone)]
pub struct WASP {
    pub cache: WASPCache2,
    pub d_theta: f64,
    pub d_l: f64,
    pub num_f_calls: Arc<Mutex<usize>>
}
impl WASP {
    pub fn new(n: usize, m: usize, d_theta: f64, d_l: f64, orthonormalize: bool) -> Self {
        Self {
            cache: WASPCache2::new(n, m, orthonormalize),
            d_theta,
            d_l,
            num_f_calls: Arc::new(Mutex::new(0)),
        }
    }

    pub fn get_num_f_calls(&self) -> usize {
        self.num_f_calls.lock().unwrap().clone()
    }
}
impl DerivativeMethodTrait for WASP {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut num_f_calls = self.num_f_calls.lock().unwrap();
        *num_f_calls = 0;
        let f_x_k_vec = function.call(inputs, false);
        let f_x_k = DVector::from_column_slice(&f_x_k_vec);
        let x_k = DVector::from_column_slice(inputs);
        *num_f_calls += 1;

        let mut return_result;

        loop {
            let mut i = self.cache.i.lock().unwrap();

            let epsilon = 0.00001;
            let c_1_mat = &self.cache.c_1_mats[*i];
            let c_2_mat = &self.cache.c_2_mats[*i];
            let delta_x_i = DVector::from_column_slice(self.cache.delta_x_mat.column(*i).as_slice());
            let f_x_k_delta = DVector::from_column_slice(&function.call((&x_k + epsilon * &delta_x_i).as_slice(), true));
            *num_f_calls += 1;
            let delta_f_i = (f_x_k_delta - &f_x_k) / epsilon;
            let mut delta_f_hat_t = self.cache.delta_f_mat_t.lock().unwrap();
            let delta_f_i_hat = DVector::from_column_slice(delta_f_hat_t.row(*i).transpose().as_slice());

            return_result = close_enough2(&delta_f_i, &delta_f_i_hat, self.d_theta, self.d_l);

            delta_f_hat_t.set_row(*i, &delta_f_i.transpose());
            let d_t = c_1_mat * &*delta_f_hat_t + c_2_mat * &delta_f_i.transpose();
            *delta_f_hat_t = &self.cache.delta_x_mat.transpose() * &d_t;

            *i = (*i + 1) % inputs.len();

            if return_result {
                return (f_x_k_vec, d_t.transpose());
            }
        }
    }
}

pub struct DerivativeMethodClassWASPNec;
impl DerivativeMethodClass for DerivativeMethodClassWASPNec {
    type DerivativeMethod = WASPNec;
}
#[derive(Clone)]
pub struct WASPNec {
    pub cache: WASPCache,
    pub first_call: Arc<Mutex<bool>>,
    pub num_f_calls: Arc<Mutex<usize>>
}
impl WASPNec {
    pub fn new(n: usize, m: usize, alpha: f64, orthonormalize: bool) -> Self {
        Self {
            cache: WASPCache::new(n, m, alpha, orthonormalize),
            first_call: Arc::new(Mutex::new(true)),
            num_f_calls: Arc::new(Mutex::new(0)),
        }
    }

    pub fn get_num_f_calls(&self) -> usize {
        self.num_f_calls.lock().unwrap().clone()
    }
}
impl DerivativeMethodTrait for WASPNec {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut num_f_calls = self.num_f_calls.lock().unwrap();
        *num_f_calls = 0;
        let f_x_k_vec = function.call(inputs, false);
        let f_x_k = DVector::from_column_slice(&f_x_k_vec);
        *num_f_calls += 1;

        let mut first_call = self.first_call.lock().unwrap();
        if *first_call {
            let epsilon = 0.00001;
            let x_k = DVector::from_column_slice(inputs);
            let n = inputs.len();
            for i in 0..n {
                let delta_x_i = DVector::from_column_slice(self.cache.delta_x_mat.column(i).as_slice());
                let f_x_k_delta = DVector::from_column_slice(&function.call((&x_k + epsilon*&delta_x_i).as_slice(), true));
                let delta_f_i = (f_x_k_delta - &f_x_k) / epsilon;
                let mut delta_f_hat_t = self.cache.delta_f_mat_t.lock().unwrap();
                delta_f_hat_t.set_row(i, &delta_f_i.transpose());
            }
            *first_call = false;
        }

        let d_t = wasp_projection(function, &f_x_k, inputs, &self.cache);
        *num_f_calls += 1;
        let mut i = self.cache.i.lock().unwrap();
        *i = (*i + 1) % inputs.len();

        return (f_x_k_vec, d_t.transpose());
    }
}

pub struct DerivativeMethodClassWASPEc;
impl DerivativeMethodClass for DerivativeMethodClassWASPEc {
    type DerivativeMethod = WASPEc;
}

#[derive(Clone)]
pub struct WASPEc {
    pub cache_a: WASPCache,
    pub cache_b: WASPCache,
    pub l: usize,
    pub d_theta: f64,
    pub num_f_calls: Arc<Mutex<usize>>
}
impl WASPEc {
    pub fn new(n: usize, m: usize, alpha: f64, orthonormalize: bool, l: usize, d_theta: f64) -> Self {
        assert!(l <= m);

        Self {
            cache_a: WASPCache::new(n, m, alpha, orthonormalize),
            cache_b: WASPCache::new(n, m, alpha, orthonormalize),
            l,
            d_theta,
            num_f_calls: Arc::new(Mutex::new(0)),
        }
    }

    pub fn get_num_f_calls(&self) -> usize {
        self.num_f_calls.lock().unwrap().clone()
    }
}
impl DerivativeMethodTrait for WASPEc {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut num_f_calls = self.num_f_calls.lock().unwrap();
        *num_f_calls = 0;
        let f_x_k_vec = function.call(inputs, false);
        let f_x_k = DVector::from_column_slice(&f_x_k_vec);
        *num_f_calls += 1;

        loop {
            let d_a_t = wasp_projection(function, &f_x_k, inputs, &self.cache_a);
            let d_b_t = wasp_projection(function, &f_x_k, inputs, &self.cache_b);
            *num_f_calls += 2;
            let mut i_a = self.cache_a.i.lock().unwrap();
            let mut i_b = self.cache_b.i.lock().unwrap();
            *i_a = (*i_a + 1) % inputs.len();
            *i_b = (*i_b + 1) % inputs.len();

            if close_enough(&d_a_t, &d_b_t, self.l, function.num_outputs(), self.d_theta) {
                return (f_x_k_vec, (d_a_t.transpose() + d_b_t.transpose()) * 0.5);
            }
        }
    }
}

*/

#[derive(Clone)]
pub struct SPSA;
impl SPSA {
    pub fn new() -> Self { Self {} }
}
impl DerivativeMethodTrait for SPSA {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let f0 = function.call(inputs, false);

        let mut rng = rand::thread_rng();

        let epsilon = 0.00000001;

        let r: Vec<f64> = (0..inputs.len()).into_iter().map(|_x| rng.gen_range(-1.0..=1.0)).collect();
        let x = DVector::from_column_slice(inputs);
        let delta_k = DVector::from_column_slice(&r);
        let xpos = &x + epsilon*&delta_k;
        let xneg = &x - epsilon*&delta_k;
        let fpos = DVector::from_column_slice(&function.call(xpos.as_slice(), false));
        let fneg = DVector::from_column_slice(&function.call(xneg.as_slice(), false));
        let v = (&fpos - &fneg) / (2.0 * epsilon);
        let delta_k_inverse = DVector::from_column_slice(&delta_k.iter().map(|x| 1.0 / *x).collect::<Vec<f64>>());
        let out = &v * &delta_k_inverse.transpose();

        (f0, out)
    }
}

pub struct DerivativeMethodClassAlwaysZero;
impl DerivativeMethodClass for DerivativeMethodClassAlwaysZero {
    type DerivativeMethod = DerivativeAlwaysZero;
}

#[derive(Clone)]
pub struct DerivativeAlwaysZero;
impl DerivativeAlwaysZero {
    pub fn new() -> Self {
        Self {}
    }
}
impl DerivativeMethodTrait for DerivativeAlwaysZero {
    type T = f64;

    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, _inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let num_outputs = function.num_outputs();
        let num_inputs = function.num_inputs();
        (vec![0.0; num_outputs], DMatrix::from_vec(num_outputs, num_inputs, vec![0.0; num_outputs*num_inputs]))
    }
}