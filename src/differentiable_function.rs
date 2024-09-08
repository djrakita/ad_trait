use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
// use apollo_rust_linalg::{ApolloDMatrixTrait, ApolloDVectorTrait, M, V};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use crate::{AD};
use crate::forward_ad::adfn::adfn;
use crate::forward_ad::ForwardADTrait;
use crate::reverse_ad::adr::{adr, GlobalComputationGraph};
use crate::simd::f64xn::f64xn;

/*
pub trait DifferentiableFunctionTrait {
    type ArgsType<'a, T: AD>;

    fn call<'a, T1: AD>(inputs: &[T1], args: &Self::ArgsType<'a, T1>) -> Vec<T1>;
    fn num_inputs<T1: AD>(args: &Self::ArgsType<'_, T1>) -> usize;
    fn num_outputs<T1: AD>(args: &Self::ArgsType<'_, T1>) -> usize;
    fn derivative<'a, E: DerivativeMethodTrait>(inputs: &[f64], args: &Self::ArgsType<'a, E::T>, derivative_method_data: &E::DerivativeMethodData) -> (Vec<f64>, DMatrix<f64>) {
        E::derivative::<Self>(inputs, args, derivative_method_data)
    }
}
*/

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

/*
pub trait DifferentiableFunctionClass2 {
    type FunctionType<'a, T: AD> : DifferentiableFunctionTrait2<'a, T>;
}
pub trait DifferentiableFunctionTrait2<'a, T: AD> : AsAny {
    fn name(&self) -> &str;
    fn call_raw(&self, inputs: &[T], freeze: bool) -> Vec<T>;
    fn call(&self, output_from_call_raw: Vec<T>) -> Vec<T> {
        output_from_call_raw
    }
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
}
*/

/*
pub trait DerivativeMethodTrait {
    type T: AD;
    type DerivativeMethodData;

    fn derivative<'a, D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, derivative_method_data: &Self::DerivativeMethodData) -> (Vec<f64>, DMatrix<f64>);

    #[inline(always)]
    fn t_is_f64() -> bool {
        match Self::T::ad_num_type() {
            ADNumType::F64 => { true }
            _ => { false }
        }
    }
}
*/

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

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
pub struct DifferentiableBlock<D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> {
    derivative_method: E,
    phantom_data: PhantomData<D>
}
impl<D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> DifferentiableBlock<D, E> {
    pub fn new() -> Self {
        Self {
            derivative_method: E::new(),
            phantom_data: Default::default()
        }
    }
    pub fn call<T1: AD>(&self, inputs: &[T1], args: &D::ArgsType<T1>) -> Vec<T1> {
        D::call(inputs, args)
    }
    pub fn derivative(&self, inputs: &[f64], args: &D::ArgsType<E::T>) -> (Vec<f64>, DMatrix<f64>) {
        self.derivative_method.derivative::<D>(inputs, args)
    }
    pub fn derivative_data(&self) -> &E {
        &self.derivative_method
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
pub struct FiniteDifferencing;
impl DerivativeMethodTrait for FiniteDifferencing {
    type T = f64;
    type DerivativeMethodData = ();

    fn derivative<D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, _derivative_method_data: &()) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);

        let h = 0.0000001;

        let x0 = inputs.to_vec();
        let f0 = D::call(&x0, args);

        for col_idx in 0..num_inputs {
            let mut xh = x0.clone();
            xh[col_idx] += h;
            let fh = D::call(&xh, args);
            for row_idx in 0..num_outputs {
                out_derivative[(row_idx, col_idx)] = (fh[row_idx] - f0[row_idx]) / h;
            }
        }

        (f0, out_derivative)
    }
}
*/

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

/*
pub struct FiniteDifferencing3;
impl FiniteDifferencing3 {
    pub fn new() -> Self {
        Self {}
    }
}
impl<'a> DerivativeMethodTrait3<'a> for FiniteDifferencing3 {
    type T = f64;

    fn derivative(&self, inputs: &[f64], function: &Box<dyn DifferentiableFunctionTrait2<'a, Self::T> + 'a>) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = function.num_outputs();
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);

        let h = 0.0000001;

        let x0 = inputs.to_vec();
        // let f0 = D::call(&x0, args);
        let f0 = function.call(&x0);

        for col_idx in 0..num_inputs {
            let mut xh = x0.clone();
            xh[col_idx] += h;
            // let fh = D::call(&xh, args);
            let fh = function.call(&xh);
            for row_idx in 0..num_outputs {
                out_derivative[(row_idx, col_idx)] = (fh[row_idx] - f0[row_idx]) / h;
            }
        }

        (f0, out_derivative)
    }
}
*/

/*
pub struct ReverseAD;
impl DerivativeMethodTrait for ReverseAD {
    type T = adr;
    type DerivativeMethodData = ();

    fn derivative<D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, _derivative_method_data: &()) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);

        GlobalComputationGraph::get().reset();

        let mut inputs_ad = vec![];
        for input in inputs.iter() {
            inputs_ad.push(adr::new_variable(*input, false));
        }

        let f = D::call(&inputs_ad, args);
        assert_eq!(f.len(), num_outputs);
        let out_value = f.iter().map(|x| x.value()).collect();

        for row_idx in 0..num_outputs {
            if f[row_idx].is_constant() {
                for col_idx in 0..num_inputs {
                    out_derivative[(row_idx, col_idx)] = 0.0;
                }
            }
            else {
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
*/

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

        // let f = D::call(&inputs_ad, args);
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

/*
pub struct ForwardAD;
impl DerivativeMethodTrait for ForwardAD {
    type T = adfn<1>;
    type DerivativeMethodData = ();

    fn derivative<D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, _derivative_method_data: &()) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
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

            let f = D::call(&inputs_ad, args);
            assert_eq!(f.len(), num_outputs);
            for (row_idx, res) in f.iter().enumerate() {
                if out_value.len() < num_outputs {
                    out_value.push(res.value);
                }
                out_derivative[(row_idx, col_idx)] = res.tangent[0];
            }
        }

        (out_value, out_derivative)
    }
}
*/

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

/*
pub struct ForwardADMulti<A: AD + ForwardADTrait> {
    p: PhantomData<A>
}
impl<A: AD + ForwardADTrait> DerivativeMethodTrait for ForwardADMulti<A> {
    type T = A;
    type DerivativeMethodData = ();

    fn derivative<D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, _derivative_method_data: &()) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        let mut curr_idx = 0;

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

            let f = D::call(&inputs_ad, args);
            assert_eq!(f.len(), num_outputs);

            for (row_idx, res) in f.iter().enumerate() {
                if out_value.len() < num_outputs {
                    out_value.push(res.value());
                }
                let curr_tangent = res.tangent_as_vec();
                'l3: for i in 0..k {
                    if curr_idx + i >= num_inputs { break 'l3; }
                    // out_derivative[(row_idx, curr_idx+i)] = res.tangent[i];
                    out_derivative[(row_idx, curr_idx+i)] = curr_tangent[i];
                }
            }

            curr_idx += k;
            if curr_idx > num_inputs { break 'l1; }
        }

        return (out_value, out_derivative)
    }
}
*/

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

/*
pub struct FiniteDifferencingMulti<const K: usize>;
impl<const K: usize> DerivativeMethodTrait for FiniteDifferencingMulti<K> {
    type T = f64xn<K>;
    type DerivativeMethodData = ();

    fn derivative<D: DifferentiableFunctionTrait + ?Sized>(inputs: &[f64], args: &D::ArgsType<'_, Self::T>, _derivative_method_data: &()) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
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

            let f = D::call(&inputs_ad, args);
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
*/

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


pub struct DerivativeMethodClassWASP;
impl DerivativeMethodClass for DerivativeMethodClassWASP {
    type DerivativeMethod = WASP;
}

#[derive(Clone)]
pub struct WASP {
    n: usize,
    m: usize,
    lagrange_multiplier_inf_norm_cutoff: f64,
    p_matrices: Vec<DMatrix<f64>>,
    delta_x_mat: DMatrix<f64>,
    #[allow(unused)]
    delta_x_mat_t: DMatrix<f64>,
    delta_f_hat_mat_t: Arc<Mutex<DMatrix<f64>>>,
    #[allow(unused)]
    r: usize,
    i: Arc<Mutex<usize>>,
    num_f_calls: Arc<Mutex<usize>>,
    max_iters: usize,
}
impl WASP {
    pub fn new(num_inputs: usize, num_outputs: usize, lagrange_multiplier_inf_norm_cutoff: f64, max_iters: usize) -> Self {
        assert!(max_iters >= 1);

        let n = num_inputs;
        let m = num_outputs;

        let r = n + 5;

        let mut rng = rand::thread_rng();
        let mut mm = DMatrix::zeros(n, r);

        for i in 0..n {
            for j in 0..r {
                mm[(i, j)] = rng.gen_range(-1.0..=1.0);
            }
        }

        let delta_x_mat = mm;
        let delta_x_mat_t = delta_x_mat.transpose();
        let delta_f_hat_mat_t = DMatrix::zeros(r, m);

        let tmp = 2.0 * (&delta_x_mat * &delta_x_mat_t);
        let mut p_matrices = vec![];
        for i in 0..r {
            let mut p = DMatrix::zeros(n+1, n+1);
            p.view_mut((0,0), (n,n)).copy_from(&tmp);
            let delta_x_i = delta_x_mat.column(i);
            p.view_mut((n,0), (1,n)).copy_from_slice(delta_x_i.as_slice());
            p.view_mut((0,n), (n,1)).copy_from_slice((-delta_x_i).as_slice());
            p_matrices.push(p.try_inverse().expect("error"));
        }

        Self {
            n,
            m,
            lagrange_multiplier_inf_norm_cutoff,
            p_matrices,
            delta_x_mat,
            delta_x_mat_t,
            delta_f_hat_mat_t: Arc::new(Mutex::new(delta_f_hat_mat_t)),
            r,
            i: Arc::new(Mutex::new(0)),
            num_f_calls: Arc::new(Mutex::new(0)),
            max_iters,
        }
    }

    #[inline(always)]
    pub fn get_num_f_calls(&self) -> usize {
        *self.num_f_calls.lock().expect("error")
    }

    #[inline(always)]
    fn derivative_internal<D: DifferentiableFunctionTrait<f64> + ?Sized>(&self, inputs: &[f64], function: &D, _recursive_call: bool, f0: Option<DVector<f64>>, delta_f_hat_mat_t: &mut DMatrix<f64>, i: &mut usize, num_f_calls: &mut usize, curr_count: usize) -> (Vec<f64>, DMatrix<f64>) {
        let n = self.n;
        let m = self.m;

        let delta_x_i = self.delta_x_mat.column(*i);

        let f0 = match f0 {
            None => { *num_f_calls += 1; DVector::from_column_slice(&function.call(inputs, false))  }
            Some(f0) => { f0 }
        };

        let p = 0.00001;
        let x = DVector::from_column_slice(inputs);
        let xh = x + (p * delta_x_i);
        let fh = DVector::from_column_slice(&function.call(&xh.as_slice(), true));
        *num_f_calls += 1;

        let delta_f_i = (fh - &f0) / p;
        delta_f_hat_mat_t.view_mut((*i, 0), (1, m)).copy_from_slice(delta_f_i.as_slice());

        // let start = Instant::now();
        let a_mat = 2.0 * &self.delta_x_mat * &*delta_f_hat_mat_t;
        // println!("{:?}", start.elapsed());

        let mut b_mat = DMatrix::zeros(n+1, m);
        b_mat.view_mut((0,0), (n, m)).copy_from(&a_mat);
        b_mat.view_mut((n,0), (1, m)).copy_from_slice(delta_f_i.as_slice());

        let p_mat_i = &self.p_matrices[*i];
        let c_mat = p_mat_i * b_mat;
        let d_mat_t = c_mat.view((0,0), (n, m));
        let lagrange_multiplier_row = c_mat.view((n,0), (1, m));
        let inf_norm = lagrange_multiplier_row.iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).expect("error").abs();

        *i = (*i + 1) % self.p_matrices.len();

        return if inf_norm > self.lagrange_multiplier_inf_norm_cutoff && !(curr_count >= self.max_iters) {
            self.derivative_internal(inputs, function, true, Some(f0), delta_f_hat_mat_t, i, num_f_calls, curr_count + 1)
        } else {
            *delta_f_hat_mat_t = &self.delta_x_mat_t * d_mat_t;
            (f0.as_slice().to_vec(), d_mat_t.transpose())
        }
    }
}
impl DerivativeMethodTrait for WASP {
    type T = f64;

    #[inline(always)]
    fn derivative<D: DifferentiableFunctionTrait<Self::T> + ?Sized>(&self, inputs: &[f64], function: &D) -> (Vec<f64>, DMatrix<f64>) {
        let mut i = self.i.lock().expect("error");
        let mut delta_f_hat_mat_t = self.delta_f_hat_mat_t.lock().expect("error");
        let mut num_f_calls = self.num_f_calls.lock().expect("error");
        *num_f_calls = 0;

        return self.derivative_internal::<D>(inputs, function, false, None, &mut *delta_f_hat_mat_t, &mut *i, &mut *num_f_calls, 1);
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

/*
pub struct Ricochet<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    ricochet_data: RicochetData<T>,
    ricochet_termination: RicochetTermination,
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> Ricochet<D, T> {
    pub fn new(args: &D::U<T>, ricochet_termination: RicochetTermination) -> Self {

        let num_inputs = D::num_inputs(args);
        let num_outputs = D::num_outputs(args);

        Self {
            ricochet_data: RicochetData::new(num_inputs, num_outputs, T::tangent_size(), -1., 1.0, None),
            ricochet_termination,
            p: PhantomData::default()
        }
    }
    pub fn ricochet_data(&self) -> &RicochetData<T> {
        &self.ricochet_data
    }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeTrait<D, T> for Ricochet<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        let mut t = self.ricochet_termination.map_to_new_internal();

        loop {
            let curr_affine_space_idx = self.ricochet_data.curr_affine_space_idx.read().unwrap().clone();
            let tangent_transpose_pseudoinverse_matrix = &self.ricochet_data.tangent_transpose_pseudoinverse_matrices[curr_affine_space_idx];
            let z_chain_matrix = &self.ricochet_data.z_chain_matrices[curr_affine_space_idx];
            let previous_derivative = self.ricochet_data.previous_derivative.read().unwrap().clone();
            let previous_derivative_transpose = self.ricochet_data.previous_derivative_transpose.read().unwrap().clone();

            let mut inputs_ad = self.ricochet_data.input_templates[curr_affine_space_idx].clone();
            assert_eq!(inputs_ad.len(), inputs.len());
            inputs.iter().zip(inputs_ad.iter_mut()).for_each(|(x, y)| y.set_value(*x));

            let f = D::call(&inputs_ad, args);

            // let directional_derivative = &fh - &f0;
            // let directional_derivative_transpose = directional_derivative.transpose();

            let mut directional_derivative_transpose = DMatrix::zeros(T::tangent_size(), f.len());
            for (col_idx, output) in f.iter().enumerate() {
                let tangent = output.tangent_as_vec();
                for (row_idx, t) in tangent.iter().enumerate() {
                    directional_derivative_transpose[(row_idx, col_idx)] = *t;
                }
            }

            let minimum_norm_solution_transpose = tangent_transpose_pseudoinverse_matrix * &directional_derivative_transpose;

            let new_derivative_transpose = &minimum_norm_solution_transpose + z_chain_matrix * (&previous_derivative_transpose - &minimum_norm_solution_transpose);
            let new_derivative = new_derivative_transpose.transpose();

            self.ricochet_data.update_previous_derivative(&new_derivative);
            self.ricochet_data.increment_curr_affine_space_idx();

            let terminate = t.terminate(&previous_derivative, &new_derivative);

            if terminate {
                let mut output_value = vec![];
                for ff in f { output_value.push(ff.value()); }
                return (output_value, new_derivative)
            }
        }
    }
}

pub struct SpiderForwardAD<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    spider_data: SpiderData,
    input_templates: Vec<Vec<T>>,
    decay_multiple: f64,
    p: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> SpiderForwardAD<D, T> {
    pub fn new(args: &D::U<T>, decay_multiple: f64) -> Self {
        let num_inputs = D::num_inputs(args);
        let num_outputs = D::num_outputs(args);

        assert!(decay_multiple > 0.0);
        assert!(decay_multiple < 1.0);

        let spider_data = SpiderData::new(num_inputs, num_outputs, T::tangent_size(), -1.0, 1.0);

        let mut input_templates = vec![];

        let t_mat_affines = &spider_data.t_mat_affines;

        for t_mat_affine in t_mat_affines {
            let mut curr_input_template = vec![];
            t_mat_affine.row_iter().for_each(|x| {
                let mut curr_input = T::constant(0.0);
                x.iter().enumerate().for_each(|(i, y)| curr_input.set_tangent_value(i, *y) );
                curr_input_template.push(curr_input);
            });
            input_templates.push(curr_input_template);
        }

        Self {
            spider_data,
            input_templates,
            decay_multiple,
            p: Default::default()
        }
    }
    pub fn spider_data(&self) -> &SpiderData {
        &self.spider_data
    }
    pub fn input_templates(&self) -> &Vec<Vec<T>> { &self.input_templates }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeTrait<D, T> for SpiderForwardAD<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        let curr_affine_space_idx = self.spider_data.curr_affine_space_idx();

        let mut inputs_ad = self.input_templates[curr_affine_space_idx].clone();
        inputs_ad.iter_mut().zip(inputs.iter()).for_each(|(x, y)| x.set_value(*y) );

        let res = D::call(&inputs_ad, args);
        let output_value: Vec<f64> = res.iter().map(|x| x.value() ).collect();

        let mut f_l = DMatrix::<f64>::zeros(D::num_outputs(args), T::tangent_size());
        res.iter().enumerate().for_each(|(row_idx, x)| {
            let tangent_vec = x.tangent_as_vec();
            tangent_vec.iter().enumerate().for_each(|(col_idx, y)| {
                f_l[(row_idx, col_idx)] = *y;
            });
        });

        // println!(" >>> {}", f_l);
        self.spider_data.update_f_mat(&f_l);
        self.spider_data.update_w(self.decay_multiple);
        // self.spider_data.print_w();

        let f_l = &f_l;
        let t_l_pinv = &self.spider_data.t_mat_affine_pinvs[curr_affine_space_idx];
        let _z_l_chain = &self.spider_data.t_mat_affine_z_chains[curr_affine_space_idx];
        let w_mat = &self.spider_data.get_w_mat();
        let f_mat = &*self.spider_data.f_mat.read().unwrap();
        let t_mat_pinv = &self.spider_data.t_mat_pinv;
        // println!(" >>>> {}", f_mat);
        // println!("{}", w_mat);
        // let w_mat = DMatrix::<f64>::identity(4, 4);
        println!(" >>> {}", w_mat);

        let _f_l_t_l_pinv = f_l * t_l_pinv;
        let f_w_mat_t_mat_pinv = f_mat * w_mat * t_mat_pinv;

        // let d = &f_l_t_l_pinv + ( &f_w_mat_t_mat_pinv - &f_l_t_l_pinv ) * z_l_chain;
        let d = f_w_mat_t_mat_pinv;

        self.spider_data.increment_curr_affine_space_idx();

        return (output_value, d)
    }
}

pub struct Spider2ForwardAD<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    spider_data: Spider2Data,
    input_templates: Vec<Vec<T>>,
    project_onto_affine_space: bool,
    p: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> Spider2ForwardAD<D, T> {
    pub fn new(args: &D::U<T>, decay_multiple: f64, project_onto_affine_space: bool) -> Self {
        let num_inputs = D::num_inputs(args);
        let num_outputs = D::num_outputs(args);

        let spider_data = Spider2Data::new(num_inputs, num_outputs, T::tangent_size(), -1.0, 1.0, decay_multiple);

        let mut input_templates = vec![];

        let t_mat_affines = &spider_data.t_mat_affines;

        for t_mat_affine in t_mat_affines {
            let mut curr_input_template = vec![];
            t_mat_affine.row_iter().for_each(|x| {
                let mut curr_input = T::constant(0.0);
                x.iter().enumerate().for_each(|(i, y)| curr_input.set_tangent_value(i, *y) );
                curr_input_template.push(curr_input);
            });
            input_templates.push(curr_input_template);
        }

        Self {
            spider_data,
            input_templates,
            project_onto_affine_space,
            p: Default::default()
        }
    }
    pub fn spider_data(&self) -> &Spider2Data {
        &self.spider_data
    }
    pub fn input_templates(&self) -> &Vec<Vec<T>> { &self.input_templates }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeTrait<D, T> for Spider2ForwardAD<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        let curr_affine_space_idx = self.spider_data.curr_affine_space_idx();

        let mut inputs_ad = self.input_templates[curr_affine_space_idx].clone();
        inputs_ad.iter_mut().zip(inputs.iter()).for_each(|(x, y)| x.set_value(*y) );

        let res = D::call(&inputs_ad, args);
        let output_value: Vec<f64> = res.iter().map(|x| x.value() ).collect();

        let mut f_l = DMatrix::<f64>::zeros(D::num_outputs(args), T::tangent_size());
        res.iter().enumerate().for_each(|(row_idx, x)| {
            let tangent_vec = x.tangent_as_vec();
            tangent_vec.iter().enumerate().for_each(|(col_idx, y)| {
                f_l[(row_idx, col_idx)] = *y;
            });
        });

        self.spider_data.update_f_mat(&f_l);

        let f_mat = &*self.spider_data.f_mat.read().unwrap();
        let w_t_chain = if self.spider_data.first_pass() {
            &self.spider_data.w_t_chains_first_pass[curr_affine_space_idx]
        } else {
            &self.spider_data.w_t_chains[curr_affine_space_idx]
        };

        let d_wls = f_mat * w_t_chain;

        if !self.project_onto_affine_space {
            self.spider_data.increment_curr_affine_space_idx();
            return (output_value, d_wls);
        }

        let f_l = &f_l;
        let t_l_pinv = &self.spider_data.t_mat_affine_pinvs[curr_affine_space_idx];
        let t_l = &self.spider_data.t_mat_affines[curr_affine_space_idx];
        let z_l_chain = &self.spider_data.t_mat_affine_transpose_z_chains[curr_affine_space_idx];

        let d_mns = f_l * t_l_pinv;

        let d = &d_mns + (&d_wls - &d_mns) * z_l_chain;
        // let d_t = &d_mns.transpose() + z_l_chain*(&d_wls.transpose() - &d_mns.transpose());
        // let d = d_t.transpose();

        println!("{:?}", d);
        println!("{:?}", &t_l);
        println!("1 >>> {}", &d*t_l);
        println!("2 >>> {}", &f_l);

        self.spider_data.increment_curr_affine_space_idx();

        return (output_value, d);
    }
}
*/
/*
pub struct FlowForwardAD<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    flow_data: FlowData,
    input_templates: Vec<Vec<T>>,
    p: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> FlowForwardAD<D, T> {
    pub fn new(args: &D::U<T>, decay_multiple: f64) -> Self {
        assert!(T::tangent_size() > 1);

        let num_inputs = D::num_inputs(args);
        let num_outputs = D::num_outputs(args);

        let flow_data = FlowData::new(num_inputs, num_outputs, T::tangent_size()-1, -1.0, 1.0, decay_multiple);

        let mut input_templates = vec![];

        let t_mat_affines = &flow_data.t_mat_affines;

        for t_mat_affine in t_mat_affines {
            let mut curr_input_template = vec![];
            t_mat_affine.row_iter().for_each(|x| {
                let mut curr_input = T::constant(0.0);
                x.iter().enumerate().for_each(|(i, y)| curr_input.set_tangent_value(i, *y) );
                curr_input_template.push(curr_input);
            });
            input_templates.push(curr_input_template);
        }

        Self {
            flow_data,
            input_templates,
            p: Default::default()
        }
    }
    pub fn flow_data(&self) -> &FlowData {
        &self.flow_data
    }
    pub fn input_templates(&self) -> &Vec<Vec<T>> { &self.input_templates }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeTrait<D, T> for FlowForwardAD<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        let curr_affine_space_idx = self.flow_data.curr_affine_space_idx();

        let tangent_size = T::tangent_size();

        let mut inputs_ad = self.input_templates[curr_affine_space_idx].clone();
        inputs_ad.iter_mut().zip(inputs.iter()).for_each(|(x, y)| x.set_value(*y) );

        let mut rng = thread_rng();
        let mut tangent_test = DVector::<f64>::zeros(D::num_inputs(&args));
        tangent_test.iter_mut().for_each(|x| *x = rng.gen_range(-1.0..1.0) );
        inputs_ad.iter_mut().zip(tangent_test.iter()).for_each(|(x, y)| x.set_tangent_value(tangent_size-1, *y));

        let res = D::call(&inputs_ad, args);
        let output_value: Vec<f64> = res.iter().map(|x| x.value() ).collect();

        let mut f_l = DMatrix::<f64>::zeros(D::num_outputs(args), tangent_size-1);
        let mut f_t = DVector::<f64>::zeros(D::num_outputs(args));
        res.iter().enumerate().for_each(|(row_idx, x)| {
            let tangent_vec = x.tangent_as_vec();
            tangent_vec.iter().enumerate().for_each(|(col_idx, y)| {
                if col_idx < tangent_size-1 {
                    f_l[(row_idx, col_idx)] = *y;
                } else {
                    f_t[row_idx] = *y;
                }
            });
        });

        self.flow_data.update_f_mat(&f_l, curr_affine_space_idx);

        let f_mat = &*self.flow_data.f_mat.read().unwrap();
        let w_t_chain = if self.flow_data.first_pass() {
            &self.flow_data.w_t_chains_first_pass[curr_affine_space_idx]
        } else {
            &self.flow_data.w_t_chains[curr_affine_space_idx]
        };

        let d_wls = f_mat * w_t_chain;

        println!("{}", d_wls);
        let directional_derivative_test = &d_wls*&tangent_test;
        println!("{:?}, {:?}", directional_derivative_test, f_t);
        println!("{:?}", d_wls.transpose().dot(&DVector::from_vec(vec![1.,1.,1.,1.])));

        todo!()
    }
}
*/
/*
pub struct FlowFiniteDiff<D: DifferentiableBlockTrait> {
    flow_data: FlowData,
    num_test_samples: usize,
    num_function_calls_on_previous_derivative: RwLock<usize>,
    max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock<f64>,
    max_allowable_error_dis_from_1: f64,
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> FlowFiniteDiff<D> {
    pub fn new(args: &D::U<f64>, decay_multiple: f64, num_test_samples: usize, max_allowable_error_dis_from_1: f64) -> Self {
        assert!(num_test_samples > 0);

        let flow_data = FlowData::new(D::num_inputs(args), D::num_outputs(args), 1, -0.000001, 0.000001, decay_multiple);

        Self {
            flow_data,
            num_test_samples,
            num_function_calls_on_previous_derivative: RwLock::new(0),
            max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock::new(0.0),
            max_allowable_error_dis_from_1,
            p: Default::default()
        }
    }
    pub fn num_function_calls_on_previous_derivative(&self) -> usize {
        *self.num_function_calls_on_previous_derivative.read().unwrap()
    }
    pub fn max_test_error_ratio_dis_from_1_on_previous_derivative(&self) -> f64 {
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.read().unwrap()
    }
}
impl<D: DifferentiableBlockTrait> DerivativeTrait<D, f64> for FlowFiniteDiff<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U<f64>) -> (Vec<f64>, DMatrix<f64>) {
        *self.num_function_calls_on_previous_derivative.write().unwrap() = 0;
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = 0.0;

        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);

        let f0 = D::call(inputs, args);
        *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
        let f0_dvec = DVector::from_vec(f0.clone());

        let mut rng = thread_rng();

        // delta x values for tests.  these are n x 1 vectors.
        let mut test_perturbations = vec![];
        for _ in 0..self.num_test_samples {
            let mut test_tangent = vec![];

            for _ in 0..num_inputs { test_tangent.push(rng.gen_range(-0.000001..0.000001)) }

            test_perturbations.push(test_tangent);
        }

        // delta x values for tests.  these are n x 1 vectors
        let test_perturbations_dvecs: Vec<DVector<f64>> = test_perturbations.iter().map(|x| DVector::from_column_slice(x)).collect();

        // ground truth directional derivatives for tests.  these are m x 1 vectors.
        let mut test_directional_derivatives = vec![];
        for p in test_perturbations {
            let xh = get_perturbed_inputs(inputs, &p);
            let fh = D::call(&xh, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            let fh_dvec = DVector::<f64>::from_vec(fh);
            let test_directional_derivative = &fh_dvec - &f0_dvec;
            test_directional_derivatives.push(test_directional_derivative);
        }

        'l1: loop {
            let curr_affine_space_idx = self.flow_data.curr_affine_space_idx();
            let t_mat_affine = &self.flow_data.t_mat_affines[curr_affine_space_idx];
            let p = t_mat_affine.as_slice();
            let xh = get_perturbed_inputs(inputs, &p);
            let fh = D::call(&xh, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            let fh_dvec = DVector::<f64>::from_vec(fh);
            let directional_derivative = &fh_dvec - &f0_dvec;
            let directional_derivative_as_dmatrix = DMatrix::from_row_slice(num_outputs, self.flow_data.affine_space_dimension, directional_derivative.as_slice());
            self.flow_data.update_f_mat(&directional_derivative_as_dmatrix, curr_affine_space_idx);

            let f_mat = &*self.flow_data.f_mat.read().unwrap();
            let w_t_chain = &self.flow_data.w_t_chains[curr_affine_space_idx];
            let d = f_mat * w_t_chain;

            let max_allowable_error_dis_from_1 = self.max_allowable_error_dis_from_1;
            let mut max_error = f64::NEG_INFINITY;
            for (test_perturbation, test_directional_derivative) in test_perturbations_dvecs.iter().zip(test_directional_derivatives.iter()) {
                let evaluation_directional_derivative = &d * test_perturbation;
                assert_eq!(evaluation_directional_derivative.ncols(), test_directional_derivative.ncols());
                assert_eq!(evaluation_directional_derivative.nrows(), test_directional_derivative.nrows());

                for (x, y) in test_directional_derivative.iter().zip(evaluation_directional_derivative.iter()) {
                    let ratio = *x / *y;
                    let error_ratio_dis_from_1 = (ratio - 1.0).abs();
                    if error_ratio_dis_from_1 > max_allowable_error_dis_from_1 {
                        self.flow_data.increment_curr_affine_space_idx();
                        continue 'l1;
                    }
                    if error_ratio_dis_from_1 > max_error { max_error = error_ratio_dis_from_1; }
                }
            }
            *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = max_error;
            return (f0, d);
        }
    }
}

pub struct FlowForwardAD<D: DifferentiableBlockTrait> {
    flow_data: FlowData,
    num_test_samples: usize,
    num_function_calls_on_previous_derivative: RwLock<usize>,
    max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock<f64>,
    max_allowable_error_dis_from_1: f64,
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> FlowForwardAD<D> {
    pub fn new(args: &D::U<f64>, decay_multiple: f64, num_test_samples: usize, max_allowable_error_dis_from_1: f64) -> Self {
        assert!(num_test_samples > 0);

        let flow_data = FlowData::new(D::num_inputs(args), D::num_outputs(args), 1, -1.0, 1.0, decay_multiple);

        Self {
            flow_data,
            num_test_samples,
            num_function_calls_on_previous_derivative: RwLock::new(0),
            max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock::new(0.0),
            max_allowable_error_dis_from_1,
            p: Default::default()
        }
    }
    pub fn num_function_calls_on_previous_derivative(&self) -> usize {
        *self.num_function_calls_on_previous_derivative.read().unwrap()
    }
    pub fn max_test_error_ratio_dis_from_1_on_previous_derivative(&self) -> f64 {
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.read().unwrap()
    }
}
impl<D: DifferentiableBlockTrait> DerivativeTrait<D, adfn<1>> for FlowForwardAD<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U<adfn<1>>) -> (Vec<f64>, DMatrix<f64>) {
        *self.num_function_calls_on_previous_derivative.write().unwrap() = 0;
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = 0.0;

        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);

        let mut rng = thread_rng();

        // delta x values for tests.  these are n x 1 vectors.
        let mut test_perturbations = vec![];
        for _ in 0..self.num_test_samples {
            let mut test_tangent = vec![];

            for _ in 0..num_inputs { test_tangent.push(rng.gen_range(-1.0..1.0)) }

            test_perturbations.push(test_tangent);
        }

        // delta x values for tests.  these are n x 1 vectors
        let test_perturbations_dvecs: Vec<DVector<f64>> = test_perturbations.iter().map(|x| DVector::from_column_slice(x)).collect();

        let mut out_values: Option<Vec<f64>> = None;

        // ground truth directional derivatives for tests.  these are m x 1 vectors.
        let mut test_directional_derivatives = vec![];
        for p in test_perturbations {
            let inputs: Vec<adfn<1>> = inputs.iter().zip(p.iter()).map(|(x, y)| {
                adfn::<1>::new(*x, [*y])
            }).collect();

            let res = D::call(&inputs, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            match out_values {
                None => { out_values = Some(recover_output_values_from_forward_ad_vec(&res)) }
                _ => {  }
            }

            let output_tangents = recover_output_tangents_from_forward_ad_vec(&res);

            // there will only be one channel here
            test_directional_derivatives.push(DVector::<f64>::from_vec(output_tangents[0].to_owned()));

            /*
            let xh = get_perturbed_inputs(inputs, &p);
            let fh = D::call(&xh, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            let fh_dvec = DVector::<f64>::from_vec(fh);
            let test_directional_derivative = &fh_dvec - &f0_dvec;
            test_directional_derivatives.push(test_directional_derivative);
            */
        }

        'l1: loop {
            let curr_affine_space_idx = self.flow_data.curr_affine_space_idx();
            let t_mat_affine = &self.flow_data.t_mat_affines[curr_affine_space_idx];
            let p = t_mat_affine.as_slice();
            // let xh = get_perturbed_inputs(inputs, &p);
            let inputs: Vec<adfn<1>> = inputs.iter().zip(p.iter()).map(|(x, y)| adfn::new(*x, [*y]) ).collect();
            let res = D::call(&inputs, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            let directional_derivative = &recover_output_tangents_from_forward_ad_vec(&res)[0];
            let directional_derivative_as_dmatrix = DMatrix::from_row_slice(num_outputs, self.flow_data.affine_space_dimension, directional_derivative.as_slice());
            self.flow_data.update_f_mat(&directional_derivative_as_dmatrix, curr_affine_space_idx);

            let f_mat = &*self.flow_data.f_mat.read().unwrap();
            let w_t_chain = &self.flow_data.w_t_chains[curr_affine_space_idx];
            let d = f_mat * w_t_chain;

            let max_allowable_error_dis_from_1 = self.max_allowable_error_dis_from_1;
            let mut max_error = f64::NEG_INFINITY;
            for (test_perturbation, test_directional_derivative) in test_perturbations_dvecs.iter().zip(test_directional_derivatives.iter()) {
                let evaluation_directional_derivative = &d * test_perturbation;
                assert_eq!(evaluation_directional_derivative.ncols(), test_directional_derivative.ncols());
                assert_eq!(evaluation_directional_derivative.nrows(), test_directional_derivative.nrows());

                for (x, y) in test_directional_derivative.iter().zip(evaluation_directional_derivative.iter()) {
                    let ratio = *x / *y;
                    let error_ratio_dis_from_1 = (ratio - 1.0).abs();
                    if error_ratio_dis_from_1 > max_allowable_error_dis_from_1 {
                        self.flow_data.increment_curr_affine_space_idx();
                        continue 'l1;
                    }
                    if error_ratio_dis_from_1 > max_error { max_error = error_ratio_dis_from_1; }
                }
            }
            *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = max_error;
            return (out_values.unwrap(), d);
        }
    }
}

pub struct FlowForwardADMulti<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    flow_data: FlowData,
    num_test_samples: usize,
    num_function_calls_on_previous_derivative: RwLock<usize>,
    max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock<f64>,
    max_allowable_error_dis_from_1: f64,
    input_templates: Vec<Vec<T>>,
    p: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> FlowForwardADMulti<D, T> {
    pub fn new(args: &D::U<T>, decay_multiple: f64, num_test_samples: usize, max_allowable_error_dis_from_1: f64) -> Self {
        assert!(num_test_samples > 0 && num_test_samples < T::tangent_size());

        let num_inputs = D::num_inputs(args);
        let num_outputs = D::num_outputs(args);

        let flow_data = FlowData::new(num_inputs, num_outputs, T::tangent_size()-num_test_samples, -1.0, 1.0, decay_multiple);

        let mut input_templates = vec![];

        let t_mat_affines = &flow_data.t_mat_affines;

        for t_mat_affine in t_mat_affines {
            let mut curr_input_template = vec![];
            t_mat_affine.row_iter().for_each(|x| {
                let mut curr_input = T::constant(0.0);
                x.iter().enumerate().for_each(|(i, y)| curr_input.set_tangent_value(i, *y) );
                curr_input_template.push(curr_input);
            });
            input_templates.push(curr_input_template);
        }

        Self {
            flow_data,
            num_test_samples,
            num_function_calls_on_previous_derivative: RwLock::new(0),
            max_test_error_ratio_dis_from_1_on_previous_derivative: RwLock::new(0.0),
            max_allowable_error_dis_from_1,
            input_templates,
            p: Default::default()
        }
    }
    pub fn input_templates(&self) -> &Vec<Vec<T>> { &self.input_templates }
    pub fn num_function_calls_on_previous_derivative(&self) -> usize {
        *self.num_function_calls_on_previous_derivative.read().unwrap()
    }
    pub fn max_test_error_ratio_dis_from_1_on_previous_derivative(&self) -> f64 {
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.read().unwrap()
    }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeTrait<D, T> for FlowForwardADMulti<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        *self.num_function_calls_on_previous_derivative.write().unwrap() = 0;
        *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = 0.0;

        let n = D::num_inputs(args);
        let m = D::num_outputs(args);

        let tangent_size = T::tangent_size();
        let num_test_samples = self.num_test_samples;
        let test_start_channel = tangent_size - num_test_samples;
        let mut rng = thread_rng();

        let mut all_test_tangents: Vec<DVector<f64>> = vec![];
        let mut all_test_directional_derivatives: Vec<DVector<f64>> = vec![];

        'l1: loop {
            let curr_affine_space_idx = self.flow_data.curr_affine_space_idx();
            let mut inputs_ad = self.input_templates[curr_affine_space_idx].clone();

            let mut curr_test_tangents = vec![];
            for _ in 0..num_test_samples {
                let curr_test_tangent: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
                curr_test_tangents.push(DVector::<f64>::from_vec(curr_test_tangent));
            }

            curr_test_tangents.iter().for_each(|x| {
               all_test_tangents.push(x.clone());
            });

            inputs_ad.iter_mut().enumerate().for_each(|(input_idx, x)| {
                x.set_value(inputs[input_idx]);
                curr_test_tangents.iter().enumerate().for_each(|(test_idx, y)| {
                    x.set_tangent_value(test_idx + test_start_channel, y[input_idx]);
                });
            });

            let res = D::call(&inputs_ad, args);
            *self.num_function_calls_on_previous_derivative.write().unwrap() += 1;
            let output_value = res.iter().map(|x| x.value()).collect();

            let mut f_block = DMatrix::<f64>::zeros(m, self.flow_data.affine_space_dimension);
            let mut test_directional_derivatives = vec![ DVector::<f64>::zeros(m); num_test_samples ];

            res.iter().enumerate().for_each(|(output_idx, x)| {
                let tangent_as_vec= x.tangent_as_vec();
                for (tangent_val_idx, tangent_val) in tangent_as_vec.iter().enumerate() {
                    if tangent_val_idx >= test_start_channel {
                        let test_idx = tangent_val_idx - test_start_channel;
                        test_directional_derivatives[test_idx][output_idx] = *tangent_val;
                    } else {
                        f_block[(output_idx, tangent_val_idx)] = *tangent_val;
                    }
                }
            });

            test_directional_derivatives.iter().for_each(|x| {
               all_test_directional_derivatives.push(x.clone());
            });

            self.flow_data.update_f_mat(&f_block, curr_affine_space_idx);

            let f_mat = &*self.flow_data.f_mat.read().unwrap();
            let w_t_chain = &self.flow_data.w_t_chains[curr_affine_space_idx];
            let d = f_mat * w_t_chain;

            let max_allowable_error_dis_from_1 = self.max_allowable_error_dis_from_1;
            let mut max_error = f64::NEG_INFINITY;
            for (test_perturbation, test_directional_derivative) in all_test_tangents.iter().zip(all_test_directional_derivatives.iter()) {
                let evaluation_directional_derivative = &d * test_perturbation;
                assert_eq!(evaluation_directional_derivative.ncols(), test_directional_derivative.ncols());
                assert_eq!(evaluation_directional_derivative.nrows(), test_directional_derivative.nrows());

                for (x, y) in test_directional_derivative.iter().zip(evaluation_directional_derivative.iter()) {
                    // println!(" test directional derivative: {}", x);
                    // println!(" evaluation directional derivative: {}", y);

                    let ratio = *x / *y;
                    // println!("ratio: {}", ratio);
                    let error_ratio_dis_from_1 = (ratio - 1.0).abs();
                    // println!("error_ratio_dis_from_1: {}", error_ratio_dis_from_1);
                    if error_ratio_dis_from_1 > max_allowable_error_dis_from_1 {
                        self.flow_data.increment_curr_affine_space_idx();
                        continue 'l1;
                    }
                    if error_ratio_dis_from_1 > max_error { max_error = error_ratio_dis_from_1; }
                }
            }
            *self.max_test_error_ratio_dis_from_1_on_previous_derivative.write().unwrap() = max_error;
            return (output_value, d);
        }
    }
}

fn get_perturbed_inputs(inputs: &[f64], perturbations: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), perturbations.len());
    return inputs.iter().zip(perturbations.iter()).map(|(x, y)| *x + *y ).collect();
}

fn recover_output_values_from_forward_ad_vec<T: AD + ForwardADTrait>(v: &Vec<T>) -> Vec<f64> {
    v.iter().map(|x| x.value() ).collect()
}

/// includes all channels
fn recover_output_tangents_from_forward_ad_vec<T: AD + ForwardADTrait>(v: &Vec<T>) -> Vec<Vec<f64>> {
    let tangent_size = T::tangent_size();
    let mut out_vec = vec![ vec![]; tangent_size ];
    v.iter().for_each(|x| {
        let tangent_as_vec = x.tangent_as_vec();
        tangent_as_vec.iter().enumerate().for_each(|(i, y)| {
            out_vec[i].push(*y);
        });
    });
    out_vec
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#[derive(Debug)]
#[allow(dead_code)]
pub struct RicochetData<T: AD + ForwardADTrait> {
    num_affine_spaces: usize,
    previous_derivative: RwLock<DMatrix<f64>>,
    previous_derivative_transpose: RwLock<DMatrix<f64>>,
    tangent_transpose_matrices: Vec<DMatrix<f64>>,
    tangent_transpose_pseudoinverse_matrices: Vec<DMatrix<f64>>,
    z_chain_matrices: Vec<DMatrix<f64>>,
    input_templates: Vec<Vec<T>>,
    curr_affine_space_idx: RwLock<usize>
}
impl<T: AD + ForwardADTrait> RicochetData<T> {
    pub fn new(num_inputs: usize, num_outputs: usize, affine_space_dimension: usize, sample_lower_bound: f64, sample_upper_bound: f64, previous_derivative: Option<&DMatrix<f64>>) -> Self {
        let num_affine_spaces = (num_inputs as f64 / affine_space_dimension as f64).ceil() as usize;

        let previous_derivative = match previous_derivative {
            None => {
                DMatrix::identity(num_outputs, num_inputs)
            }
            Some(previous_derivative) => {
                let shape = previous_derivative.shape();
                assert_eq!(num_outputs, shape.0);
                assert_eq!(num_inputs, shape.1);
                previous_derivative.clone()
            }
        };
        let previous_derivative_transpose = previous_derivative.transpose();

        let mut tangent_transpose_matrices = vec![];
        let mut tangent_transpose_pseudoinverse_matrices = vec![];
        let mut z_chain_matrices = vec![];

        let mut rng = thread_rng();
        for _ in 0..num_affine_spaces {
            let mut tangent_transpose_matrix = DMatrix::zeros(affine_space_dimension, num_inputs);
            tangent_transpose_matrix.iter_mut().for_each(|x| *x = rng.gen_range(sample_lower_bound..sample_upper_bound) );
            let tangent_transpose_matrix_faer = nalgebra_dmatrix_to_faer_mat(&tangent_transpose_matrix);

            let rank = tangent_transpose_matrix.rank(0.0);

            let tangent_transpose_pseudoinverse_matrix = tangent_transpose_matrix.clone().pseudo_inverse(0.0).unwrap();

            let mut u: Mat<f64> = Mat::zeros(affine_space_dimension,affine_space_dimension);
            let mut s: Mat<f64> = Mat::zeros(rank,1);
            let mut v: Mat<f64> = Mat::zeros(num_inputs,num_inputs);

            let mut mem = GlobalMemBuffer::new(
                compute_svd_req::<f64>(
                    tangent_transpose_matrix_faer.nrows(),
                    tangent_transpose_matrix_faer.ncols(),
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    Parallelism::None,
                    SvdParams::default(),
                )
                    .unwrap(),
            );
            let stack = DynStack::new(&mut mem);

            compute_svd(tangent_transpose_matrix_faer.as_ref(), s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), f64::EPSILON, f64::MIN_POSITIVE, Parallelism::None, stack, SvdParams::default());

            let v = faer_mat_to_nalgebra_dmatrix(&v);

            let z = v.view((0, rank), (num_inputs, num_inputs-rank));

            let z_chain_matrix = &z*(&z.transpose()*&z).try_inverse().unwrap()*&z.transpose();

            tangent_transpose_matrices.push(tangent_transpose_matrix);
            tangent_transpose_pseudoinverse_matrices.push(tangent_transpose_pseudoinverse_matrix);
            z_chain_matrices.push(z_chain_matrix);

            /*
            println!("{:?}", tangent_transpose_matrix.rank(0.0));

            let rank = tangent_transpose_matrix.rank(0.0);

            let svd = tangent_transpose_matrix.svd(true, true);
            println!(">>>{}", svd.v_t.as_ref().unwrap().transpose());
            let tmp = svd.v_t.as_ref().unwrap().transpose();
            let z = tmp.slice((rank, 0), (num_inputs - rank, affine_space_dimension)).into_owned();
            println!(" >>>> {}", z);
            */
        }

        let mut input_templates = vec![];

        for tangent_transpose_matrix in &tangent_transpose_matrices {
            let mut curr_input_template = vec![];

            for i in 0..num_inputs {
                let mut curr_input = T::zero();
                tangent_transpose_matrix.column(i).iter().enumerate().for_each(|(j, x)| curr_input.set_tangent_value(j, *x));
                curr_input_template.push(curr_input);
            }

            input_templates.push(curr_input_template);
        }

        return Self {
            num_affine_spaces,
            previous_derivative: RwLock::new(previous_derivative),
            previous_derivative_transpose: RwLock::new(previous_derivative_transpose),
            tangent_transpose_matrices,
            tangent_transpose_pseudoinverse_matrices,
            z_chain_matrices,
            input_templates,
            curr_affine_space_idx: RwLock::new(0)
        }
    }
    pub fn update_previous_derivative(&self, previous_derivative: &DMatrix<f64>) {
        // let raw_ptr1 = &self.previous_derivative as *const DMatrix<f64>;
        // let raw_mut_ptr1 = raw_ptr1 as *mut DMatrix<f64>;

        // let raw_ptr2 = &self.previous_derivative_transpose as *const DMatrix<f64>;
        // let raw_mut_ptr2 = raw_ptr2 as *mut DMatrix<f64>;

        // unsafe {
            // *raw_mut_ptr1 = previous_derivative.clone();
            // *raw_mut_ptr2 = previous_derivative.transpose();
        // }
        *self.previous_derivative.write().unwrap() = previous_derivative.clone();
        *self.previous_derivative_transpose.write().unwrap() = previous_derivative.transpose();
    }
    pub fn previous_derivative(&self) -> DMatrix<f64> {
        self.previous_derivative.read().unwrap().clone()
    }
    pub fn previous_derivative_transpose(&self) -> DMatrix<f64> {
        self.previous_derivative_transpose.read().unwrap().clone()
    }
    pub fn curr_affine_space_idx(&self) -> usize { self.curr_affine_space_idx.read().unwrap().clone() }
    pub fn increment_curr_affine_space_idx(&self) {
        // let curr_idx = self.curr_affine_space_idx();
        // let raw_ptr1 = &self.curr_affine_space_idx as *const usize;
        // let raw_mut_ptr1 = raw_ptr1 as *mut usize;
        // unsafe { *raw_mut_ptr1 = (curr_idx + 1) % self.num_affine_spaces; }
        let curr_idx = self.curr_affine_space_idx.read().unwrap().clone();
        *self.curr_affine_space_idx.write().unwrap() = (curr_idx + 1) % self.num_affine_spaces;
    }
}

pub struct SpiderData {
    num_affine_spaces: usize,
    affine_space_dimension: usize,
    f_mat: RwLock<DMatrix<f64>>,
    t_mat: DMatrix<f64>,
    t_mat_pinv: DMatrix<f64>,
    t_mat_affines: Vec<DMatrix<f64>>,
    t_mat_affine_pinvs: Vec<DMatrix<f64>>,
    t_mat_affine_z_chains: Vec<DMatrix<f64>>,
    w: RwLock<DVector<f64>>,
    curr_affine_space_idx: RwLock<usize>
}
impl SpiderData {
    pub fn new(num_inputs: usize, num_outputs: usize, affine_space_dimension: usize, sample_lower_bound: f64, sample_upper_bound: f64) -> Self {
        let num_affine_spaces = (num_inputs as f64 / affine_space_dimension as f64).ceil() as usize;

        let f_mat = DMatrix::<f64>::zeros(num_outputs, num_affine_spaces*affine_space_dimension);

        let mut rng = thread_rng();

        let mut t_mat = DMatrix::<f64>::zeros(num_inputs, num_affine_spaces*affine_space_dimension);
        t_mat.iter_mut().for_each(|x| *x = rng.gen_range(sample_lower_bound..sample_upper_bound));

        let t_mat_pinv = t_mat.clone().pseudo_inverse(0.0).unwrap();

        let mut t_mat_affines: Vec<DMatrix<f64>> = vec![];
        for i in 0..num_affine_spaces {
            t_mat_affines.push( t_mat.view((0, affine_space_dimension*i), (num_inputs, affine_space_dimension)).into() );
        }

        let mut t_mat_affine_pinvs = vec![];
        for a in &t_mat_affines {
            t_mat_affine_pinvs.push( a.clone().pseudo_inverse(0.0).unwrap() );
        }

        let mut t_mat_affine_z_chains = vec![];
        for a in &t_mat_affines {
            let z = get_null_space_basis_matrix(&a.transpose());
            let z_chain = &z * (&z.transpose()*&z).try_inverse().unwrap() * &z.transpose();
            t_mat_affine_z_chains.push(z_chain);
        }

        let w = DVector::<f64>::zeros(num_affine_spaces*affine_space_dimension);

        Self {
            num_affine_spaces,
            affine_space_dimension,
            f_mat: RwLock::new(f_mat),
            t_mat,
            t_mat_pinv,
            t_mat_affines,
            t_mat_affine_pinvs,
            t_mat_affine_z_chains,
            w: RwLock::new(w),
            curr_affine_space_idx: RwLock::new(0)
        }
    }
    pub fn curr_affine_space_idx(&self) -> usize { self.curr_affine_space_idx.read().unwrap().clone() }
    pub fn increment_curr_affine_space_idx(&self) {
        let curr_idx = self.curr_affine_space_idx.read().unwrap().clone();
        *self.curr_affine_space_idx.write().unwrap() = (curr_idx + 1) % self.num_affine_spaces;
    }
    pub fn update_w(&self, decay_multiple: f64) {
        let mut w = self.w.write().unwrap();
        let curr_affine_space_idx = self.curr_affine_space_idx();
        w.iter_mut().for_each(|x|
            if *x > 1.0 { *x = 1.0 }
            else { *x *= decay_multiple  }
        );

        let start_idx = curr_affine_space_idx * self.affine_space_dimension;
        for i in start_idx..start_idx + self.affine_space_dimension {
            w[i] = 1.0;
        }
    }
    pub fn get_w_mat(&self) -> DMatrix<f64> {
        let w = self.w.read().unwrap();
        let w_pinv = w.clone().pseudo_inverse(0.0).unwrap();
        return &*w * &w_pinv;
    }
    pub fn print_w(&self) {
        println!("{}", self.w.read().unwrap());
    }
    pub fn update_f_mat(&self, f_block: &DMatrix<f64>) {
        let mut f_mat = self.f_mat.write().unwrap();

        let m = f_mat.shape().0;

        let s = f_block.shape();

        assert_eq!(s.0, m);
        assert_eq!(s.1, self.affine_space_dimension);

        let curr_affine_space_idx = self.curr_affine_space_idx();

        let start_idx = curr_affine_space_idx * self.affine_space_dimension;
        let mut vm = f_mat.view_mut((0, start_idx), (m, self.affine_space_dimension));
        vm.iter_mut().zip(f_block.as_slice().iter()).for_each(|(x, y)| *x = *y);
    }
    pub fn print_f_mat(&self) { println!("{}", self.f_mat.read().unwrap()); }
    pub fn print_t_mat(&self) {
        println!("{}", self.t_mat);
    }
}

pub struct Spider2Data {
    num_affine_spaces: usize,
    affine_space_dimension: usize,
    f_mat: RwLock<DMatrix<f64>>,
    t_mat: DMatrix<f64>,
    w_t_chains_first_pass: Vec<DMatrix<f64>>,
    w_t_chains: Vec<DMatrix<f64>>,
    t_mat_affines: Vec<DMatrix<f64>>,
    t_mat_affine_pinvs: Vec<DMatrix<f64>>,
    t_mat_affine_transpose_z_chains: Vec<DMatrix<f64>>,
    curr_affine_space_idx: RwLock<usize>,
    first_pass: RwLock<bool>
}
impl Spider2Data {
    pub fn new(num_inputs: usize, num_outputs: usize, affine_space_dimension: usize, sample_lower_bound: f64, sample_upper_bound: f64, decay_multiple: f64) -> Self {
        assert!(0.0 < decay_multiple && decay_multiple < 1.0);

        let n = num_inputs;
        let m = num_outputs;
        let r = affine_space_dimension;

        let num_affine_spaces = (n as f64 / r as f64).ceil() as usize * 2;
        // if num_affine_spaces * r <= n {  num_affine_spaces += 1; }

        let k = num_affine_spaces*r;
        let f_mat = DMatrix::<f64>::zeros(m, k);

        let mut rng = thread_rng();

        let mut t_mat = DMatrix::<f64>::zeros(n, k);
        t_mat.iter_mut().for_each(|x| *x = rng.gen_range(sample_lower_bound..sample_upper_bound));

        let mut w_t_chains_first_pass = vec![];

        let mut curr_w = vec![0.0; k];
        let max_weight = 10.0;
        for curr_affine_space_idx in 0..num_affine_spaces {
            curr_w.iter_mut().for_each(|x| {
                if *x == max_weight { *x = 1.0; }
                else { *x *= decay_multiple; }
            });

            let curr_start_idx = curr_affine_space_idx*r;
            for idx in curr_start_idx..curr_start_idx+r {
                curr_w[idx] = max_weight;
            }

            let w_mat = DMatrix::from_partial_diagonal(k, k, &curr_w);

            let w_t_chain_first_pass = &w_mat*&t_mat.transpose()*&(&t_mat*&w_mat*&t_mat.transpose()).try_inverse().unwrap().transpose();
            w_t_chains_first_pass.push(w_t_chain_first_pass);
        }

        let mut w_t_chains = vec![];
        for curr_affine_space_idx in 0..num_affine_spaces {
            curr_w.iter_mut().for_each(|x| {
                if *x == max_weight { *x = 1.0; }
                else { *x *= decay_multiple; }
            });

            let curr_start_idx = curr_affine_space_idx*r;
            for idx in curr_start_idx..curr_start_idx+r {
                curr_w[idx] = max_weight;
            }

            let w_mat = DMatrix::from_partial_diagonal(k, k, &curr_w);

            let w_t_chain = &w_mat*&t_mat.transpose()*&(&t_mat*&w_mat*&t_mat.transpose()).try_inverse().unwrap().transpose();
            w_t_chains.push(w_t_chain);
        }

        let mut t_mat_affines: Vec<DMatrix<f64>> = vec![];
        for i in 0..num_affine_spaces {
            t_mat_affines.push( t_mat.view((0, r*i), (n, r)).into() );
        }

        let mut t_mat_affine_pinvs = vec![];
        for a in &t_mat_affines {
            t_mat_affine_pinvs.push( a.clone().pseudo_inverse(0.0).unwrap() );
        }

        let mut t_mat_affine_transpose_z_chains = vec![];
        for a in &t_mat_affines {
            let z = get_null_space_basis_matrix(&a.transpose());
            let z_chain = &z * (&z.transpose()*&z).try_inverse().unwrap() * &z.transpose();
            t_mat_affine_transpose_z_chains.push(z_chain);
        }

        Self {
            num_affine_spaces,
            affine_space_dimension,
            f_mat: RwLock::new(f_mat),
            t_mat,
            w_t_chains_first_pass,
            w_t_chains,
            t_mat_affines,
            t_mat_affine_pinvs,
            t_mat_affine_transpose_z_chains,
            curr_affine_space_idx: RwLock::new(0),
            first_pass: RwLock::new(true)
        }
    }
    pub fn curr_affine_space_idx(&self) -> usize { self.curr_affine_space_idx.read().unwrap().clone() }
    pub fn first_pass(&self) -> bool {
        *self.first_pass.read().unwrap()
    }
    pub fn increment_curr_affine_space_idx(&self) {
        let curr_idx = self.curr_affine_space_idx.read().unwrap().clone();
        let new_idx = (curr_idx + 1) % self.num_affine_spaces;
        if new_idx == 0 { *self.first_pass.write().unwrap() = false; }
        *self.curr_affine_space_idx.write().unwrap() = new_idx;
    }
    pub fn update_f_mat(&self, f_block: &DMatrix<f64>) {
        let mut f_mat = self.f_mat.write().unwrap();

        let m = f_mat.shape().0;

        let s = f_block.shape();

        assert_eq!(s.0, m);
        assert_eq!(s.1, self.affine_space_dimension);

        let curr_affine_space_idx = self.curr_affine_space_idx();

        let start_idx = curr_affine_space_idx * self.affine_space_dimension;
        let mut vm = f_mat.view_mut((0, start_idx), (m, self.affine_space_dimension));
        vm.iter_mut().zip(f_block.as_slice().iter()).for_each(|(x, y)| *x = *y);
    }
    pub fn print_f_mat(&self) { println!("{}", self.f_mat.read().unwrap()); }
    pub fn print_t_mat(&self) {
        println!("{}", self.t_mat);
    }
}

pub struct FlowData {
    num_affine_spaces: usize,
    affine_space_dimension: usize,
    f_mat: RwLock<DMatrix<f64>>,
    t_mat: DMatrix<f64>,
    // w_t_chains_first_pass: Vec<DMatrix<f64>>,
    w_t_chains: Vec<DMatrix<f64>>,
    t_mat_affines: Vec<DMatrix<f64>>,
    curr_affine_space_idx: RwLock<usize>,
    first_pass: RwLock<bool>
}
impl FlowData {
    pub fn new(num_inputs: usize, num_outputs: usize, affine_space_dimension: usize, sample_lower_bound: f64, sample_upper_bound: f64, decay_multiple: f64) -> Self {
        assert!(0.0 < decay_multiple && decay_multiple < 1.0);

        let n = num_inputs;
        let m = num_outputs;
        let r = affine_space_dimension;

        let mut num_affine_spaces = (n as f64 / r as f64).ceil() as usize;
        if num_affine_spaces * r <= n {  num_affine_spaces += 1; }

        let k = num_affine_spaces*r;
        let f_mat = DMatrix::<f64>::zeros(m, k);

        let mut rng = thread_rng();

        let mut t_mat = DMatrix::<f64>::zeros(n, k);
        t_mat.iter_mut().for_each(|x| *x = rng.gen_range(sample_lower_bound..sample_upper_bound));

        // let mut w_t_chains_first_pass = vec![];

        let mut curr_w = vec![0.0; k];
        // let max_weight = 10.0;
        for curr_affine_space_idx in 0..num_affine_spaces {
            curr_w.iter_mut().for_each(|x| {
                // if *x == max_weight { *x = 1.0; }
                // else { *x *= decay_multiple; }
                *x *= decay_multiple;
            });

            let curr_start_idx = curr_affine_space_idx*r;
            for idx in curr_start_idx..curr_start_idx+r {
                curr_w[idx] = 1.0;
            }

            /*
            let w_mat = DMatrix::from_partial_diagonal(k, k, &curr_w);

            let try_mat = &(&t_mat*&w_mat*&t_mat.transpose()).try_inverse();
            match try_mat {
                Some(m) => {
                    let w_t_chain_first_pass = &w_mat*&t_mat.transpose()*m.transpose();
                    w_t_chains_first_pass.push(w_t_chain_first_pass);
                }
                None => {
                    return Self::new(num_inputs, num_outputs, affine_space_dimension, sample_lower_bound, sample_upper_bound, decay_multiple);
                }
            }
            */
        }

        let mut w_t_chains = vec![];
        for curr_affine_space_idx in 0..num_affine_spaces {
            curr_w.iter_mut().for_each(|x| {
                // if *x == max_weight { *x = 1.0; }
                // else { *x *= decay_multiple; }
                *x *= decay_multiple;
            });

            let curr_start_idx = curr_affine_space_idx*r;
            for idx in curr_start_idx..curr_start_idx+r {
                curr_w[idx] = 1.0;
            }

            let w_mat = DMatrix::from_partial_diagonal(k, k, &curr_w);

            let w_t_chain = &w_mat*&t_mat.transpose()*&(&t_mat*&w_mat*&t_mat.transpose()).try_inverse().unwrap().transpose();
            w_t_chains.push(w_t_chain);
        }

        let mut t_mat_affines: Vec<DMatrix<f64>> = vec![];
        for i in 0..num_affine_spaces {
            t_mat_affines.push( t_mat.view((0, r*i), (n, r)).into() );
        }

        Self {
            num_affine_spaces,
            affine_space_dimension,
            f_mat: RwLock::new(f_mat),
            t_mat,
            // w_t_chains_first_pass,
            w_t_chains,
            t_mat_affines,
            curr_affine_space_idx: RwLock::new(0),
            first_pass: RwLock::new(true)
        }
    }
    pub fn curr_affine_space_idx(&self) -> usize { self.curr_affine_space_idx.read().unwrap().clone() }
    pub fn first_pass(&self) -> bool {
        *self.first_pass.read().unwrap()
    }
    pub fn increment_curr_affine_space_idx(&self) {
        let curr_idx = self.curr_affine_space_idx.read().unwrap().clone();
        let new_idx = (curr_idx + 1) % self.num_affine_spaces;
        if new_idx == 0 { *self.first_pass.write().unwrap() = false; }
        *self.curr_affine_space_idx.write().unwrap() = new_idx;
    }
    /// f_block expected to be num_function_outputs (m) x affine_space_dimension
    pub fn update_f_mat(&self, f_block: &DMatrix<f64>, affine_space_idx: usize) {
        let mut f_mat = self.f_mat.write().unwrap();

        let m = f_mat.shape().0;

        let s = f_block.shape();

        assert_eq!(s.0, m);
        assert_eq!(s.1, self.affine_space_dimension);

        // let curr_affine_space_idx = self.curr_affine_space_idx();

        let start_idx = affine_space_idx * self.affine_space_dimension;
        let mut vm = f_mat.view_mut((0, start_idx), (m, self.affine_space_dimension));
        vm.iter_mut().zip(f_block.as_slice().iter()).for_each(|(x, y)| *x = *y);
    }
    pub fn print_f_mat(&self) { println!("{}", self.f_mat.read().unwrap()); }
    pub fn print_t_mat(&self) {
        println!("{}", self.t_mat);
    }
}
*/

/*
fn faer_mat_to_nalgebra_dmatrix(mat: &Mat<f64>) -> DMatrix<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    let mut out = DMatrix::zeros(nrows, ncols);

    for i in 0..nrows {
        for j in 0..ncols {
            out[(i,j)] = mat.read(i, j);
        }
    }

    out
}
fn nalgebra_dmatrix_to_faer_mat(dmatrix: &DMatrix<f64>) -> Mat<f64> {
    let nrows = dmatrix.nrows();
    let ncols = dmatrix.ncols();

    let mut out = Mat::with_capacity(nrows, ncols);
    // let out = Mat::with_dims(nrows, ncols, |i, j| dmatrix[(i,j)]);

    out
}
pub fn get_null_space_basis_matrix(mat: &DMatrix<f64>) -> DMatrix<f64> {
    let rank = mat.rank(0.0);
    let shape = mat.shape();
    let m = shape.0;
    let n = shape.1;

    let mut u: Mat<f64> = Mat::zeros(m, m);
    let mut s: Mat<f64> = Mat::zeros(usize::min(m, n), 1);
    let mut v: Mat<f64> = Mat::zeros(n, n);

    let mut mem = GlobalMemBuffer::new(
        compute_svd_req::<f64>(
            mat.nrows(),
            mat.ncols(),
            ComputeVectors::Full,
            ComputeVectors::Full,
            Parallelism::None,
            SvdParams::default(),
        ).unwrap(),
    );
    let stack = DynStack::new(&mut mem);

    let mat_faer = nalgebra_dmatrix_to_faer_mat(mat);
    compute_svd(mat_faer.as_ref(), s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), f64::EPSILON, f64::MIN_POSITIVE, Parallelism::None, stack, SvdParams::default());

    let v = faer_mat_to_nalgebra_dmatrix(&v);

    let z = v.view((0, rank), (n, n - rank));

    return z.into()
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#[derive(Debug, Clone, Copy)]
pub enum RicochetTermination {
    MaxIters(usize),
    MaxTime(Duration),
    L1 { threshold: f64, max_iters: usize },
    L2 { threshold: f64, max_iters: usize },
    LInf { threshold: f64, max_iters: usize }
}
impl RicochetTermination {
    fn map_to_new_internal(&self) -> RicochetTerminationInternal {
        match self {
            RicochetTermination::MaxIters(iter) => { RicochetTerminationInternal::MaxIters { max_iters: *iter, curr_iter_count: 1 } }
            RicochetTermination::MaxTime(duration) => { RicochetTerminationInternal::MaxTime { threshold: duration.clone(), start_instant: Instant::now() } }
            RicochetTermination::L1 { threshold, max_iters } => {
                RicochetTerminationInternal::L1 {
                    threshold: *threshold,
                    max_iters: *max_iters,
                    curr_iter_count: 1,
                }
            }
            RicochetTermination::L2 { threshold, max_iters } => {
                RicochetTerminationInternal::L1 {
                    threshold: *threshold,
                    max_iters: *max_iters,
                    curr_iter_count: 1,
                }
            }
            RicochetTermination::LInf { threshold, max_iters } => {
                RicochetTerminationInternal::LInf {
                    threshold: *threshold,
                    max_iters: *max_iters,
                    curr_iter_count: 1,
                }
            }
        }
    }
}

#[allow(dead_code)]
enum RicochetTerminationInternal {
    MaxIters { max_iters: usize, curr_iter_count: usize },
    MaxTime { threshold: Duration, start_instant: Instant },
    L1 { threshold: f64, max_iters: usize, curr_iter_count: usize },
    L2 { threshold: f64, max_iters: usize, curr_iter_count: usize },
    LInf { threshold: f64, max_iters: usize, curr_iter_count: usize }
}
impl RicochetTerminationInternal {
    fn terminate(&mut self, previous_derivative: &DMatrix<f64>, new_derivative: &DMatrix<f64>) -> bool {
        return match self {
            RicochetTerminationInternal::MaxIters { max_iters: iter_threshold, curr_iter_count } => {
                if *curr_iter_count >= *iter_threshold {
                    true
                } else {
                    *curr_iter_count += 1;
                    false
                }
            }
            RicochetTerminationInternal::MaxTime { threshold, start_instant } => {
                if start_instant.elapsed() >= *threshold {
                    true
                } else {
                    false
                }
            }
            RicochetTerminationInternal::L1 { threshold, max_iters, curr_iter_count } => {
                if *curr_iter_count >= *max_iters {
                    return true;
                } else {
                    *curr_iter_count += 1;
                }

                let mut l1 = 0.0;
                previous_derivative.as_slice().iter().zip(new_derivative.as_slice().iter()).for_each(|(x, y)| l1 += (*x - *y).abs());
                if l1 < *threshold { true } else { false }
            }
            RicochetTerminationInternal::L2 { threshold, max_iters, curr_iter_count } => {
                if *curr_iter_count >= *max_iters {
                    return true;
                } else {
                    *curr_iter_count += 1;
                }

                let mut l2 = 0.0;
                previous_derivative.as_slice().iter().zip(new_derivative.as_slice().iter()).for_each(|(x, y)| l2 += (*x - *y).powi(2));
                if l2.sqrt() < *threshold { true } else { false }
            }
            RicochetTerminationInternal::LInf { threshold, max_iters, curr_iter_count } => {
                if *curr_iter_count >= *max_iters {
                    return true;
                } else {
                    *curr_iter_count += 1;
                }

                let mut linf = f64::MIN;
                previous_derivative.as_slice().iter().zip(new_derivative.as_slice().iter()).for_each(|(x, y)| {
                    let diff = (*x - *y).abs();
                    if diff > linf { linf = diff; }
                });

                if linf < *threshold {
                    return true;
                }

                false
            }
        }
    }
}
*/
