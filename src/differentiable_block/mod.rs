use std::marker::PhantomData;
use std::sync::RwLock;
use faer_core::Mat;
use nalgebra::{DMatrix};
use rand::{Rng, thread_rng};
use crate::AD;
use crate::forward_ad::adf::adf;
use crate::reverse_ad::adr::{adr, GlobalComputationGraph};
use crate::simd::f64xn::f64xn;

pub trait DifferentiableBlockTrait {
    type U;

    fn call<T: AD>(inputs: &[T], args: &Self::U) -> Vec<T>;
    fn num_outputs(args: &Self::U) -> usize;
}

pub trait DerivativeDataTrait<D: DifferentiableBlockTrait> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct DifferentiableBlock<D: DifferentiableBlockTrait, E: DerivativeDataTrait<D>> {
    derivative_data: E,
    phantom_data: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, E: DerivativeDataTrait<D>> DifferentiableBlock<D, E> {
    pub fn new(derivative_data: E) -> Self {
        Self {
            derivative_data,
            phantom_data: Default::default()
        }
    }
    pub fn call<T: AD>(&self, inputs: &[T], args: &D::U) -> Vec<T> {
        D::call(inputs, args)
    }
    pub fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
        self.derivative_data.derivative(inputs, args)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct FiniteDifferencing<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> FiniteDifferencing<D> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D> for FiniteDifferencing<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
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

pub struct ForwardAD<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> ForwardAD<D> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D> for ForwardAD<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        for col_idx in 0..num_inputs {
            let mut inputs_ad = vec![];
            for (i, input) in inputs.iter().enumerate() {
                if i == col_idx {
                    inputs_ad.push(adf::new(*input, [1.0]))
                } else {
                    inputs_ad.push(adf::new(*input, [0.0]))
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

pub struct ReverseAD<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> ReverseAD<D> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D> for ReverseAD<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
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
            let grad_output = f[row_idx].get_backwards_mode_grad();
            for col_idx in 0..num_inputs {
                let d = grad_output.wrt(&inputs_ad[col_idx]);
                out_derivative[(row_idx, col_idx)] = d;
            }
        }

        (out_value, out_derivative)
    }
}

pub struct ForwardADMulti<D: DifferentiableBlockTrait, const K: usize> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, const K: usize> ForwardADMulti<D, K> {
    pub fn new() -> Self {
        assert!(K > 0);
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait, const K: usize> DerivativeDataTrait<D> for ForwardADMulti<D, K> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        let mut curr_idx = 0;

        'l1: loop {
            let mut inputs_ad = vec![];
            for input in inputs.iter() {
                inputs_ad.push(adf::new(*input, [0.0; K]))
            }

            'l2: for i in 0..K {
                if curr_idx + i >= num_inputs { break 'l2; }
                inputs_ad[curr_idx+i].tangent[i] = 1.0;
            }

            let f = D::call(&inputs_ad, args);
            assert_eq!(f.len(), num_outputs);

            for (row_idx, res) in f.iter().enumerate() {
                if out_value.len() < num_outputs {
                    out_value.push(res.value);
                }
                'l3: for i in 0..K {
                    if curr_idx + i >= num_inputs { break 'l3; }
                    out_derivative[(row_idx, curr_idx+i)] = res.tangent[i];
                }
            }

            curr_idx += K;
            if curr_idx > num_inputs { break 'l1; }
        }

        return (out_value, out_derivative)
    }
}

pub struct FiniteDifferencingMulti<D: DifferentiableBlockTrait, const K: usize> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, const K: usize> FiniteDifferencingMulti<D, K> {
    pub fn new() -> Self {
        assert!(K > 1);
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait, const K: usize> DerivativeDataTrait<D> for FiniteDifferencingMulti<D, K> {
    fn derivative(&self, inputs: &[f64], args: &D::U) -> (Vec<f64>, DMatrix<f64>) {
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

pub struct RicochetForwardADMulti<D: DifferentiableBlockTrait, const K: usize> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, const K: usize> RicochetForwardADMulti<D, K> {
    pub fn new() -> Self {
        Self {
            p: Default::default()
        }
    }
}

pub struct RicochetFiniteDifferencing<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> RicochetFiniteDifferencing<D> {
    pub fn new() -> Self {
        Self {
            p: Default::default()
        }
    }
}

pub struct RicochetFiniteDifferencingMulti<D: DifferentiableBlockTrait, const K: usize> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, const K: usize> RicochetFiniteDifferencingMulti<D, K> {
    pub fn new() -> Self {
        Self {
            p: Default::default()
        }
    }
}

pub struct RicochetData {
    num_affine_spaces: usize,
    previous_derivative: RwLock<DMatrix<f64>>,
    previous_derivative_transpose: RwLock<DMatrix<f64>>,
    tangent_transpose_matrices: Vec<DMatrix<f64>>,
    tangent_transpose_pseudoinverse_matrices: Vec<DMatrix<f64>>,
    z_chain_matrices: Vec<DMatrix<f64>>
}
impl RicochetData {
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

        let mut rng = thread_rng();
        for _ in 0..num_affine_spaces {
            let mut tangent_transpose_matrix = DMatrix::zeros(affine_space_dimension, num_inputs);
            tangent_transpose_matrix.iter_mut().for_each(|x| *x = rng.gen_range(sample_lower_bound..sample_upper_bound) );

            // println!(" > {:?}", tangent_transpose_matrix);

            let tangent_transpose_pseudoinverse_matrix = tangent_transpose_matrix.clone().pseudo_inverse(0.0).unwrap();
            // println!(" >>> {:?}", tangent_transpose_pseudoinverse_matrix);

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

        todo!()
    }
}

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

    let out = Mat::with_dims(nrows, ncols, |i, j| dmatrix[(i,j)]);

    out
}