use std::marker::PhantomData;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use dyn_stack::{DynStack, GlobalMemBuffer};
use faer_core::{Mat, Parallelism};
use faer_svd::{compute_svd, compute_svd_req, ComputeVectors, SvdParams};
use nalgebra::{DMatrix};
use rand::{Rng, thread_rng};
use crate::{AD, ADNumMode};
use crate::forward_ad::adfn::adfn;
use crate::forward_ad::ForwardADTrait;
use crate::reverse_ad::adr::{adr, GlobalComputationGraph};
use crate::simd::f64xn::f64xn;

pub trait DifferentiableBlockTrait {
    type U<T: AD>;

    fn call<T1: AD>(inputs: &[T1], args: &Self::U<T1>) -> Vec<T1>;
    fn num_inputs<T1: AD>(args: &Self::U<T1>) -> usize;
    fn num_outputs<T1: AD>(args: &Self::U<T1>) -> usize;
}

pub trait DerivativeDataTrait<D: DifferentiableBlockTrait, T: AD> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct DifferentiableBlock<D: DifferentiableBlockTrait, E: DerivativeDataTrait<D, T>, T: AD, > {
    derivative_data: E,
    phantom_data: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, E: DerivativeDataTrait<D, T>, T: AD, > DifferentiableBlock<D, E, T> {
    pub fn new(derivative_data: E) -> Self {
        Self {
            derivative_data,
            phantom_data: Default::default()
        }
    }
    pub fn call<T1: AD>(&self, inputs: &[T1], args: &D::U<T1>) -> Vec<T1> {
        D::call(inputs, args)
    }
    pub fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
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
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D, f64> for FiniteDifferencing<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U<f64>) -> (Vec<f64>, DMatrix<f64>) {
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

pub struct ReverseAD<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> ReverseAD<D> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D, adr> for ReverseAD<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U<adr>) -> (Vec<f64>, DMatrix<f64>) {
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

pub struct ForwardAD<D: DifferentiableBlockTrait> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait> ForwardAD<D> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait> DerivativeDataTrait<D, adfn<1>> for ForwardAD<D> {
    fn derivative(&self, inputs: &[f64], args: &D::U<adfn<1>>) -> (Vec<f64>, DMatrix<f64>) {
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

pub struct ForwardADMulti<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> {
    p: PhantomData<(D, T)>
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> ForwardADMulti<D, T> {
    pub fn new() -> Self {
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeDataTrait<D, T> for ForwardADMulti<D, T> {
    fn derivative(&self, inputs: &[f64], args: &D::U<T>) -> (Vec<f64>, DMatrix<f64>) {
        let num_inputs = inputs.len();
        let num_outputs = D::num_outputs(args);
        let mut out_derivative = DMatrix::zeros(num_outputs, num_inputs);
        let mut out_value = vec![];

        let mut curr_idx = 0;

        let k = T::tangent_size();
        'l1: loop {
            let mut inputs_ad = vec![];
            for input in inputs.iter() {
                // inputs_ad.push(adf::new(*input, [0.0; K]))
                inputs_ad.push(T::constant(*input));
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

pub struct FiniteDifferencingMulti<D: DifferentiableBlockTrait, const K: usize> {
    p: PhantomData<D>
}
impl<D: DifferentiableBlockTrait, const K: usize> FiniteDifferencingMulti<D, K> {
    pub fn new() -> Self {
        assert!(K > 1);
        Self { p: PhantomData::default() }
    }
}
impl<D: DifferentiableBlockTrait, const K: usize> DerivativeDataTrait<D, f64xn<K>> for FiniteDifferencingMulti<D, K> {
    fn derivative(&self, inputs: &[f64], args: &D::U<f64xn<K>>) -> (Vec<f64>, DMatrix<f64>) {
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
            ricochet_data: RicochetData::new(num_inputs, num_outputs, T::tangent_size(), -100.0, 100.0, None),
            ricochet_termination,
            p: PhantomData::default()
        }
    }
    pub fn ricochet_data(&self) -> &RicochetData<T> {
        &self.ricochet_data
    }
}
impl<D: DifferentiableBlockTrait, T: AD + ForwardADTrait> DerivativeDataTrait<D, T> for Ricochet<D, T> {
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

            let terminate = t.terminate(&previous_derivative, &new_derivative);

            if terminate {
                let mut output_value = vec![];
                for ff in f { output_value.push(ff.value()); }
                return (output_value, new_derivative)
            } else {
                self.ricochet_data.update_previous_derivative(&new_derivative);
                self.ricochet_data.increment_curr_affine_space_idx();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
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

////////////////////////////////////////////////////////////////////////////////////////////////////

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