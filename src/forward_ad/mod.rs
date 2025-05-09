pub mod adfn;
#[cfg(feature = "nightly")]
pub mod adf;

pub trait ForwardADTrait {
    fn value(&self) -> f64;
    fn tangent_size() -> usize;
    fn tangent_as_vec(&self) -> Vec<f64>;
    fn set_value(&mut self, value: f64);
    fn set_tangent(&mut self, tangent: Vec<f64>) {
        let t = Self::tangent_size();
        assert_eq!(tangent.len(), t);
        for i in 0..t {
            self.set_tangent_value(i, tangent[i]);
        }
    }
    fn set_tangent_value(&mut self, idx: usize, value: f64);
}

/*
use std::ops::{Add, AddAssign, Sub};

#[macro_export]
macro_rules! add_output_value {
    ($self: ident, $rhs: ident) => { $self.value + $rhs.value }
}
#[macro_export]
macro_rules! add_derivative1_value {
    ($self: ident, $rhs: ident) => { 1.0 }
}
#[macro_export]
macro_rules! add_derivative2_value {
    ($self: ident, $rhs: ident) => { 1.0 }
}
#[macro_export]
macro_rules! sub_output_value {
    ($self: ident, $rhs: ident) => { $self.value - $rhs.value }
}
#[macro_export]
macro_rules! sub_derivative1_value {
    ($self: ident, $rhs: ident) => { 1.0 }
}
#[macro_export]
macro_rules! sub_derivative2_value {
    ($self: ident, $rhs: ident) => { -1.0 }
}


#[macro_export]
macro_rules! mul_output_value {
    () => { self.value * rhs.value }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Copy)]
pub struct adf {
    pub (crate) value: f64,
    pub (crate) tangent: f64
}
impl adf {
    pub fn new(value: f64, tangent: f64) -> Self {
        Self {
            value,
            tangent
        }
    }
    pub fn new_constant(value: f64) -> Self {
        Self {
            value,
            tangent: 0.0
        }
    }
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.value
    }
    #[inline(always)]
    pub fn tangent(&self) -> f64 {
        self.tangent
    }
}

/*
impl Add<Self> for adf {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let output_value = self.value + rhs.value;
        let d_add_d_arg1 = 1.0;
        let d_add_d_arg2 = 1.0;
        let output_tangent = d_add_d_arg1*self.tangent + d_add_d_arg2*rhs.tangent;

        adf {
            value: output_value,
            tangent: output_tangent
        }

    }
}
*/
impl AddAssign<Self> for adf {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[macro_export]
macro_rules! forward_ad_standard_two_input_trait_adf {
    ($trait_name: tt, $function_name: tt, $output_value_macro: tt, $derivative1: tt, $derivative2: tt) => {
        impl $trait_name<Self> for adf {
            type Output = Self;

            fn $function_name(self, rhs: Self) -> Self::Output {
                let output_value = $output_value_macro!(self, rhs);
                let d1 = $derivative1!(self, rhs);
                let d2 = $derivative2!(self, rhs);
                let output_tangent = d1*self.tangent + d2*rhs.tangent;

                adf {
                    value: output_value,
                    tangent: output_tangent
                }
            }
        }
    }
}
// forward_ad_standard_two_input_trait_adf!(Add, add, add_output_value, add_derivative1_value, add_derivative2_value);
// forward_ad_standard_two_input_trait_adf!(Sub, sub, sub_output_value, sub_derivative1_value, sub_derivative2_value);

#[macro_export]
macro_rules! forward_ad_standard_two_input_trait_nongeneric_wrapper {
    ($t: tt, $inner_macro: tt, $trait_name: tt, $function_name: tt, $output_value_macro: tt, $derivative1: tt, $derivative2: tt) => {
        impl $trait_name<Self> for $t {
            $inner_macro!($trait_name, $function_name, $output_value_macro, $derivative1, $derivative2);
        }
    }
}
#[macro_export]
macro_rules! forward_ad_standard_two_input_trait_inner_macro {
    ($trait_name: tt, $function_name: tt, $output_value_macro: tt, $derivative1: tt, $derivative2: tt) => {
        type Output = Self;

        fn $function_name(self, rhs: Self) -> Self::Output {
            let output_value = $output_value_macro!(self, rhs);
            let d1 = $derivative1!(self, rhs);
            let d2 = $derivative2!(self, rhs);
            let output_tangent = d1*self.tangent + d2*rhs.tangent;

            adf {
                value: output_value,
                tangent: output_tangent
            }
        }
    }
}

forward_ad_standard_two_input_trait_nongeneric_wrapper!(adf, forward_ad_standard_two_input_trait_inner_macro, Add, add, add_output_value, add_derivative1_value, add_derivative2_value);
*/
