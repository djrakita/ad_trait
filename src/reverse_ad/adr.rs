use once_cell::sync::OnceCell;

use std::sync::{RwLock};

use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{DefaultAllocator, Dim, DimName, Matrix, OPoint, RawStorageMut};
use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use tinyvec::{TinyVec, tiny_vec};
use crate::{AD, ADNumMode, F64};
use serde::{Serialize, Deserialize, Serializer, de, Deserializer};
use serde::de::{MapAccess, Visitor};
use serde::ser::{SerializeStruct};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct adr {
    value: f64,
    node_idx: NodeIdx,
    computation_graph: &'static ComputationGraph
}
impl Debug for adr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("adr { ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}
impl Default for adr {
    fn default() -> Self {
        Self::zero()
    }
}
impl adr {
    pub fn new_variable(value: f64, reset_computation_graph: bool) -> Self {
        if reset_computation_graph { GlobalComputationGraph::get().reset(); }
        GlobalComputationGraph::get().spawn_value(value)
    }
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn get_backwards_mode_grad(&self) -> BackwardsModeGradOutput {
        let nodes = self.computation_graph.nodes.read().unwrap();
        let l = nodes.len();
        let mut adjoints = vec![0.0; l];
        match self.node_idx {
            NodeIdx::Constant => { panic!("cannot get backwards mode grad on a constant.") }
            NodeIdx::Idx(node_idx) => { adjoints[node_idx] = 1.0; }
        }

        for node_idx in (0..l).rev() {
            let node = &nodes[node_idx];
            let parent_adjoints = node.node_type.get_derivatives_wrt_parents(node.parent_0, node.parent_1);
            if parent_adjoints.len() == 1 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
            } else if parent_adjoints.len() == 2 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                let parent_1_idx = node.parent_1_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant  {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
                if parent_1_idx != NodeIdx::Constant  {
                    adjoints[parent_1_idx.get_idx()] += curr_adjoint * parent_adjoints[1];
                }
            }
        }

        BackwardsModeGradOutput {
            adjoints
        }
    }
}

impl Serialize for adr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer, {
        let mut state = serializer.serialize_struct("adr", 1)?;
        state.serialize_field("value", &self.value)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for adr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de>, {
        enum Field { Value }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                        formatter.write_str("value or tangent")
                    }

                    fn visit_str<E: de::Error>(self, value: &str) -> Result<Field, E> {
                        match value {
                            "value" => Ok(Field::Value),
                            _ => { Err(de::Error::unknown_field(value, FIELDS)) }
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct AdrVisitor;

        impl<'de> Visitor<'de> for AdrVisitor {
            type Value = adr;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("struct adr")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<adr, V::Error> {
                let mut value = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() { return Err(de::Error::duplicate_field("value")); }
                            value = Some(map.next_value()?);
                        }
                    }
                }

                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(adr::constant(value))
            }
        }

        const FIELDS: &'static [&'static str] = &["value"];
        deserializer.deserialize_struct("adr", FIELDS, AdrVisitor)
    }
}

#[derive(Clone, Debug)]
pub struct BackwardsModeGradOutput {
    adjoints: Vec<f64>
}
impl BackwardsModeGradOutput {
    pub fn wrt(&self, v: &adr) -> f64 {
        self.adjoints[v.node_idx.get_idx()]
    }
}

impl AD for adr {
    fn constant(constant: f64) -> Self {
        return unsafe {
            adr {
                value: constant,
                node_idx: NodeIdx::Constant,
                computation_graph: &*GlobalComputationGraph::get().0
            }
        }
    }

    fn to_constant(&self) -> f64 {
        self.value()
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::ReverseAD
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 -  Self::constant(arg2)
    }

    fn mul_scalar(arg1: f64, arg2: Self) -> Self {
         Self::constant(arg1) * arg2
    }

    fn div_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) / arg2
    }

    fn div_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 / Self::constant(arg2)
    }

    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) % arg2
    }

    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 % Self::constant(arg2)
    }
}

#[derive(Debug)]
pub struct ComputationGraph {
    nodes: RwLock<Vec<ComputationGraphNode>>
}
impl ComputationGraph {
    fn new() -> Self {
        Self {
            nodes: RwLock::new(vec![])
        }
    }
    fn reset(&self) {
        self.nodes.write().expect("error").clear()
    }
    fn spawn_variable(&'static self, value: f64) -> adr {
        let mut nodes = self.nodes.write().expect("error");
        let node_idx = nodes.len();
        let node = ComputationGraphNode {
            node_idx: node_idx,
            node_type: NodeType::Constant,
            value,
            parent_0: None,
            parent_1: None,
            parent_0_idx: None,
            parent_1_idx: None
        };
        nodes.push(node);
        adr {
            value,
            node_idx: NodeIdx::Idx(node_idx),
            computation_graph: self
        }
    }
    fn add_node(&'static self, node_type: NodeType, value: f64, parent_0: Option<f64>, parent_1: Option<f64>, parent_0_idx: Option<NodeIdx>, parent_1_idx: Option<NodeIdx>) -> adr {
        if parent_0_idx.is_some() {
            if parent_1_idx.is_some() {
                if parent_0_idx.unwrap() == NodeIdx::Constant && parent_1_idx.unwrap() == NodeIdx::Constant {
                    return adr {
                        value,
                        node_idx: NodeIdx::Constant,
                        computation_graph: self,
                    }
                }
            } else {
                if parent_0_idx.unwrap() == NodeIdx::Constant {
                    return adr {
                        value,
                        node_idx: NodeIdx::Constant,
                        computation_graph: self
                    }
                }
            }
        }

        let mut nodes = self.nodes.write().expect("error");
        let node_idx = nodes.len();
        nodes.push( ComputationGraphNode {
            node_idx,
            node_type,
            value,
            parent_0,
            parent_1,
            parent_0_idx,
            parent_1_idx
        } );

        adr {
            value,
            node_idx: NodeIdx::Idx(node_idx),
            computation_graph: self
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ComputationGraphNode {
    node_idx: usize,
    node_type: NodeType,
    value: f64,
    parent_0: Option<f64>,
    parent_1: Option<f64>,
    parent_0_idx: Option<NodeIdx>,
    parent_1_idx: Option<NodeIdx>
}

#[derive(Clone, Debug, Copy)]
pub enum NodeType {
    Constant,
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Abs,
    Signum,
    Max,
    Min,
    Atan2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Fract,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Log,
    Sqrt,
    Exp,
    Powf
}
impl NodeType {
    fn get_derivatives_wrt_parents(&self, parent_0: Option<f64>, parent_1: Option<f64>) -> TinyVec<[f64; 2]> {
        return match self {
            NodeType::Constant => { tiny_vec!([f64; 2]) }
            NodeType::Add => { tiny_vec!([f64; 2] => 1.0, 1.0) }
            NodeType::Mul => { tiny_vec!([f64; 2] => parent_1.unwrap(), parent_0.unwrap()) }
            NodeType::Sub => { tiny_vec!([f64; 2] => 1.0, -1.0) }
            NodeType::Div => { tiny_vec!([f64; 2] => 1.0/parent_1.unwrap(), -parent_0.unwrap()/(parent_1.unwrap()*parent_1.unwrap())) }
            NodeType::Neg => { tiny_vec!([f64; 2] => -1.0)  }
            NodeType::Abs => {
                let val = parent_1.unwrap();
                if val >= 0.0 { tiny_vec!([f64; 2] => 1.0) } else { tiny_vec!([f64; 2] => -1.0)  }
            }
            NodeType::Signum => {
                tiny_vec!([f64; 2] => 0.0)
            }
            NodeType::Max => {
                if parent_0.unwrap() >= parent_1.unwrap() {
                    tiny_vec!([f64; 2] => 1.0, 0.0)
                } else {
                    tiny_vec!([f64; 2] => 0.0, 1.0)
                }
            }
            NodeType::Min => {
                if parent_0.unwrap() <= parent_1.unwrap() {
                    tiny_vec!([f64; 2] => 1.0, 0.0)
                } else {
                    tiny_vec!([f64; 2] => 0.0, 1.0)
                }
            }
            NodeType::Atan2 => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                tiny_vec!([f64; 2] => rhs/(lhs*lhs + rhs*rhs), -lhs/(lhs*lhs + rhs*rhs))
            }
            NodeType::Floor => { tiny_vec!([f64; 2] => 0.0)  }
            NodeType::Ceil => { tiny_vec!([f64; 2] => 0.0) }
            NodeType::Round => { tiny_vec!([f64; 2] => 0.0) }
            NodeType::Trunc => { tiny_vec!([f64; 2] => 0.0) }
            NodeType::Fract => { tiny_vec!([f64; 2] => 0.0) }
            NodeType::Sin => { tiny_vec!([f64; 2] => ComplexField::cos(parent_0.unwrap())) }
            NodeType::Cos => { tiny_vec!([f64; 2] => ComplexField::sin(-parent_0.unwrap())) }
            NodeType::Tan => {
                let c = ComplexField::cos(parent_0.unwrap());
                tiny_vec!([f64; 2] => 1.0 / (c*c))
            }
            NodeType::Asin => {
                tiny_vec!([f64; 2] => 1.0 / ComplexField::sqrt(1.0 - parent_0.unwrap() * parent_0.unwrap()))
            }
            NodeType::Acos => {
                tiny_vec!([f64; 2] => -1.0/ComplexField::sqrt(1.0 - parent_0.unwrap() * parent_0.unwrap()))
            }
            NodeType::Atan => {
                tiny_vec!([f64; 2] => 1.0/(parent_0.unwrap()*parent_0.unwrap() + 1.0))
            }
            NodeType::Sinh => { tiny_vec!([f64; 2] => ComplexField::cosh(parent_0.unwrap())) }
            NodeType::Cosh => { tiny_vec!([f64; 2] => ComplexField::sinh(parent_0.unwrap())) }
            NodeType::Tanh => {
                let c = ComplexField::cosh(parent_0.unwrap());
                tiny_vec!([f64; 2] => 1.0 / (c*c))
            }
            NodeType::Asinh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f64; 2] => 1.0/(lhs*lhs + 1.0).sqrt())
            }
            NodeType::Acosh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f64; 2] => 1.0/(ComplexField::sqrt(lhs - 1.0)*ComplexField::sqrt(lhs + 1.0)) )
            }
            NodeType::Atanh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f64; 2] => 1.0/(1.0 - lhs*lhs))
            }
            NodeType::Log => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                let ln_rhs = ComplexField::ln(rhs);
                let ln_lhs = ComplexField::ln(lhs);
                tiny_vec!([f64; 2] => 1.0/(lhs * ln_rhs), -ln_lhs / (rhs * ln_rhs * ln_rhs))
            }
            NodeType::Sqrt => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f64; 2] => 1.0/(2.0*ComplexField::sqrt(lhs)))
            }
            NodeType::Exp => {
                tiny_vec!([f64; 2] => ComplexField::exp(parent_0.unwrap()))
            }
            NodeType::Powf => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                tiny_vec!([f64; 2] => rhs * ComplexField::powf(lhs, rhs - 1.0), ComplexField::powf(lhs, rhs) * ComplexField::ln(lhs))
            }
        };
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum NodeIdx {
    Constant,
    Idx(usize)
}
impl NodeIdx {
    #[inline]
    pub fn get_idx(&self) -> usize {
        match self {
            NodeIdx::Constant => { panic!("cannot get idx from constant.") }
            NodeIdx::Idx(idx) => { return *idx }
        }
    }
}

static mut _GLOBAL_COMPUTATION_GRAPHS: OnceCell<ComputationGraph> = OnceCell::new();

pub struct GlobalComputationGraph(*const ComputationGraph);
impl GlobalComputationGraph {
    pub fn reset(&self) {
        unsafe { return (*self.0).reset() }
    }
    pub fn spawn_value(&self, value: f64) -> adr {
        unsafe { return (*self.0).spawn_variable(value) }
    }
    pub fn get() -> GlobalComputationGraph {
        let computation_graph = unsafe { _GLOBAL_COMPUTATION_GRAPHS.get_or_init(|| ComputationGraph::new()) };
        let r: *const ComputationGraph = computation_graph;
        return GlobalComputationGraph(r);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<F64> for adr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        AD::add_scalar(rhs.0, self)
    }
}

impl AddAssign<F64> for adr {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        *self = *self + rhs;
    }
}

impl Mul<F64> for adr {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        AD::mul_scalar(rhs.0, self)
    }
}

impl MulAssign<F64> for adr {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        *self = *self * rhs;
    }
}

impl Sub<F64> for adr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        AD::sub_r_scalar(self, rhs.0)
    }
}

impl SubAssign<F64> for adr {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        *self = *self - rhs;
    }
}

impl Div<F64> for adr {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        AD::div_r_scalar(self, rhs.0)
    }
}

impl DivAssign<F64> for adr {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        *self = *self / rhs;
    }
}

impl Rem<F64> for adr {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        AD::rem_r_scalar(self, rhs.0)
    }
}

impl RemAssign<F64> for adr {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        *self = *self % rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<Self> for adr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let out_value = self.value() + rhs.value();
        self.computation_graph.add_node(NodeType::Add, out_value, Some(self.value()), Some(rhs.value()), Some(self.node_idx), Some(rhs.node_idx))
    }
}
impl AddAssign<Self> for adr {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul<Self> for adr {
    type Output = Self;

        #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let out_value = self.value() * rhs.value();
        self.computation_graph.add_node(NodeType::Mul, out_value, Some(self.value()), Some(rhs.value()), Some(self.node_idx), Some(rhs.node_idx))
    }
}
impl MulAssign<Self> for adr {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sub<Self> for adr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let out_value = self.value() - rhs.value();
        self.computation_graph.add_node(NodeType::Sub, out_value, Some(self.value()), Some(rhs.value()), Some(self.node_idx), Some(rhs.node_idx))
    }
}
impl SubAssign<Self> for adr {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Div<Self> for adr {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let out_value = self.value() / rhs.value();
        self.computation_graph.add_node(NodeType::Div, out_value, Some(self.value()), Some(rhs.value()), Some(self.node_idx), Some(rhs.node_idx))
    }
}
impl DivAssign<Self> for adr {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem<Self> for adr {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self - ComplexField::floor(self/rhs)*rhs
    }
}
impl RemAssign<Self> for adr {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Neg for adr {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let out_value = self.value().neg();
        self.computation_graph.add_node(NodeType::Neg, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl Float for adr {
    fn nan() -> Self {
        Self::constant(f64::NAN)
    }

    fn infinity() -> Self {
        Self::constant(f64::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self::constant(f64::NEG_INFINITY)
    }

    fn neg_zero() -> Self { -Self::zero() }

    fn min_value() -> Self { Self::constant(f64::MIN) }

    fn min_positive_value() -> Self {
        Self::constant(f64::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }

    fn is_nan(self) -> bool { self.value().is_nan() }

    fn is_infinite(self) -> bool {
        self.value().is_infinite()
    }

    fn is_finite(self) -> bool {
        self.value().is_finite()
    }

    fn is_normal(self) -> bool {
        self.value().is_normal()
    }

    fn classify(self) -> FpCategory {
        self.value().classify()
    }

    fn floor(self) -> Self { ComplexField::floor(self) }

    fn ceil(self) -> Self {
        ComplexField::ceil(self)
    }

    fn round(self) -> Self {
        ComplexField::round(self)
    }

    fn trunc(self) -> Self {
        ComplexField::trunc(self)
    }

    fn fract(self) -> Self {
        ComplexField::fract(self)
    }

    fn abs(self) -> Self {
        ComplexField::abs(self)
    }

    fn signum(self) -> Self {
        ComplexField::signum(self)
    }

    fn is_sign_positive(self) -> bool { RealField::is_sign_positive(&self) }

    fn is_sign_negative(self) -> bool { RealField::is_sign_negative(&self) }

    fn mul_add(self, a: Self, b: Self) -> Self { ComplexField::mul_add(self, a, b) }

    fn recip(self) -> Self { ComplexField::recip(self) }

    fn powi(self, n: i32) -> Self {
        ComplexField::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        ComplexField::powf(self, n)
    }

    fn sqrt(self) -> Self {
        ComplexField::sqrt(self)
    }

    fn exp(self) -> Self {
        ComplexField::exp(self)
    }

    fn exp2(self) -> Self {
        ComplexField::exp2(self)
    }

    fn ln(self) -> Self {
        ComplexField::ln(self)
    }

    fn log(self, base: Self) -> Self {
        ComplexField::log(self, base)
    }

    fn log2(self) -> Self {
        ComplexField::log2(self)
    }

    fn log10(self) -> Self {
        ComplexField::log10(self)
    }

    fn max(self, other: Self) -> Self {
        RealField::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        RealField::min(self, other)
    }

    fn abs_sub(self, other: Self) -> Self {
        Signed::abs_sub(&self, &other)
    }

    fn cbrt(self) -> Self { ComplexField::cbrt(self) }

    fn hypot(self, other: Self) -> Self {
        ComplexField::hypot(self, other)
    }

    fn sin(self) -> Self {
        ComplexField::sin(self)
    }

    fn cos(self) -> Self {
        ComplexField::cos(self)
    }

    fn tan(self) -> Self {
        ComplexField::tan(self)
    }

    fn asin(self) -> Self {
        ComplexField::asin(self)
    }

    fn acos(self) -> Self {
        ComplexField::acos(self)
    }

    fn atan(self) -> Self {
        ComplexField::atan(self)
    }

    fn atan2(self, other: Self) -> Self {
        RealField::atan2(self, other)
    }

    fn sin_cos(self) -> (Self, Self) {
        ComplexField::sin_cos(self)
    }

    fn exp_m1(self) -> Self {
        ComplexField::exp_m1(self)
    }

    fn ln_1p(self) -> Self {
        ComplexField::ln_1p(self)
    }

    fn sinh(self) -> Self {
        ComplexField::sinh(self)
    }

    fn cosh(self) -> Self {
        ComplexField::cosh(self)
    }

    fn tanh(self) -> Self {
        ComplexField::tanh(self)
    }

    fn asinh(self) -> Self {
        ComplexField::asinh(self)
    }

    fn acosh(self) -> Self {
        ComplexField::acosh(self)
    }

    fn atanh(self) -> Self {
        ComplexField::atanh(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) { return self.value().integer_decode() }
}

impl NumCast for adr {
    fn from<T: ToPrimitive>(_n: T) -> Option<Self> { unimplemented!() }
}

impl ToPrimitive for adr {
    fn to_i64(&self) -> Option<i64> {
        self.value().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value().to_u64()
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl PartialEq for adr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl PartialOrd for adr {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

impl Display for adr {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl From<f64> for adr {
    fn from(value: f64) -> Self { GlobalComputationGraph::get().spawn_value(value) }
}
impl Into<f64> for adr {
    fn into(self) -> f64 { self.value() }
}
impl From<f32> for adr {
    fn from(value: f32) -> Self {
        GlobalComputationGraph::get().spawn_value(value as f64)
    }
}
impl Into<f32> for adr {
    fn into(self) -> f32 {
        self.value() as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl  UlpsEq for adr {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl  AbsDiffEq for adr {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if ComplexField::abs(diff.value()) < epsilon.value() {
            true
        } else {
            false
        }
    }
}

impl  RelativeEq for adr {
    fn default_max_relative() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, _max_relative: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if ComplexField::abs(diff.value()) < epsilon.value() {
            true
        } else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl SimdValue for adr {
    type Element = Self;
    type SimdBool = bool;

    fn lanes() -> usize { 4 }

    fn splat(val: Self::Element) -> Self {
        val
    }

    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adr, R, C>> Mul<Matrix<adr, R, C, S>> for adr {
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

/*
impl< R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for adr2 {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}
*/

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adr, R, C>> Mul<&Matrix<adr, R, C, S>> for adr {
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: &Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

/*
impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<&Matrix<f64, R, C, S>> for adr2 {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: &Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}
*/

impl<D: DimName> Mul<OPoint<adr, D>> for adr where DefaultAllocator: nalgebra::allocator::Allocator<adr, D> {
    type Output = OPoint<adr, D>;

    fn mul(self, rhs: OPoint<adr, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<D: DimName> Mul<&OPoint<adr, D>> for adr where DefaultAllocator: nalgebra::allocator::Allocator<adr, D> {
    type Output = OPoint<adr, D>;

    fn mul(self, rhs: &OPoint<adr, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Zero for adr {
    #[inline]
    fn zero() -> Self {
        return Self::constant(0.0)
    }

    fn is_zero(&self) -> bool {
        return self.value() == 0.0;
    }
}

impl One for adr {
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl Num for adr {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::constant(val))
    }
}

impl Signed for adr {

    fn abs(&self) -> Self {
        let out_value = self.value().abs();
        self.computation_graph.add_node(NodeType::Abs, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        return if self.value() <= other.value() {
            Self::constant(0.0)
        } else {
            *self - *other
        };
    }

    fn signum(&self) -> Self {
        let out_value = self.value().signum();
        self.computation_graph.add_node(NodeType::Signum, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    fn is_positive(&self) -> bool {
        return self.value() > 0.0;
    }

    fn is_negative(&self) -> bool {
        return self.value() < 0.0;
    }
}

impl FromPrimitive for adr {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }
}

impl Bounded for adr {
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }

    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl RealField for adr {
    fn is_sign_positive(&self) -> bool {
        return self.is_positive();
    }

    fn is_sign_negative(&self) -> bool {
        return self.is_negative();
    }

    fn copysign(self, sign: Self) -> Self {
        return if sign.is_positive() {
            ComplexField::abs(self)
        } else {
            -ComplexField::abs(self)
        };
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let out_value = self.value().max(other.value());
        self.computation_graph.add_node(NodeType::Max, out_value, Some(self.value()), Some(other.value()), Some(self.node_idx), Some(other.node_idx))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let out_value = self.value().min(other.value());
        self.computation_graph.add_node(NodeType::Min, out_value, Some(self.value()), Some(other.value()), Some(self.node_idx), Some(other.node_idx))
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min.value() <= max.value());
        return RealField::min(RealField::max(self, min), max);
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let out_value = self.value().atan2(other.value());
        self.computation_graph.add_node(NodeType::Atan2, out_value, Some(self.value()), Some(other.value()), Some(self.node_idx), Some(other.node_idx))
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    #[inline]
    fn pi() -> Self {
        Self::constant(std::f64::consts::PI)
    }

    #[inline]
    fn two_pi() -> Self {
        Self::constant(2.0 * std::f64::consts::PI)
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_2)
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_3)
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_4)
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_6)
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_8)
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_1_PI)
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_PI)
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_SQRT_PI)
    }

    #[inline]
    fn e() -> Self {
        Self::constant(std::f64::consts::E)
    }

    #[inline]
    fn log2_e() -> Self {
        Self::constant(std::f64::consts::LOG2_E)
    }

    #[inline]
    fn log10_e() -> Self {
        Self::constant(std::f64::consts::LOG10_E)
    }

    #[inline]
    fn ln_2() -> Self {
        Self::constant(std::f64::consts::LN_2)
    }

    #[inline]
    fn ln_10() -> Self {
        Self::constant(std::f64::consts::LN_10)
    }
}

impl ComplexField for adr {
    type RealField = Self;

    fn from_real(re: Self::RealField) -> Self { re.clone() }

    fn real(self) -> <Self as ComplexField>::RealField { self.clone() }

    fn imaginary(self) -> Self::RealField { Self::zero() }

    fn modulus(self) -> Self::RealField { return ComplexField::abs(self); }

    fn modulus_squared(self) -> Self::RealField { self * self }

    fn argument(self) -> Self::RealField { unimplemented!(); }

    fn norm1(self) -> Self::RealField { return ComplexField::abs(self); }

    fn scale(self, factor: Self::RealField) -> Self { return self * factor; }

    fn unscale(self, factor: Self::RealField) -> Self { return self / factor; }

    #[inline]
    fn floor(self) -> Self {
        let out_value = self.value().floor();
        self.computation_graph.add_node(NodeType::Floor, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn ceil(self) -> Self {
        let out_value = self.value().ceil();
        self.computation_graph.add_node(NodeType::Ceil, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn round(self) -> Self {
        let out_value = self.value().round();
        self.computation_graph.add_node(NodeType::Round, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn trunc(self) -> Self {
        let out_value = self.value().trunc();
        self.computation_graph.add_node(NodeType::Trunc, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn fract(self) -> Self {
        let out_value = self.value().fract();
        self.computation_graph.add_node(NodeType::Fract, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self { return (self * a) + b; }

    #[inline]
    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        return ComplexField::sqrt(ComplexField::powi(self, 2) + ComplexField::powi(other, 2));
    }

    #[inline]
    fn recip(self) -> Self { return Self::constant(1.0) / self; }

    #[inline]
    fn conjugate(self) -> Self { return self; }

    #[inline]
    fn sin(self) -> Self {
        let out_value = self.value().sin();
        self.computation_graph.add_node(NodeType::Sin, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn cos(self) -> Self {
        let out_value = self.value().cos();
        self.computation_graph.add_node(NodeType::Cos, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        return (ComplexField::sin(self), ComplexField::cos(self));
    }

    #[inline]
    fn tan(self) -> Self {
        let out_value = self.value().tan();
        self.computation_graph.add_node(NodeType::Tan, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn asin(self) -> Self {
        let out_value = self.value().asin();
        self.computation_graph.add_node(NodeType::Asin, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn acos(self) -> Self {
        let out_value = self.value().acos();
        self.computation_graph.add_node(NodeType::Acos, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn atan(self) -> Self {
        let out_value = self.value().atan();
        self.computation_graph.add_node(NodeType::Atan, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn sinh(self) -> Self {
        let out_value = self.value().sinh();
        self.computation_graph.add_node(NodeType::Sinh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn cosh(self) -> Self {
        let out_value = self.value().cosh();
        self.computation_graph.add_node(NodeType::Cosh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn tanh(self) -> Self {
        let out_value = self.value().tanh();
        self.computation_graph.add_node(NodeType::Tanh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn asinh(self) -> Self {
        let out_value = self.value().asinh();
        self.computation_graph.add_node(NodeType::Asinh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn acosh(self) -> Self {
        let out_value = self.value().acosh();
        self.computation_graph.add_node(NodeType::Acosh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn atanh(self) -> Self {
        let out_value = self.value().atanh();
        self.computation_graph.add_node(NodeType::Atanh, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        let out_value = self.value().log(base.value());
        self.computation_graph.add_node(NodeType::Log, out_value, Some(self.value()), Some(base.value()), Some(self.node_idx), Some(base.node_idx))
    }

    #[inline]
    fn log2(self) -> Self { return ComplexField::log(self, Self::constant(2.0)); }

    #[inline]
    fn log10(self) -> Self { return ComplexField::log(self, Self::constant(10.0)); }

    #[inline]
    fn ln(self) -> Self { return ComplexField::log(self, Self::constant(std::f64::consts::E)); }

    #[inline]
    fn ln_1p(self) -> Self { ComplexField::ln(Self::constant(1.0) + self) }

    #[inline]
    fn sqrt(self) -> Self {
        let out_value = self.value().sqrt();
        self.computation_graph.add_node(NodeType::Sqrt, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn exp(self) -> Self {
        let out_value = self.value().exp();
        self.computation_graph.add_node(NodeType::Exp, out_value, Some(self.value()), None, Some(self.node_idx), None)
    }

    #[inline]
    fn exp2(self) -> Self { ComplexField::powf(Self::constant(2.0), self) }

    #[inline]
    fn exp_m1(self) -> Self { return ComplexField::exp(self) - Self::constant(1.0); }

    #[inline]
    fn powi(self, n: i32) -> Self { return ComplexField::powf(self, Self::constant(n as f64)); }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        let out_value = self.value().powf(n.value());
        self.computation_graph.add_node(NodeType::Powf, out_value, Some(self.value()), Some(n.value()), Some(self.node_idx), Some(n.node_idx))
    }

    #[inline]
    fn powc(self, n: Self) -> Self { return ComplexField::powf(self, n); }

    #[inline]
    fn cbrt(self) -> Self { return ComplexField::powf(self, Self::constant(1.0 / 3.0)); }

    fn is_finite(&self) -> bool { return self.value().is_finite(); }

    fn try_sqrt(self) -> Option<Self> {
        Some(ComplexField::sqrt(self))
    }
}

impl SubsetOf<Self> for adr {
    fn to_superset(&self) -> Self {
        self.clone()
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &adr) -> bool {
        true
    }
}

impl Field for adr {}

impl PrimitiveSimdValue for adr {}

impl SubsetOf<adr> for f32 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as f32
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for f64 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value()
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for u32 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as u32
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for u64 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as u64
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for u128 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as u128
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for i32 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as i32
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for i64 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as i64
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}

impl SubsetOf<adr> for i128 {
    fn to_superset(&self) -> adr {
        adr::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr) -> Self {
        element.value() as i128
    }

    fn is_in_subset(_: &adr) -> bool {
        false
    }
}