use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8, f64x1, f64x16, f64x2, f64x32, f64x4, f64x64, f64x8};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use bevy_reflect::Reflect;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;
use crate::F64;

/// you should not use this struct, it's only used to populate the macro below
#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Copy, Reflect)]
pub struct adf_template {
    pub (crate) value: f64,
    pub (crate) tangent: f32x1
}
impl adf_template {
    pub fn new(value: f64, tangent: f32x1) -> Self {
        Self {
            value,
            tangent,
        }
    }
    pub fn new_constant(value: f64) -> Self {
        Self {
            value,
            tangent: f32x1::splat(0.0),
        }
    }
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }
    #[inline]
    pub fn tangent(&self) -> f32x1 {
        self.tangent
    }
    pub fn tangent_size() -> usize {
        1
    }
    #[inline]
    pub fn tangent_as_vec(&self) -> Vec<f32> {
        self.tangent.as_array().to_vec()
    }
}

impl Serialize for adf_template {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        let mut state = serializer.serialize_struct("adf", 2)?;
        state.serialize_field("value", &self.value)?;
        let tangent_as_vec = self.tangent_as_vec();
        state.serialize_field("tangent", &tangent_as_vec)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for adf_template {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        enum Field { Value, Tangent }

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
                            "tangent" => Ok(Field::Tangent),
                            _ => { Err(de::Error::unknown_field(value, FIELDS)) }
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct AdfnVisitor;

        impl<'de> Visitor<'de> for AdfnVisitor {
            type Value = adf_template;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("struct adfn")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<adf_template, V::Error> {
                let mut value = None;
                let mut tangent = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() { return Err(de::Error::duplicate_field("value")); }
                            value = Some(map.next_value()?);
                        }
                        Field::Tangent => {
                            if tangent.is_some() { return Err(de::Error::duplicate_field("tangent")); }
                            let tangent_as_vec = map.next_value::<Vec<f32>>()?;
                            let mut tangent_as_slice = [0.0; 1];
                            for (i, t) in tangent_as_vec.iter().enumerate() { tangent_as_slice[i] = *t; }
                            tangent = Some(tangent_as_slice);
                        }
                    }
                }

                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                let tangent = tangent.ok_or_else(|| de::Error::missing_field("tangent"))?;
                Ok(adf_template { value, tangent: f32x1::from_slice(&tangent) })
            }
        }

        const FIELDS: &'static [&'static str] = &["value", "tangent"];
        deserializer.deserialize_struct("adfn", FIELDS, AdfnVisitor)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<F64> for adf_template {
    type Output = adf_template;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        todo!()
    }
}

impl AddAssign<F64> for adf_template {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        todo!()
    }
}

impl Mul<F64> for adf_template {
    type Output = adf_template;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        todo!()
    }
}

impl MulAssign<F64> for adf_template {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        todo!()
    }
}

impl Sub<F64> for adf_template {
    type Output = adf_template;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        todo!()
    }
}

impl SubAssign<F64> for adf_template {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        todo!()
    }
}

impl Div<F64> for adf_template {
    type Output = adf_template;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        todo!()
    }
}

impl DivAssign<F64> for adf_template {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        todo!()
    }
}

impl Rem<F64> for adf_template {
    type Output = adf_template;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        todo!()
    }
}

impl RemAssign<F64> for adf_template {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        todo!()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add for adf_template {
    type Output = adf_template;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl AddAssign for adf_template {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Mul for adf_template {
    type Output = adf_template;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl MulAssign for adf_template {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Sub for adf_template {
    type Output = adf_template;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl SubAssign for adf_template {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Div for adf_template {
    type Output = adf_template;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl DivAssign for adf_template {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Rem for adf_template {
    type Output = adf_template;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl RemAssign for adf_template {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl Neg for adf_template {
    type Output = adf_template;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            tangent: f32x1::splat(-1.0) * self.tangent,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl PartialEq for adf_template {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for adf_template {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Display for adf_template {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl From<f64> for adf_template {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new_constant(value)
    }
}

impl Into<f64> for adf_template {
    #[inline]
    fn into(self) -> f64 {
        self.value
    }
}

impl From<f32> for adf_template {
    #[inline]
    fn from(value: f32) -> Self {
        Self::new_constant(value as f64)
    }
}

impl Into<f32> for adf_template {
    #[inline]
    fn into(self) -> f32 {
        self.value as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl UlpsEq for adf_template {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl AbsDiffEq for adf_template {
    type Epsilon = adf_template;

    fn default_epsilon() -> Self::Epsilon {
        Self::new_constant(0.000000001)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        todo!()
    }
}

impl RelativeEq for adf_template {
    fn default_max_relative() -> Self::Epsilon {
        Self::new_constant(0.000000001)
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        todo!()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
#[macro_export]
macro_rules! make_adf2 {
    ($tangent_type: tt, $float_type: tt, $struct_name: tt, $tangent_size: tt, $struct_name_as_str: tt) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Debug, Copy, Reflect)]
        pub struct $struct_name {
            pub (crate) value: f64,
            pub (crate) tangent: $tangent_type
        }
        impl $struct_name {
            pub fn new(value: f64, tangent: $tangent_type) -> Self {
                Self {
                    value,
                    tangent,
                }
            }
            pub fn new_constant(value: f64) -> Self {
                Self {
                    value,
                    tangent: $tangent_type::splat(0.0),
                }
            }
            #[inline]
            pub fn value(&self) -> f64 {
                self.value
            }
            #[inline]
            pub fn tangent(&self) -> $tangent_type {
                self.tangent
            }
            pub fn tangent_size() -> usize {
                $tangent_size
            }
            #[inline]
            pub fn tangent_as_vec(&self) -> Vec<$float_type> {
                self.tangent.as_array().to_vec()
            }
        }

        impl Serialize for $struct_name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
                let mut state = serializer.serialize_struct($struct_name_as_str, 2)?;
                state.serialize_field("value", &self.value)?;
                let tangent_as_vec = self.tangent_as_vec();
                state.serialize_field("tangent", &tangent_as_vec)?;
                state.end()
            }
        }

        impl<'de> Deserialize<'de> for $struct_name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                enum Field { Value, Tangent }

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
                                    "tangent" => Ok(Field::Tangent),
                                    _ => { Err(de::Error::unknown_field(value, FIELDS)) }
                                }
                            }
                        }

                        deserializer.deserialize_identifier(FieldVisitor)
                    }
                }

                struct AdfnVisitor;

                impl<'de> Visitor<'de> for AdfnVisitor {
                    type Value = $struct_name;

                    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                        formatter.write_str("struct adfn")
                    }

                    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<$struct_name, V::Error> {
                        let mut value = None;
                        let mut tangent = None;
                        while let Some(key) = map.next_key()? {
                            match key {
                                Field::Value => {
                                    if value.is_some() { return Err(de::Error::duplicate_field("value")); }
                                    value = Some(map.next_value()?);
                                }
                                Field::Tangent => {
                                    if tangent.is_some() { return Err(de::Error::duplicate_field("tangent")); }
                                    let tangent_as_vec = map.next_value::<Vec<$float_type>>()?;
                                    let mut tangent_as_slice = [0.0; $tangent_size];
                                    for (i, t) in tangent_as_vec.iter().enumerate() { tangent_as_slice[i] = *t; }
                                    tangent = Some(tangent_as_slice);
                                }
                            }
                        }

                        let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                        let tangent = tangent.ok_or_else(|| de::Error::missing_field("tangent"))?;
                        Ok($struct_name { value, tangent: $tangent_type::from_slice(&tangent) })
                    }
                }

                const FIELDS: &'static [&'static str] = &["value", "tangent"];
                deserializer.deserialize_struct($struct_name_as_str, FIELDS, AdfnVisitor)
            }
        }

        impl PartialEq for $struct_name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.value == other.value
            }
        }

        impl PartialOrd for $struct_name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.value.partial_cmp(&other.value)
            }
        }

        impl Display for $struct_name {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.write_str(&format!("{:?}", self)).expect("error");
                Ok(())
            }
        }

        impl From<f64> for $struct_name {
            #[inline]
            fn from(value: f64) -> Self {
                Self::new_constant(value)
            }
        }

        impl Into<f64> for $struct_name {
            #[inline]
            fn into(self) -> f64 {
                self.value
            }
        }

        impl From<f32> for $struct_name {
            #[inline]
            fn from(value: f32) -> Self {
                Self::new_constant(value as f64)
            }
        }

        impl Into<f32> for $struct_name {
            #[inline]
            fn into(self) -> f32 {
                self.value as f32
            }
        }
    };
}


make_adf2!(f32x1,  f32,  adf_f32x1,    1,   "adf_f32x1");
make_adf2!(f32x2,  f32,  adf_f32x2,    2,   "adf_f32x2");
make_adf2!(f32x4,  f32,  adf_f32x4,    4,   "adf_f32x4");
make_adf2!(f32x8,  f32,  adf_f32x8,    8,   "adf_f32x8");
make_adf2!(f32x16, f32,  adf_f32x16,  16,   "adf_f32x16");
make_adf2!(f32x32, f32,  adf_f32x32,  32,   "adf_f32x32");
make_adf2!(f32x64, f32,  adf_f32x64,  64,   "adf_f32x64");
make_adf2!(f64x1,  f64,  adf_f64x1,    1,   "adf_f64x1");
make_adf2!(f64x2,  f64,  adf_f64x2,    2,   "adf_f64x2");
make_adf2!(f64x4,  f64,  adf_f64x4,    4,   "adf_f64x4");
make_adf2!(f64x8,  f64,  adf_f64x8,    8,   "adf_f64x8");
make_adf2!(f64x16, f64,  adf_f64x16,  16,   "adf_f64x16");
make_adf2!(f64x32, f64,  adf_f64x32,  32,   "adf_f64x32");
make_adf2!(f64x64, f64,  adf_f64x64,  64,   "adf_f64x64");