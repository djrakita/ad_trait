use num_traits::One;
use simba::scalar::RealField;
use ad_trait::forward_ad::adfn::adfn;
fn main() {
    let a = adfn::new(1.0, [1.,2.,3.]);
    let b = adfn::<3>::one().copysign(a);
    println!("{:?}", b);
}