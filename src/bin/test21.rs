use simba::simd::f32x4;
use ad_trait::forward_ad::adf::adf_f32x4;
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;
use ad_trait::simd::f64xn::f64xn;

fn main() {
    let a = adf_f32x4::new(1.0, f32x4::new(1., 2., 3., 4.));

    let res = serde_json::to_string(&a);
    println!("{:?}", res);

    let res2: Result<adf_f32x4, _> = serde_json::from_str(&res.unwrap());
    println!("{:?}", res2);

    let a = adfn::new(1.0, [1.,2.]);
    let res = serde_json::to_string(&a);
    println!("{:?}", res);

    let res2 = serde_json::from_str::<adfn<2>>(&res.unwrap());
    println!("{:?}", res2);

    let a = adr::new_variable(1.0, true);
    let res = serde_json::to_string(&a);
    println!("{:?}", res);

    let res2 = serde_json::from_str::<adr>(&res.unwrap());
    println!("{:?}", res2);

    let a = f64xn::new([1.,2.,3.]);
    let res = serde_json::to_string(&a);
    println!("{:?}", res);

    let res2 = serde_json::from_str::<f64xn<3>>(&res.unwrap());
    println!("{:?}", res2);
}