#![feature(type_alias_impl_trait)]

use ad_trait::AD;

pub type T = impl AD;

pub fn test(a: T, b: T) -> T {
    a + b
}

fn main() {
    let res = test(1.0 as f32, 2.0);
    println!("{:?}", res);

    let res = test(1.0, 2.0);
    println!("{:?}", res);
}
