use std::time::Instant;

fn add_vecs<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = a[i] + b[i];
    }
    out
}

fn main() {
    let mut a = [0.0001; 16];
    let b = [0.00001; 16];

    let start = Instant::now();
    for _ in 0..1000 {
        a = add_vecs(&a, &b);
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", a);
}