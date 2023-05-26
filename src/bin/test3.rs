

#[macro_export]
macro_rules! test_1 {
    () => { println!("hello"); }
}

#[macro_export]
macro_rules! test_2 {
    () => { println!("yep"); }
}

#[macro_export]
macro_rules! test {
    ($t: tt) => {  $t!();  }
}

fn main() {
    test!(test_1);
}