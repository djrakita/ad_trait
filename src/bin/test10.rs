use as_any::{AsAny, Downcast};

struct Test;

trait Custom: AsAny {
    // whatever you like to put inside of your trait
}

impl Custom for Test {}

fn main() {
    let x = Test;
    // let y: &dyn Custom = &x;
    let y: Box<dyn Custom> = Box::new(x);
    // With (extension) trait `Downcast` in scope.
    let res = y.as_ref().downcast_ref::<Test>();
    // let res = (*y).downcast_ref::<Test>();
    println!("{:?}", res.is_some());
}