
#[derive(Debug)]
pub struct Tester {
    pub ta: TesterA,
    pub tb: TesterB
}

#[derive(Debug)]
pub struct TesterA {
    pub float: f64,
}

#[derive(Debug)]
pub struct TesterB {
    pub float: f32,
}

#[macro_export]
macro_rules! test {
    ($object: expr, $field: tt, $new_value: expr) => {
        $object.ta.$field = $new_value.into();
        $object.tb.$field = $new_value.into();
    }
}

fn main() {
    let mut tt = Tester { ta: TesterA { float: 0.0 }, tb: TesterB { float: 0.0 } };

    test!(&mut tt, float, 5.0);

    println!("{:?}", tt);
}