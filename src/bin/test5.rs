use ad_trait::reverse_ad::adr::GlobalComputationGraph;

fn main() {
    let g = GlobalComputationGraph::get();
    let a = g.spawn_value(1.0);
    let b = g.spawn_value(2.0);

    println!("{:?}", a);
    println!("{:?}", b);

    let a = GlobalComputationGraph::get().spawn_value(1.0);
    let b = GlobalComputationGraph::get().spawn_value(2.0);

    let c = a*b;
    println!("{:?}", c.get_backwards_mode_grad().wrt(&a));
}