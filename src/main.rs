mod engine;

fn main() {
    let a = engine::Scalar::new(3.5);
    println!("a: {}", a);

    let b = engine::Scalar::new(5.0);
    println!("b: {}", b);

    let c = engine::Scalar::new(-1.3);

    println!("(a + b) * c = {}", (a + b) * c);
}
