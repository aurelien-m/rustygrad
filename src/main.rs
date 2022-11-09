mod engine;

fn main() {
    let a = engine::Value::new(3.5);
    println!("a: {}", a);

    let b = engine::Value::new(5.0);
    println!("b: {}", b);

    let c = engine::Value::new(-1.3);

    println!("(a + b) * c = {}", (a + b) * c);
}
