mod engine;

fn main() {
    let a = engine::Value::new(3.5);
    println!("a: {}", a);

    let b = engine::Value::new(5.0);
    println!("b: {}", b);

    println!("a + b = {}", a + b);
}
