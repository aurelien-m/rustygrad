mod scalar;

fn main() {
    let a = scalar::Scalar::new(10.0);
    let b = scalar::Scalar::new(20.0);

    let c = &a + &b;
    let d = &a * &b;

    let mut e = c + d;

    e.backward();
    println!("");

    // Basic neuron example
    let x1 = scalar::Scalar::new(2.0);
    let x2 = scalar::Scalar::new(0.0);

    let w1 = scalar::Scalar::new(-3.0);
    let w2 = scalar::Scalar::new(1.0);

    let b = scalar::Scalar::new(6.8813735870195432);

    let sum = (x1 * w1) + (x2 * w2) + b;
    let mut y = sum.tanh();

    println!("y: {}", y);

    y.backward();
}
