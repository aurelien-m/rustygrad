mod scalar;

fn main() {
    // Basic neuron
    let x1 = scalar::Scalar::new(2.0);
    let x2 = scalar::Scalar::new(0.0);
    
    let w1 = scalar::Scalar::new(-3.0);
    let w2 = scalar::Scalar::new(1.0);

    let b = scalar::Scalar::new(6.8813735870195432);

    let sum = (x1 * w1) + (x2 * w2) + b;
    let mut y = sum.tanh();

    println!("Output: {}", y);

    y.backward();

    println!("(a + b) * c = {}", (a + b) * c);
}
