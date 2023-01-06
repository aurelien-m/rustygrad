mod nn;
mod scalar;
use scalar::Scalar;

#[macro_use]
extern crate is_close;

fn main() {
    // Autograd example
    let x1 = Scalar::new(2.0);
    let x2 = Scalar::new(0.0);

    let w1 = Scalar::new(-3.0);
    let w2 = Scalar::new(1.0);

    let b = Scalar::new(6.8813735870195432);

    let n = (&x1 * &w1) + (&x2 * &w2) + b;
    let e = (n * 2.0).exp();
    let mut output = (&e - 1.0) / (&e + 1.0);

    output.backward();

    assert!(is_close!(x1.grad(), -1.5, abs_tol = 1e-5));
    assert!(is_close!(x2.grad(), 0.5, abs_tol = 1e-5));
    assert!(is_close!(w1.grad(), 1.0, abs_tol = 1e-5));
    assert!(is_close!(w2.grad(), 0.0, abs_tol = 1e-5));

    // Tiny neural network example
    let x = vec![
        scalar::svec![2.0, 3.0, -1.0],
        scalar::svec![3.0, -1.0, 0.5],
        scalar::svec![0.5, 1.0, 1.0],
        scalar::svec![1.0, 1.0, -1.0],
    ];
    let y = scalar::svec![1.0, -1.0, -1.0, 1.0];

    let model = nn::Sequential::new(vec![
        nn::Linear::new(3, 4),
        nn::Tanh::new(),
        nn::Linear::new(4, 4),
        nn::Tanh::new(),
        nn::Linear::new(4, 1),
        nn::Tanh::new(),
    ]);

    let iterations = 100;
    for i in 0..iterations {
        let mut predictions = Vec::new();
        for a in &x {
            let out = model.forward(a);
            predictions.push(out[0].clone());
        }

        let mut loss = scalar::Scalar::new(0.0);
        for i in 0..y.len() {
            loss = loss + (y[i].clone() - predictions[i].clone()).powf(2.0);
        }

        model.zero_grad();
        loss.backward();

        // println!("{:?}", model.parameters());
        for param in model.parameters() {
            param.set_data(param.data() - 0.1 * param.grad());
        }
        println!("{}, {}", model.parameters()[0].data(), model.parameters()[0].grad());

        println!("{}/{} | loss: {}", i, iterations, loss.data());
    }

    for i in 0..x.len() {
        let pred = model.forward(&x[i]);
        println!("Pred: {} | Ground truth: {}", pred[0].data(), y[i].data());
        assert!(is_close!(pred[0].data(), y[i].data(), abs_tol = 0.1));
    }
}
