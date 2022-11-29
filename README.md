# Some attempt at making micrograd in Rust

Thank you lord Karpathy for making [this video](https://www.youtube.com/watch?v=VMj-3S1tku0) 🙌

#### What works

```Rust
let x1 = scalar::Scalar::new(2.0);
let x2 = scalar::Scalar::new(0.0);

let w1 = scalar::Scalar::new(-3.0);
let w2 = scalar::Scalar::new(1.0);

let b = scalar::Scalar::new(6.8813735870195432);

let n = (&x1 * &w1) + (&x2 * &w2) + b;
let e = (n * 2.0).exp();
let mut output = (&e - 1.0) / (&e + 1.0);

output.backward();

assert!(is_close!(x1.grad(), -1.5, abs_tol=1e-5));
assert!(is_close!(x2.grad(), 0.5, abs_tol=1e-5));
assert!(is_close!(w1.grad(), 1.0, abs_tol=1e-5));
assert!(is_close!(w2.grad(), 0.0, abs_tol=1e-5));

println!("output: {}", output);
```