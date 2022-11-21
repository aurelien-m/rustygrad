# Terrible attempt at making micrograd in Rust

Thank you lord Karpathy for making [this video](https://www.youtube.com/watch?v=VMj-3S1tku0) 🙌

#### What works

```Rust
let a = scalar::Scalar::new(10.0);
let b = scalar::Scalar::new(20.0);

let c = &a + &b;
let d = &a * &b;

let mut e = c + d;

e.backward();
```