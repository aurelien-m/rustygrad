use std::fmt;
use std::ops::{Add, Mul};
use std::vec::Vec;

enum Ops {
    New,
    Add,
    Mul,
}

pub struct Scalar {
    data: f32,
    ops: Ops,
    children: Vec<Scalar>,
}

impl Scalar {
    pub fn new(value: f32) -> Scalar {
        Scalar {
            data: value,
            ops: Ops::New,
            children: Vec::new(),
        }
    }
}

impl Add for Scalar {
    type Output = Self;
    fn add(self, right_value: Self) -> Self {
        Scalar {
            data: self.data + right_value.data,
            ops: Ops::Add,
            children: vec![self, right_value],
        }
    }
}

impl Mul for Scalar {
    type Output = Self;
    fn mul(self, right_value: Self) -> Self {
        Scalar {
            data: self.data * right_value.data,
            ops: Ops::Mul,
            children: vec![self, right_value],
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}
