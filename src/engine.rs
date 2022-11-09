use std::fmt;
use std::ops::{Add, Mul};
use std::vec::Vec;

enum Ops {
    New,
    Add,
    Mul,
}

pub struct Value {
    data: f32,
    ops: Ops,
    children: Vec<Value>,
}

impl Value {
    pub fn new(value: f32) -> Value {
        Value {
            data: value,
            ops: Ops::New,
            children: Vec::new(),
        }
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, right_value: Self) -> Self {
        Value {
            data: self.data + right_value.data,
            ops: Ops::Add,
            children: vec![self, right_value],
        }
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, right_value: Self) -> Self {
        Value {
            data: self.data * right_value.data,
            ops: Ops::Mul,
            children: vec![self, right_value],
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}
