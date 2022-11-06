use std::fmt;
use std::ops;

pub struct Value {
    data: f32,
}

impl Value {
    pub fn new(value: f32) -> Value {
        Value { data: value }
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, _right_value: Value) -> Value {
        Value {
            data: self.data + _right_value.data,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}
