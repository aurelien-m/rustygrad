use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

use super::ScalarData;
use super::Scalar;

impl Add for Scalar {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() + other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), scalar.grad()),
        })))
    }
}

impl<'a, 'b> Add<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn add(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() + other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), scalar.grad()),
        })))
    }
}

impl Add<f32> for Scalar {
    type Output = Self;
    fn add(self, other: f32) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() + other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), scalar.grad()),
        })))
    }
}

impl<'a, 'b> Add<f32> for &'a Scalar {
    type Output = Scalar;

    fn add(self, other: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() + other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), scalar.grad()),
        })))
    }
}
