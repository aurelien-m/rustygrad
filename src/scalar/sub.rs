use std::cell::RefCell;
use std::ops::Sub;
use std::rc::Rc;

use super::Scalar;
use super::ScalarData;

impl Sub for Scalar {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() - other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), -scalar.grad()),
        })))
    }
}

impl<'a, 'b> Sub<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn sub(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() - other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), -scalar.grad()),
        })))
    }
}

impl Sub<f32> for Scalar {
    type Output = Self;
    fn sub(self, other: f32) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() - other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), -scalar.grad()),
        })))
    }
}

impl<'a, 'b> Sub<f32> for &'a Scalar {
    type Output = Scalar;

    fn sub(self, other: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() - other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.grad(), -scalar.grad()),
        })))
    }
}
