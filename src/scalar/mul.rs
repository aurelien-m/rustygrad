use std::cell::RefCell;
use std::ops::Mul;
use std::rc::Rc;

use super::Scalar;
use super::ScalarData;

impl Mul for Scalar {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() * other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| {
                if scalar.is_left_child_none() || scalar.is_right_child_none() {
                    return (0.0, 0.0);
                }
                (
                    scalar.right_child().unwrap().data() * scalar.grad(),
                    scalar.left_child().unwrap().data() * scalar.grad(),
                )
            },
        })))
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn mul(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() * other.data(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(other.clone()),
            visited_in_backprop: false,
            compute_grad: |scalar| {
                if scalar.is_left_child_none() || scalar.is_right_child_none() {
                    return (0.0, 0.0);
                }
                (
                    scalar.right_child().unwrap().data() * scalar.grad(),
                    scalar.left_child().unwrap().data() * scalar.grad(),
                )
            },
        })))
    }
}

impl Mul<f32> for Scalar {
    type Output = Self;
    fn mul(self, other: f32) -> Self {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() * other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| {
                if scalar.is_left_child_none() || scalar.is_right_child_none() {
                    return (0.0, 0.0);
                }
                (
                    scalar.right_child().unwrap().data() * scalar.grad(),
                    scalar.left_child().unwrap().data() * scalar.grad(),
                )
            },
        })))
    }
}

impl<'a, 'b> Mul<f32> for &'a Scalar {
    type Output = Scalar;

    fn mul(self, other: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data() * other,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: Some(Scalar::new(other)),
            visited_in_backprop: false,
            compute_grad: |scalar| {
                if scalar.is_left_child_none() || scalar.is_right_child_none() {
                    return (0.0, 0.0);
                }
                (
                    scalar.right_child().unwrap().data() * scalar.grad(),
                    scalar.left_child().unwrap().data() * scalar.grad(),
                )
            },
        })))
    }
}
