use std::cell::RefCell;
use std::rc::Rc;

use super::Scalar;
use super::ScalarData;

impl Scalar {
    pub fn tanh(self) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data().tanh(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: None,
            visited_in_backprop: false,
            compute_grad: |scalar| ((1.0 - scalar.data() * scalar.data()) * scalar.grad(), 0.0),
        })))
    }

    pub fn exp(&self) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data().exp(),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: None,
            visited_in_backprop: false,
            compute_grad: |scalar| (scalar.data() * scalar.grad(), 0.0),
        })))
    }

    pub fn powf(self, power: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.data().powf(power),
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: None,
            visited_in_backprop: false,
            compute_grad: |scalar| {
                if scalar.is_left_child_none() || scalar.is_right_child_none() {
                    return (0.0, 0.0);
                }

                let rchild = scalar.right_child().unwrap();
                let lchild = scalar.left_child().unwrap();

                (
                    rchild.data() * lchild.data().powf(rchild.data() - 1.0) * scalar.grad(),
                    lchild.data() * rchild.data().powf(lchild.data() - 1.0) * scalar.grad(),
                )
            },
        })))
    }
}
