use std::cell::RefCell;
use std::rc::Rc;

use super::Scalar;
use super::ScalarData;

impl Scalar {
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
}
