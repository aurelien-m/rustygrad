use std::cell::RefCell;
use std::rc::Rc;

mod add;
mod div;
mod mul;
mod other;
mod sub;

#[derive(Clone)]
pub struct Scalar(Rc<RefCell<ScalarData>>);

pub struct ScalarData {
    data: f32,
    grad: f32,
    visited_in_backprop: bool,
    left_child: Option<Scalar>,
    right_child: Option<Scalar>,
    compute_grad: fn(&Scalar) -> (f32, f32),
}

macro_rules! svec {
    // The macro takes a list of integers as an argument
    ($($x:expr),*) => {
        // Use the vec! macro to create a Vec of MyStruct from the list of integers
        vec![$(scalar::Scalar::new($x)),*]
    }
}
pub(crate) use svec;

impl Scalar {
    pub fn new(data: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: None,
            right_child: None,
            visited_in_backprop: false,
            compute_grad: |_scalar| (0.0, 0.0),
        })))
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    fn set_grad(&self, grad: f32) {
        self.0.borrow_mut().grad = grad;
    }

    fn add_to_grad(&self, value: f32) {
        self.0.borrow_mut().grad += value;
    }

    fn left_child(&self) -> Option<Scalar> {
        self.0.borrow().left_child.clone()
    }

    fn right_child(&self) -> Option<Scalar> {
        self.0.borrow().right_child.clone()
    }

    fn visited_in_backprop(&self) -> bool {
        self.0.borrow().visited_in_backprop
    }

    fn set_visited(&self, visited: bool) {
        self.0.borrow_mut().visited_in_backprop = visited;
    }

    fn is_right_child_none(&self) -> bool {
        self.0.borrow().right_child.is_none()
    }

    fn is_left_child_none(&self) -> bool {
        self.0.borrow().left_child.is_none()
    }

    fn compute_grad(&self) -> (f32, f32) {
        (self.0.borrow().compute_grad)(self)
    }

    pub fn backward(&mut self) {
        self.set_grad(1.0);

        let mut ordered_graph = Vec::new();
        fn back(scalar: Scalar, ordered_graph: &mut Vec<Scalar>) {
            if !scalar.visited_in_backprop() {
                scalar.set_visited(true);

                if !scalar.is_left_child_none() {
                    back(scalar.left_child().unwrap(), ordered_graph);
                }
                if !scalar.is_right_child_none() {
                    back(scalar.right_child().unwrap(), ordered_graph);
                }

                ordered_graph.push(scalar);
            }
        }
        back(self.clone(), &mut ordered_graph);

        while ordered_graph.len() > 0 {
            let s = ordered_graph.pop().unwrap();
            let (left_grad, right_grad) = s.compute_grad();

            if !s.is_left_child_none() {
                s.left_child().unwrap().add_to_grad(left_grad);
            }
            if !s.is_right_child_none() {
                s.right_child().unwrap().add_to_grad(right_grad);
            }
        }
    }
}
