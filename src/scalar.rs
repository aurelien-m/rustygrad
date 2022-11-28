use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Div, Mul};
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Ops {
    Add,
    Mul,
    Div,
    Tanh,
    Exp,
    Pow,
    Nil,
}

#[derive(Clone)]
pub struct Scalar(Rc<RefCell<ScalarData>>);

#[derive(Clone)]
pub struct ScalarData {
    data: f32,
    grad: f32,
    left_child: Option<Scalar>,
    right_child: Option<Scalar>,
    ops: Ops,
    visited_in_backprop: bool,
}

fn compute_grad(scalar: &mut ScalarData) {
    if scalar.visited_in_backprop {
        scalar.visited_in_backprop = false;
    }

    match scalar.ops {
        Ops::Add => {
            if scalar.left_child.is_none() || scalar.right_child.is_none() {
                return;
            }

            scalar.left_child.as_ref().unwrap().0.borrow_mut().grad += scalar.grad;
            scalar.right_child.as_ref().unwrap().0.borrow_mut().grad += scalar.grad;
        }
        Ops::Mul => {
            if scalar.left_child.is_none() || scalar.right_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            let right_child = &mut scalar.right_child.as_ref().unwrap().0.borrow_mut();

            left_child.grad += right_child.data * scalar.grad;
            right_child.grad += left_child.data * scalar.grad;
        }
        Ops::Div => {
            if scalar.left_child.is_none() || scalar.right_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            let right_child = &mut scalar.right_child.as_ref().unwrap().0.borrow_mut();

            left_child.grad += scalar.grad / right_child.data;
            right_child.grad += -(scalar.grad * left_child.data) / (right_child.data.powi(2));
            // left_child.grad += -(1.0 / right_child.data.powi(2)) * scalar.grad;
            // right_child.grad += -(1.0 / left_child.data.powi(2)) * scalar.grad;
        }
        Ops::Tanh => {
            if scalar.left_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            left_child.grad += (1.0 - scalar.data * scalar.data) * scalar.grad;
        }
        Ops::Exp => {
            if scalar.left_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            left_child.grad += scalar.data * scalar.grad;
        }
        Ops::Pow => {
            if scalar.left_child.is_none() || scalar.right_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            let right_child = &mut scalar.right_child.as_ref().unwrap().0.borrow_mut();

            left_child.grad +=
                right_child.data * left_child.data.powf(right_child.data - 1.0) * scalar.grad;
            right_child.grad +=
                left_child.data * right_child.data.powf(left_child.data - 1.0) * scalar.grad;
        }
        Ops::Nil => {}
    }
}

impl Scalar {
    pub fn new(data: f32) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: None,
            right_child: None,
            ops: Ops::Nil,
            visited_in_backprop: false,
        })))
    }

    pub fn tanh(self) -> Scalar {
        let data = self.0.borrow().data.tanh();
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self),
            right_child: None,
            ops: Ops::Tanh,
            visited_in_backprop: false,
        })))
    }

    pub fn exp(&self) -> Scalar {
        let data = self.0.borrow().data.exp();
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self.clone()),
            right_child: None,
            ops: Ops::Exp,
            visited_in_backprop: false,
        })))
    }

    pub fn powf(self, power: f32) -> Scalar {
        let data = self.0.borrow().data.powf(power);
        let other = Scalar(Rc::new(RefCell::new(ScalarData {
            data: power,
            grad: 0.0,
            left_child: None,
            right_child: None,
            ops: Ops::Nil,
            visited_in_backprop: false,
        })));

        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self),
            right_child: Some(other),
            ops: Ops::Pow,
            visited_in_backprop: false,
        })))
    }

    pub fn backward(&mut self) {
        // TODO: Check if there is a non-recursive way of doing this

        let mut ordered_graph = Vec::new();
        fn back(scalar: Scalar, ordered_graph: &mut Vec<Scalar>) {
            if !scalar.0.borrow().visited_in_backprop {
                scalar.0.borrow_mut().visited_in_backprop = true;

                if !scalar.0.borrow().right_child.is_none() {
                    back(scalar.0.borrow().right_child.clone().unwrap(), ordered_graph);
                }
                if !scalar.0.borrow().left_child.is_none() {
                    back(scalar.0.borrow().left_child.clone().unwrap(), ordered_graph);
                }

                ordered_graph.push(scalar);
            }
        }
        back(self.clone(), &mut ordered_graph);

        self.0.borrow_mut().grad = 1.0;
        while ordered_graph.len() > 0 {
            let s = ordered_graph.pop().unwrap();
            compute_grad(&mut s.0.borrow_mut());
            println!("{}", s);
        }
    }
}

impl Add for Scalar {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let data = self.0.borrow().data + other.0.borrow().data;
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self),
            right_child: Some(other),
            ops: Ops::Add,
            visited_in_backprop: false,
        })))
    }
}

impl<'a, 'b> Add<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn add(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            left_child: Some(Scalar(Rc::clone(&self.0))),
            right_child: Some(Scalar(Rc::clone(&other.0))),
            ops: Ops::Add,
            visited_in_backprop: false,
        })))
    }
}

impl Mul for Scalar {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let data = self.0.borrow().data * other.0.borrow().data;
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self),
            right_child: Some(other),
            ops: Ops::Mul,
            visited_in_backprop: false,
        })))
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn mul(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            left_child: Some(Scalar(Rc::clone(&self.0))),
            right_child: Some(Scalar(Rc::clone(&other.0))),
            ops: Ops::Mul,
            visited_in_backprop: false,
        })))
    }
}

impl Div for Scalar {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let data = self.0.borrow().data / other.0.borrow().data;
        Scalar(Rc::new(RefCell::new(ScalarData {
            data,
            grad: 0.0,
            left_child: Some(self),
            right_child: Some(other),
            ops: Ops::Div,
            visited_in_backprop: false,
        })))
    }
}

impl<'a, 'b> Div<&'b Scalar> for &'a Scalar {
    type Output = Scalar;

    fn div(self, other: &'b Scalar) -> Scalar {
        Scalar(Rc::new(RefCell::new(ScalarData {
            data: self.0.borrow().data / other.0.borrow().data,
            grad: 0.0,
            left_child: Some(Scalar(Rc::clone(&self.0))),
            right_child: Some(Scalar(Rc::clone(&other.0))),
            ops: Ops::Div,
            visited_in_backprop: false,
        })))
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scalar = self.0.borrow();
        write!(
            f,
            "Scalar(data={}, grad={}, ops={:?}, vis_in_b={})",
            scalar.data, scalar.grad, scalar.ops, scalar.visited_in_backprop,
        )
    }
}

impl fmt::Display for ScalarData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scalar(data={}, grad={}, ops={:?}, vis_in_b={})",
            self.data, self.grad, self.ops, self.visited_in_backprop,
        )
    }
}
