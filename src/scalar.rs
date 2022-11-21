use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Ops {
    Add,
    Mul,
    Tanh,
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
}

fn compute_grad(scalar: &mut ScalarData) {
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
        Ops::Tanh => {
            if scalar.left_child.is_none() {
                return;
            }

            let left_child = &mut scalar.left_child.as_ref().unwrap().0.borrow_mut();
            left_child.grad += (1.0 - scalar.data * scalar.data) * scalar.grad;
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
        })))
    }

    pub fn backward(&mut self) {
        let mut scalar = self.0.borrow_mut();
        scalar.grad = 1.0;

        compute_grad(&mut scalar);
        println!("{}", scalar);

        let mut nodes = Vec::new();
        if !scalar.left_child.is_none() {
            nodes.push(scalar.left_child.clone().unwrap());
        }
        if !scalar.right_child.is_none() {
            nodes.push(scalar.right_child.clone().unwrap());
        }

        while nodes.len() > 0 {
            let child = nodes.pop().unwrap();

            compute_grad(&mut child.0.borrow_mut());
            println!("{}", child);

            if !child.0.borrow().left_child.is_none() {
                nodes.push(child.0.borrow().left_child.clone().unwrap());
            }
            if !child.0.borrow().right_child.is_none() {
                nodes.push(child.0.borrow().right_child.clone().unwrap());
            }
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
        })))
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scalar = self.0.borrow();
        write!(
            f,
            "Scalar(data={}, grad={}, ops={:?})",
            scalar.data, scalar.grad, scalar.ops,
        )
    }
}

impl fmt::Display for ScalarData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scalar(data={}, grad={}, ops={:?})",
            self.data, self.grad, self.ops,
        )
    }
}
