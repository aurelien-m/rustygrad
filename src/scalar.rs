use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::ops::{Add, Mul};

type BackwardFn = Option<fn(scalar: &mut Scalar)>; 

pub struct Scalar {
    data: f32,
    grad: f32,
    left_child: Option<Rc<RefCell<Scalar>>>,
    right_child: Option<Rc<RefCell<Scalar>>>,
    _backward: BackwardFn,
}

impl Scalar {
    pub fn new(data: f32) -> Scalar {
        Scalar {
            data,
            grad: 0.0,
            left_child: None,
            right_child: None,
            _backward: None,
        }
    }

    pub fn tanh(self) -> Scalar {
        Scalar {
            data: self.data.tanh(),
            grad: 0.0,
            left_child: Some(Rc::new(RefCell::new(self))),
            right_child: None,
            _backward: Some(|scalar: &mut Scalar| {
                let child;
                match &scalar.left_child {
                    Some(left_child) => child = left_child,
                    None => return,
                }
                child.borrow_mut().grad = (1.0 - scalar.data * scalar.data) * scalar.grad;
            }),
        }
    }

    pub fn backward(&mut self) {
        self.grad = 1.0;
        self._backward.unwrap()(self);  

        let child = self.left_child.as_ref();
        println!("{}", child.unwrap().borrow().grad);

        let mut nodes = vec![self.left_child.clone()];
        while nodes.len() > 0 {
            let child;
            match nodes.pop() {
                Some(popped_child) => child = popped_child.unwrap(),
                None => continue,
            }

            let mut scalar = child.borrow_mut();
            println!("{}", scalar);
            match scalar._backward {
                Some(_) => scalar._backward.unwrap()(&mut scalar),
                None => {},
            }

            if !scalar.left_child.is_none() {
                nodes.push(scalar.left_child.clone());
            }
            if !scalar.right_child.is_none() {
                nodes.push(scalar.right_child.clone());
            }
        }
    }
}

impl Add for Scalar {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Scalar {
            data: self.data + other.data,
            grad: 1.0,
            left_child: Some(Rc::new(RefCell::new(self))),
            right_child: Some(Rc::new(RefCell::new(other))),
            _backward: Some(|scalar: &mut Scalar| {
                let lchild;
                match &scalar.left_child {
                    Some(left_child) => lchild = left_child,
                    None => return,
                }

                let rchild;
                match &scalar.right_child {
                    Some(right_child) => rchild = right_child,
                    None => return,
                }

                lchild.borrow_mut().grad = scalar.grad;
                rchild.borrow_mut().grad = scalar.grad;
            }),
        }
    }
}

impl Mul for Scalar {
   type Output = Self;
   fn mul(self, other: Self) -> Self {
        Scalar {
            data: self.data * other.data,
            grad: 1.0,
            left_child: Some(Rc::new(RefCell::new(self))),
            right_child: Some(Rc::new(RefCell::new(other))),
            _backward: Some(|scalar: &mut Scalar| {
                let lchild;
                match &scalar.left_child {
                    Some(left_child) => lchild = left_child,
                    None => return,
                }

                let rchild;
                match &scalar.right_child {
                    Some(right_child) => rchild = right_child,
                    None => return,
                }

                lchild.borrow_mut().grad = rchild.borrow().data * scalar.grad;
                rchild.borrow_mut().grad = lchild.borrow().data * scalar.grad;
            }),
        }
   }
}

impl fmt::Display for Scalar {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       write!(f, "Scalar(data={}, grad={}, _backward_is_none={})", self.data, self.grad, self._backward.is_none())
   }
}
