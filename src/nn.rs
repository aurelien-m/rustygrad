use crate::scalar::Scalar;
use rand::Rng;

pub struct Neuron {
    weights: Vec<Scalar>,
    bias: Scalar,
}

impl Neuron {
    pub fn new(input_count: i32) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        for _ in 0..input_count {
            weights.push(Scalar::new(rng.gen_range(-1.0..1.0)));
        }

        let bias = Scalar::new(rng.gen_range(-1.0..1.0));

        Neuron { weights, bias }
    }

    pub fn forward(&self, x: Vec<Scalar>) -> Scalar {
        let mut out = Scalar::new(self.bias.data());
        for i in 0..self.weights.len() {
            out = out + (self.weights[i].clone() * x[i].clone());
        }
        out
    }

    pub fn zero_grad(&self) {
        for weight in &self.weights {
            weight.zero_grad();
        }
        self.bias.zero_grad();
    }

    pub fn parameters(&self) -> Vec<Scalar> {
        let mut out = self.weights.clone();
        out.push(self.bias.clone());
        out
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }

    pub fn forward(&self, x: &Vec<Scalar>) -> Vec<Scalar> {
        let mut x = x.clone();
        for layer in &self.layers {
            // println!("Previous x: {x:?}"); // TODO: print when using a verbose mode
            x = layer.forward(x);
            // println!("Computed x: {x:?}\n"); // TODO: same here
        }
        x
    }

    pub fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }

    pub fn parameters(&self) -> Vec<Scalar> {
        let mut out = Vec::new();
        for layer in &self.layers {
            out.append(&mut layer.parameters());
        }
        out
    }
}

pub trait Module {
    fn forward(&self, x: Vec<Scalar>) -> Vec<Scalar>;
    fn zero_grad(&self);
    fn parameters(&self) -> Vec<Scalar>;
}

pub struct Linear {
    neurons: Vec<Neuron>,
}

impl Linear {
    pub fn new(input_count: i32, output_count: i32) -> Box<Linear> {
        let mut neurons = Vec::new();
        for _ in 0..output_count {
            neurons.push(Neuron::new(input_count))
        }
        Box::new(Linear { neurons })
    }
}

impl Module for Linear {
    fn forward(&self, x: Vec<Scalar>) -> Vec<Scalar> {
        let mut out = Vec::new();
        for neuron in &self.neurons {
            out.push(neuron.forward(x.clone()))
        }
        out
    }

    fn zero_grad(&self) {
        for neuron in &self.neurons {
            neuron.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Scalar> {
        let mut params = Vec::new();
        for neuron in &self.neurons {
            params.append(&mut neuron.parameters());
        }
        params
    }
}

pub struct Tanh {}

impl Tanh {
    pub fn new() -> Box<Tanh> {
        Box::new(Tanh {})
    }
}

impl Module for Tanh {
    fn forward(&self, x: Vec<Scalar>) -> Vec<Scalar> {
        let mut out = Vec::new();
        for item in x {
            out.push(item.clone().tanh())
        }
        out
    }

    fn zero_grad(&self) {}
    fn parameters(&self) -> Vec<Scalar> {
        Vec::new()
    }
}
