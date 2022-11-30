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

    pub fn forward(self, x: &Vec<f32>) -> Scalar {
        let bias_d = self.bias.data();
        let mut out = Scalar::new(bias_d);
        for i in 0..self.weights.len() {
            out = out + (self.weights[i].clone() * x[i]);
        }
        out.tanh()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_count: i32, output_count: i32) -> Layer {
        let mut neurons = Vec::new();
        for _ in 0..output_count {
            neurons.push(Neuron::new(input_count))
        }

        Layer { neurons }
    }

    pub fn forward(self, x: &Vec<f32>) -> Vec<Scalar> {
        let mut out = Vec::new();
        for neuron in self.neurons {
            out.push(neuron.forward(x))
        }
        out
    }
}
