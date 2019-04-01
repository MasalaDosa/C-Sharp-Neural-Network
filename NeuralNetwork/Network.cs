using MatrixLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        static Random _r = new Random();
        List<Matrix> _synapseMatrices;
        public double _learningRate;


        /// <summary>
        /// Create a neural network
        /// </summary>
        /// <param name="learningRate">The learning rate -  a good default is 0.25</param>
        /// <param name="neuronLayers">The number of neurons in each layer.  The first value is the number of input-neurons, the last value is the number of output-neurons.  Values in between represent hidden values</param>
        public Network(double learningRate, int[] neuronLayers)
        {
            if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate), $"{nameof(learningRate)} must be +ve.");
            if (neuronLayers == null) throw new ArgumentNullException(nameof(neuronLayers), $"{nameof(learningRate)} is required.");
            if (neuronLayers.Length < 2) throw new ArgumentException($"{nameof(neuronLayers)} must have at least 2 elements.", nameof(neuronLayers));

            _learningRate = learningRate;

            // Imagine a network with 3 input neurons, 3 hidden neurons and 3 output neurons:
            //
            // I1       H1      O1
            // I2       H2      O2
            // I3       H3      O3
            //
            // I1 will have three synapses with associated weights:
            //
            // I1_H1, I1_H2 and I1_H3
            //
            // Similarly for I2: (and I3)  
            // 
            // I2_H1, I2_H2 and I2_H3
            // 
            // And similarly for H1 to the O layer
            // 
            // We can represent synapses between layers as a matrix - so we can represent this network as layerSize - 1 matrices
            // 
            // For example the I layer to H layer:
            //
            // I1_H1, I2_H1, I3_H1
            // I1_H2, I2_H2, I3_H2
            // I1_H3, I2_H3, I3_H3
            //
            // In general the matrix has Rows = Number of nuerons in current layer and Cols = Number of neurons in previous layer

            _synapseMatrices = new List<Matrix>();
            for (int i = 1; i < neuronLayers.Length; i++)
            {
                var numRows = neuronLayers[i];
                var numCols = neuronLayers[i - 1];
                double[,] weights = new double[numRows, numCols];
                Matrix m = new Matrix(weights);
                RandomizeWeights(m);
                _synapseMatrices.Add(m);
            }

        }


        /// <summary>
        /// Evaluate the network against a set of inputs.
        /// </summary>
        /// <returns>The fire.</returns>
        /// <param name="inputs">Inputs.</param>
        public double[] Fire(params double[] inputs)
        {
            // Validate inputs - must not be null, and the number of inputs must match the number of neurons in the input layer
            // We can get the number of neurons in the input layer via the number of columns in the first synapse matrix.
            if (inputs == null) throw new ArgumentNullException(nameof(inputs), $"{nameof(inputs)} is required.");
            if (inputs.Length != _synapseMatrices.First().NumberOfColumns) throw new ArgumentException($"{nameof(inputs)} must have {_synapseMatrices.First().NumberOfColumns} elements.", nameof(inputs));

            // Fire the network and get outputs from all layers
            var outputMatrices = FireInternal(inputs);

            // Return the final output as an array
            return outputMatrices.Last().Col(0);

        }


        /// <summary>
        /// Train the network against a set of inputs and targets.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="targets">Targets.</param>
        public void Train(double[] inputs, double[] targets)
        {
            // Validate inputs - must not be null, and the number of inputs must match the number of neurons in the input layer
            // We can get the number of neurons in the input layer via the number of columns in the first synapse matrix.
            if (inputs == null) throw new ArgumentNullException(nameof(inputs), $"{nameof(inputs)} must be non-null");
            if (inputs.Length != _synapseMatrices.First().NumberOfColumns) throw new ArgumentException($"{nameof(inputs)} must have {_synapseMatrices.First().NumberOfColumns} elements.", nameof(inputs));

            // Validate expected outputs - because of the sigmoid they must be >0 and <1
            if (targets == null) throw new ArgumentNullException(nameof(targets), $"{nameof(targets)} must be non-null");
            if (targets.Length != _synapseMatrices.Last().NumberOfRows) throw new ArgumentException($"{nameof(targets)} must have {_synapseMatrices.Last().NumberOfRows} elements.", nameof(targets));
            if (targets.Any(o => o <= 0 || o >= 1.0)) throw new ArgumentOutOfRangeException(nameof(targets), $"{nameof(targets)} must all be > 0 and < 1");

            // Fire the network and get outputs from all layers
            List<Matrix> outputMatrices = FireInternal(inputs);

            // Train the network
            // The initial error matrix is simply the target data - the actual results.
            var currentErrorMatrix = Matrix.NRow1ColFrom1DArray(targets).Subtract(outputMatrices.Last());
            // Loop through all our layers in reverse
            for (int i = _synapseMatrices.Count - 1; i >= 0; i--)
            {
                // The previous layers error matrix is the dot product of the current weights with the current errors
                // In essense we pass the errors backwards, scaled by the weights.
                // This is simplistic.
                // Ideally we'd like to normalise these errors by dividing them by the sum of all the weights contributing to the error 
                // But in practice the simpler merthod works just as well.
                //
                // I1       H1      O1
                // I2       H2      O2
                // I3       H3      O3
                //
                // H1_ERROR = ( H1_O1_WEIGHT * O1_ERROR ) / ( H1_O1_WEIGHT + H2_O1_WEIGHT + H3_O1_WEIGHT ) + 
                //            ( H1_O2_WEIGHT * O2_ERROR ) / ( H1_O2_WEIGHT + H2_O2_WEIGHT + H3_O2_WEIGHT ) + 
                //            ( H1_O3_WEIGHT * O3_ERROR ) / ( H1_O3_WEIGHT + H2_O3_WEIGHT + H3_O3_WEIGHT )
                //
                // Becomes
                //
                // H1_ERROR = ( H1_O1_WEIGHT * O1_ERROR ) + 
                //            ( H1_O2_WEIGHT * O2_ERROR ) + 
                //            ( H1_O3_WEIGHT * O3_ERROR )
                //
                var previousErrorMatrix = _synapseMatrices[i].Transpose().DotProduct(currentErrorMatrix);

                // We are using a gradient descent.
                // For each error E we want to find out dE/dW_jk - how the error E changes as the weight W_ik changes.
                // Once we have this we can increase or decrease the weight by 'moving down' the slope.
                //
                // The error / fitness function is the sum of the differences between the target and actual values squared, where that sum is over all the n​ output nodes.
                // We square so as not to have +ve and -ve errors cancel each other out.
                //
                // Ejk = Sigma_n (Tn - On)^2 but only where the weight Wkj links to an output
                //
                // Ejk = (Tk - Ok)^2
                //
                // dE/dWjk = dE/dk * dk/dWjk
                //
                // de/dWjk = 2(Tk - Ok) * dk/dWjk
                //
                // dE/dWjk = 2(Tk - Ok) * sigmoid(Sigma_j Wjk * Oj) * (1 - sigmoid(Sigma_j Wjk * Oj)) * Oj
                // 
                // We can throw away the initial constant 2 - we're only interested in the direction
                // The initial part then is simply our error, the second and third parts are current output matrices, and the final part is our previous output

                // weights += learningRate * ((errors * outputs * (1.0 - outputs) . previousOutputs.T)
                var currentOutputMatrix = outputMatrices[i + 1];
                var oneMinusCurrentOutputMatrix = currentOutputMatrix.ApplyFunction(d => 1 - d);
                var previousOutputMatrixTransposed = outputMatrices[i].Transpose();

                var delta = currentErrorMatrix.CellByCellProduct(currentOutputMatrix).CellByCellProduct(oneMinusCurrentOutputMatrix).DotProduct(previousOutputMatrixTransposed);
                delta = delta.ApplyFunction(d => d * _learningRate);
               
                _synapseMatrices[i] = _synapseMatrices[i].Add(delta);

                // And we'll use the previous layers error matrix as current next time around
                currentErrorMatrix = previousErrorMatrix;
            }
        }


        /// <summary>
        /// Run the network with the supplied inputs and return all outputs
        /// </summary>
        /// <returns>The internal.</returns>
        /// <param name="inputs">Inputs.</param>
        List<Matrix> FireInternal(double[] inputs)
        {
            // Build input matrices
            Matrix inputMatrix = Matrix.NRow1ColFrom1DArray(inputs);

            // Run the network - keeping track of all outputs
            List<Matrix> outputMatrices = new List<Matrix>
            {
                inputMatrix // Input is technically the output from the input layer...
            };

            // Now loop through all our layers
            foreach (var synapseMatrix in _synapseMatrices)
            {
                // The output from this layer is simply the dot product of the synapses with the previous output.
                // The sigmoid activation is then  applied.
                outputMatrices.Add(synapseMatrix.DotProduct(outputMatrices.Last()).ApplyFunction(Sigmoid));
            }

            return outputMatrices;
        }


        /// <summary>
        /// Provide initial values for weights
        /// </summary>
        /// <param name="m">M.</param>
        void RandomizeWeights(Matrix m)
        {
            // Determine the range of weights
            // If each node has N links into it, then the weight should be between +- 1/Sqrt(N) and should also avoid zero.
            // Number of links coming in is given by the number of neurons coming in - which is the number of columns in this matrix.
            // TODO - Consider a normal distribution
            double range = 1.0d / Math.Sqrt(m.NumberOfColumns);
            for (int r = 0; r < m.NumberOfRows; r++)
            {
                for (int c = 0; c < m.NumberOfColumns; c++)
                {
                    m.Data[r, c] = (2 * range * _r.NextDouble()) - range;
                }
            }
        }


        /// <summary>
        /// Runs the network in reverse.  From an ideal output we can get an idea of what the network considers to be an 'ideal' input.
        /// </summary>
        /// <returns>The fire.</returns>
        /// <param name="inputs">Inputs.</param>
        public double[] ReverseFire(params double[] inputs)
        {
            // Validate inputs - must not be null, and the number of inputs must match the number of neurons in the *output* layer
            // We can get the number of neurons in the *output* layer via the number of *rows* in the *last* synapse matrix.
            if (inputs == null) throw new ArgumentNullException(nameof(inputs), $"{nameof(inputs)} is required.");
            if (inputs.Length != _synapseMatrices.Last().NumberOfRows) throw new ArgumentException($"{nameof(inputs)} must have {_synapseMatrices.Last().NumberOfColumns} elements.", nameof(inputs));

            // Build input matrix
            Matrix inputMatrix = Matrix.NRow1ColFrom1DArray(inputs);

            // Run the network - keeping track of all outputs
            List<Matrix> outputMatrices = new List<Matrix>
            {
                inputMatrix
            };

            // Now loop through all our layers *backwards*
            // We need to transpose them for this to work.
            var transposedReversedMatrices = _synapseMatrices.Select(m => m.Transpose()).Reverse();

            foreach (var synapseMatrix in transposedReversedMatrices)
            {
                // Apply the logit function to the outputMatrix *before* synapse.DotProduct
                var outputMatrix = synapseMatrix.DotProduct(outputMatrices.Last().ApplyFunction(Logit));
                // Scale it back to 0.01 to .99
                outputMatrix = outputMatrix.Scale(0.01, 0.99);
                // # scale them back to 0.01 to .99
                //outputMatrix = outputMatrix.ApplyFunction(d => d - outputMatrix.Min());
                //outputMatrix = outputMatrix.ApplyFunction(d => d / outputMatrix.Max());
                //outputMatrix = outputMatrix.ApplyFunction(d => d * 0.98);
                //outputMatrix = outputMatrix.ApplyFunction(d => d + 0.01);

                outputMatrices.Add(outputMatrix);
            }

            // Return the final output as an array
            return outputMatrices.Last().Col(0);
        }


        public void Save(string fileName)
        {
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(fileName))
            {
                file.WriteLine($"{_learningRate}");

                List<int> layerSizes = new List<int>();
                foreach (var weights in _synapseMatrices)
                {
                    layerSizes.Add(weights.NumberOfColumns);
                }
                layerSizes.Add(_synapseMatrices.Last().NumberOfRows);
                file.WriteLine($"{string.Join(" ", layerSizes.Select(ls => ls.ToString()))}");

                foreach (var weights in _synapseMatrices)
                {
                    for (int r = 0; r < weights.NumberOfRows; r++)
                    {
                        List<double> currentRow = new List<double>();
                        for (int c = 0; c < weights.NumberOfColumns; c++)
                        {
                            currentRow.Add(weights.Data[r, c]);
                        }
                        file.WriteLine(string.Join(" ", currentRow));
                    }
                }
            }  
        }


        public static Network Load(string fileName)
        {
            using (System.IO.StreamReader file = new System.IO.StreamReader(fileName))
            {
                var learningRateString = file.ReadLine();
                double learningRate = double.Parse(learningRateString);

                string layerSizesString = file.ReadLine();
                List<int> layerSizes = new List<int>();
                foreach (var layerSize in layerSizesString.Split(' '))
                {
                    int currentLayerSize = int.Parse(layerSize);
                    layerSizes.Add(currentLayerSize);
                }

                var result = new Network(learningRate, layerSizes.ToArray());

                foreach(var weights in result._synapseMatrices)
                {
                    for (int r = 0; r < weights.NumberOfRows; r++)
                    {
                        var row = file.ReadLine();
                        int c = 0;
                        foreach(var ele in row.Split(' '))
                        {
                            if (!double.TryParse(ele, out double val)) throw new InvalidOperationException(); // TODO
                            weights.Data[r, c] = val;
                            c++;
                        }
                       
                    }
                }
                return result;
            }
        }


        /// <summary>
        /// The activation function
        /// </summary>
        /// <returns>The sigmoid.</returns>
        /// <param name="x">x.</param>
        static double Sigmoid(double x)
        {
            return 1.0d / (1.0d + Math.Exp(-x));
        }


        /// <summary>
        /// Inverse of the Sigmoid for running in reverse.
        /// </summary>
        /// <returns>The logit.</returns>
        /// <param name="x">The x coordinate.</param>
        static double Logit(double x)
        {
            return Math.Log(x / (1 - x));
        }
    }
}

