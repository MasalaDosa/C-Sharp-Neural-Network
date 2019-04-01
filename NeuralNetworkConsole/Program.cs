using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork;

namespace NeuralNetworkConsole
{
    class Program
    {
        static void Main(string[] args)
        {

            if (!File.Exists(Path.GetFullPath(Path.Combine("MNIST", "mnist_train.csv"))) ||
               !File.Exists(Path.GetFullPath(Path.Combine("MNIST", "mnist_test.csv"))))
            {
                Console.WriteLine($"Please download http://www.pjreddie.com/media/files/mnist_train.csv and http://www.pjreddie.com/media/files/mnist_test.csv to {Path.GetFullPath("MNIST")}");
                Environment.Exit(0);
            }

            var choice = Prompt("Would you like to [T]rain & test the network, [L]oad & test the network, or [M]ind read (run the network in reverse)?");
            switch (choice)
            {
                case 'L':
                    LoadAndTest();
                    break;
                case 'T':
                    TrainAndTest();
                    break;
                case 'M':
                default:
                    MindRead();
                    break;
            }
       
            Console.WriteLine("Press enter to continue");
            Console.ReadLine();
        }


        static void LoadAndTest()
        {
            if(!File.Exists(Path.Combine("MNIST", "mnist_net.dat")))
            {
                Console.WriteLine("Please train the network before loading and testing");
                Environment.Exit(0);
            }
            var network = Network.Load(Path.Combine("MNIST", "mnist_net.dat"));
            EvaluateNetwork(network);
        }


        static void TrainAndTest()
        {
            // Each row in the training data comprises 785 comma separated values.
            // The first is an integer label (0---9)
            // The remainder represent a bitmap of 28 by 28 pixels
            // So the input layer is 28x28 or 784 neurons
            // The size of the next layer is arbitrary - but it should be smaller than the input
            // (we are trying to have the network find common patterns and hence simplify things)
            // The final output layers has 10 neurons - one for each possible digit.
            var network = new Network(0.1, new int[] { 784, 100, 10 });
            Train(network);
            network.Save(Path.Combine("MNIST", "mnist_net.dat"));
            EvaluateNetwork(network);
        }


        static void MindRead()
        {

            if (!File.Exists(Path.Combine("MNIST", "mnist_net.dat")))
            {
                Console.WriteLine("Please train the network before loading and 'mind reading'");
                Environment.Exit(0);
            }
            var network = Network.Load(Path.Combine("MNIST", "mnist_net.dat"));

            double[] inputs = new double[10];

            for (int i = 0; i < 10; i++)
            {
                inputs = inputs.Select(x => 0.01).ToArray();
                inputs[i] = 0.99;

                var outputs = network.ReverseFire(inputs);
                Console.WriteLine($"Reverse firing {i}");
                DisplayChar(outputs);
            }
        }


        static void Train(Network network, int epochs = 1)
        {
            Console.WriteLine("Training network...");
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                int count = 0;
                foreach (var line in File.ReadLines(Path.Combine("MNIST", "mnist_train.csv")))
                {
                    var lineData = line.Split(',');

                    // Set the targets - avoiding the impossible to achieve 0.0 and 1.0
                    double[] targets = new double[10];
                    targets = targets.Select(d => 0.01).ToArray();
                    targets[int.Parse(lineData[0])] = 0.99;

                    double[] inputs = new double[784];
                    for (int i = 1; i < lineData.Length; i++)
                    {
                        // Store a scaled version of the inputs
                        // Again avoiding 0.0 but we don't mind having a value of 1.
                        inputs[i - 1] = (double.Parse(lineData[i]) / 255d * 0.99) + 0.01;
                    }

                    network.Train(inputs, targets);

                    count++;

                    if (count % 1000 == 0) Console.WriteLine($"{DateTime.Now.ToShortTimeString()} {count} rows trained so far...");
                }
            }
            Console.WriteLine();
        }


        static void EvaluateNetwork(Network network)
        {
            Console.WriteLine("Evaluating network.");
            int count = 0;
            int correct = 0;
            foreach (var line in File.ReadLines(Path.Combine("MNIST", "mnist_test.csv")))
            {
                var lineData = line.Split(',');

                int label = int.Parse(lineData[0]);

                double[] inputs = new double[784];
                for (int i = 1; i < lineData.Length; i++)
                {
                    inputs[i - 1] = ((double.Parse(lineData[i]) / 255d) * 0.99) + 0.01;
                }

                var outputs = network.Fire(inputs);
                double max = 0;
                int networkResult = 0;
                for (int i = 0; i < outputs.Length; i++)
                {
                    if (outputs[i] > max)
                    {
                        max = outputs[i];
                        networkResult = i;
                    }
                }

                if (label == networkResult)
                {
                    correct += 1;
                }
                count += 1;

                if (count % 100 == 0)
                {
                    Console.WriteLine($"Network says {networkResult}.  Label is {label}");
                    DisplayChar(inputs);
                }
            }
            Console.WriteLine($"Overall success rate: {(double)correct / count}");
        }


        static char? Prompt(string question)
        {
            List<char> answers = new List<char>();
            for (int i = 0; i < question.Length - 2; i++)
            {
                if (question[i] == '[' && question[i + 2] == ']')
                {
                    answers.Add(question[i + 1].ToString().ToUpperInvariant()[0]);
                }
            }
            Console.WriteLine(question);
            if (answers.Any())
            {
                char a;
                do
                {

                    a = Console.ReadKey(true).KeyChar.ToString().ToUpperInvariant()[0];
                } while (!answers.Contains(a));
                return a;
            }
            return null;
        }


        static void DisplayChar(double[] inputs)
        {
            Console.WriteLine();
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    var d = inputs[y * 28 + x];
                    if (d >= 0.9) Console.Write("*");
                    else if (d >= 0.8) Console.Write("#");
                    else if (d >= 0.7) Console.Write("%");
                    else if (d >= 0.6) Console.Write("+");
                    else if (d >= 0.5) Console.Write("-");
                    else if (d >= 0.4) Console.Write(":");
                    else if (d >= 0.3) Console.Write("'");
                    else if (d >= 0.2) Console.Write("~");
                    else if (d >= 0.1) Console.Write(".");
                    else Console.Write(" ");
                }
                Console.WriteLine();
            }
        }
    }
}
