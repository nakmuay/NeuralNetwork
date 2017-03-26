using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNetwork;

namespace NeuralNetworkApp
{

    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = {2, 2, 2};

            IDoubleEvaluatable[] activationFunction =  {
                                                        new None(),
                                                        new SigmoidTransferFunction(),
                                                        new SigmoidTransferFunction(),
                                                        };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            double[] input = {0.5, 0.25};
            double[] output;

            double[] wantedOutput = new double[input.Length];
            for (int i=0; i<input.Length; i++)
                wantedOutput[i] = Math.Sqrt(input[i]);

            double learningRate = 0.001;
            int numberOfTrainingEpochs = 1000000;

            double error;
            bpn.Run(input, out output);
            for (int i = 0; i < input.Length; i++)
                Console.WriteLine("Output before training[{0}]: {1}", i, output[i]);

            for (int epoch = 0; epoch < numberOfTrainingEpochs; epoch++)
            {
                bpn.Train(input, wantedOutput, learningRate, out error);
            }

            bpn.Train(input, wantedOutput, learningRate, out error);
            Console.WriteLine("Error: {0}", error);

            bpn.Run(input, out output);
            for (int i = 0; i < input.Length; i++)
                Console.WriteLine("Output after training[{0}]: {1}", i, output[i]);

            Console.ReadLine();
        }
    }
}
