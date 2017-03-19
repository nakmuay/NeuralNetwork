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
            int[] layerSizes = { 2, 5, 2 };

            ActivationFunction[] activationFunction =  {
                                                        new None(),
                                                        new SigmoidTransferFunction(),
                                                        new SigmoidTransferFunction(),
                                                        new SigmoidTransferFunction(),
                                                        };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);
            //bpn.Write();

            double[] input = { 0.5, 0.25 };
            double[] output;
            double[] wantedOutput = { 0.25, 0.0625 };
            double learningRate = 0.01;
            int numberOfTrainingEpochs = 100000;

            double error;
            bpn.Run(input, out output);
            Console.WriteLine("Output before training: {0}", output[1]);

            for (int epoch = 0; epoch < numberOfTrainingEpochs; epoch++)
            {
                bpn.Train(input, wantedOutput, learningRate, out error);
            }

            bpn.Train(input, wantedOutput, learningRate, out error);
            Console.WriteLine("Error: {0}", error);

            bpn.Run(input, out output);
            Console.WriteLine("Output after training: {0}", output[1]);

            Console.ReadLine();
        }
    }
}
