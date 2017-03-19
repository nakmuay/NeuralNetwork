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
            int[] layerSizes = { 1, 5, 1 };

            ActivationFunction[] activationFunction =  {
                                                        new None(),
                                                        new SigmoidTransferFunction(),
                                                        new SigmoidTransferFunction()
                                                        };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);
            //bpn.Write();

            double[] input = { 1.0 };
            double[] output;
            double[] wantedOutput = { 0.5 };
            double learningRate = 0.01;
            int numberOfTrainingEpochs = 10000;

            double error;
            bpn.Train(input, wantedOutput, learningRate, out error);
            Console.WriteLine("Error before training: {0}", error);

            for (int epoch = 0; epoch < numberOfTrainingEpochs; epoch++)
            {
                bpn.Train(input, wantedOutput, learningRate, out error);
               // Console.WriteLine("Error: {0}", error);
            }

            bpn.Train(input, wantedOutput, learningRate, out error);
            Console.WriteLine("Error after training: {0}", error);

            Console.ReadLine();
        }
    }
}
