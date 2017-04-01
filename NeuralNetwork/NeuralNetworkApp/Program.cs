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
            // Create some training data
            double xMin = 0.0;
            int numSamples = 63;
            double[][] input = new double[numSamples][];
            double[][] output = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                input[i] = new double[1];
                input[i][0] = xMin + i/10.0;

                output[i] = new double[1];
                output[i][0] = Math.Pow(input[i][0], 2.0);
            }

            IdentificationData data = new IdentificationData(input, output);

            // Create net
            int[] layerSizes = { 1, 5, 1 };
            IDoubleEvaluatable[] activationFunction =  {new None(),
                                                        new SigmoidActivationFunction(),
                                                        new Linear()};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            // Create network trainer
            SimpleNetworkTrainer trainer = new SimpleNetworkTrainer(bpn, data);

            // Create trainging options
            TrainingOptions opt = new TrainingOptions();
            opt.LearningRate = 0.01;
            opt.Momentum = 0.01;
            opt.MaxError = 1.0E-2;

            bpn.Write();
            trainer.Train(opt);

            Console.WriteLine("Training error: {0}", trainer.ErrorSum);
            Console.WriteLine("Training iterations: {0}", trainer.Iterations);

            Console.WriteLine();
            bpn.Write();

            Console.ReadLine();

        }
    }
}
