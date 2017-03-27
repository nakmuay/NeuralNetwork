﻿using System;
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
            int numSamples = 100;
            double[][] input = new double[numSamples][];
            double[][] output = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                input[i] = new double[1];
                input[i][0] = xMin + i/10.0;

                output[i] = new double[1];
                output[i][0] = Math.Sin(input[i][0]);
            }

            IdentificationData data = new IdentificationData(input, output);

            // Create net
            int[] layerSizes = { 1, 10, 10, 10, 1 };
            IDoubleEvaluatable[] activationFunction =  {
                                                        new None(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            // Create network trainer
            TrainingOptions opt = new TrainingOptions();
            opt.MaxError = 1.0E-6;
            //opt.MaxIterations = 200000;

            SimpleNetworkTrainer trainer = new SimpleNetworkTrainer(bpn, data);
            trainer.Train(opt);

            Console.WriteLine("Training error: {0}", trainer.Error);
            Console.WriteLine("Training iterations: {0}", trainer.Iterations);
            Console.ReadLine();

        }
    }
}
