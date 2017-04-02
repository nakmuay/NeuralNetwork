using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNetwork;
using System.IO;

namespace NeuralNetworkApp
{

    class Program
    {
        static void Main(string[] args)
        {
            // Create some training data
            double xMin = 0.0;
            int numSamples = 62;
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
            string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "NetworkData");
            Directory.CreateDirectory(outputFolder);
            string refFile = Path.Combine(outputFolder, "neural_net_reference.txt");
            data.TrySerialize(refFile);

            // Create net
            int[] layerSizes = { 1, 10, 10, 1 };
            IDoubleEvaluatable[] activationFunction =  {new None(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction()};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            // Write output before net has been trained
            IdentificationData beforeTrainData;
            bpn.Run(data, out beforeTrainData);
            string beforeTrainFile = Path.Combine(outputFolder, "neural_net_before_training.txt");
            beforeTrainData.TrySerialize(beforeTrainFile);

            // Create network trainer
            SimpleNetworkTrainer trainer = new SimpleNetworkTrainer(bpn, data);

            // Create trainging options
            TrainingOptions opt = new TrainingOptions();
            opt.LearningRate = 0.001;
            opt.Momentum = 0.0001;
            opt.MaxError = 1.0E-1;
            trainer.Train(opt);

            // Write output after net has been trained
            IdentificationData afterTrainData;
            bpn.Run(data, out afterTrainData);
            string afterTrainFile = Path.Combine(outputFolder, "neural_net_after_training.txt");
            afterTrainData.TrySerialize(afterTrainFile);


            Console.WriteLine("Training error: {0}", trainer.ErrorSum);
            Console.WriteLine("Training iterations: {0}", trainer.Iterations);
        }
    }
}
