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
            int numSamples = 100;
            double[][] input = new double[numSamples][];
            double[][] output = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                input[i] = new double[1];
                input[i][0] = xMin + i/100.0;

                output[i] = new double[1];
                output[i][0] = 1 / 2 * Math.Sin(2 * Math.PI * input[i][0]) + Math.Sin(2.5 * Math.PI * input[i][0])
                                + Math.Sin(3.5 * Math.PI * input[i][0]) + 1/20 * Math.Sin(5 * Math.PI * input[i][0]);
            }

            IdentificationData data = new IdentificationData(input, output);
            string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "NetworkData");
            Directory.CreateDirectory(outputFolder);
            string refFile = Path.Combine(outputFolder, "neural_net_reference.txt");
            data.TrySerialize(refFile);

            // Create net
            int[] layerSizes = { 1, 8, 8, 1 };
            IDoubleEvaluatable[] activationFunction =  {new None(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        new Linear()};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            // Write output before net has been trained
            IdentificationData beforeTrainData;
            bpn.Run(data, out beforeTrainData);
            string beforeTrainFile = Path.Combine(outputFolder, "neural_net_before_training.txt");
            beforeTrainData.TrySerialize(beforeTrainFile);

            // Create trainging options
            TrainingOptions opt = new TrainingOptions();
            /*
            opt.LearningRate = 0.01;
            opt.Momentum = 0.001;
            opt.MaxError = 1.0E-4;
            opt.MaxIterations = 10000;
            */

            // Create network trainer
            SimpleNetworkTrainer trainer = SimpleNetworkTrainer.Instance;
            TrainingInformation trainInfo = trainer.Train(bpn, data, opt);

            // Write output after net has been trained
            IdentificationData afterTrainData;
            bpn.Run(data, out afterTrainData);
            string afterTrainFile = Path.Combine(outputFolder, "neural_net_after_training.txt");
            afterTrainData.TrySerialize(afterTrainFile);

            trainInfo.WriteTrainingSummary();

            // Write training information to file
            string trainingInfoFile = Path.Combine(outputFolder, "neural_net_training_info.txt");
            trainInfo.TrySerialize(trainingInfoFile);

            Console.ReadLine();
        }
    }
}