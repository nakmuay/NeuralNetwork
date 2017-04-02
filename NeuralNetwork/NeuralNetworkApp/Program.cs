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
            int numSamples = 10;
            double[][] input = new double[numSamples][];
            double[][] output = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                input[i] = new double[1];
                input[i][0] = xMin + i/10.0;

                output[i] = new double[1];
                output[i][0] = Math.Pow(input[i][0], 2);
            }

            IdentificationData data = new IdentificationData(input, output);
            string refFile = "C:\\Users\\Martin\\Documents\\Visual Studio 2015\\Projects\\neural_net_reference.txt";
            data.TrySerialize(refFile);

            // Create net
            int[] layerSizes = { 1, 10, 10, 1 };
            IDoubleEvaluatable[] activationFunction =  {new None(),
                                                        new SigmoidActivationFunction(),
                                                        new SigmoidActivationFunction(),
                                                        new SigmoidActivationFunction()};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);

            // Write output before net has been trained
            IdentificationData beforeTrainData;
            bpn.Run(data, out beforeTrainData);
            string beforeTrainFile = "C:\\Users\\Martin\\Documents\\Visual Studio 2015\\Projects\\neural_net_before_training.txt";
            beforeTrainData.TrySerialize(beforeTrainFile);

            // Create network trainer
            SimpleNetworkTrainer trainer = new SimpleNetworkTrainer(bpn, data);

            // Create trainging options
            TrainingOptions opt = new TrainingOptions();
            opt.LearningRate = 0.01;
            opt.Momentum = 0.0001;
            opt.MaxError = 1.0E-3;
            trainer.Train(opt);

            // Write output after net has been trained
            IdentificationData afterTrainData;
            bpn.Run(data, out afterTrainData);
            string afterTrainFile = "C:\\Users\\Martin\\Documents\\Visual Studio 2015\\Projects\\neural_net_after_training.txt";
            afterTrainData.TrySerialize(afterTrainFile);


            Console.WriteLine("Training error: {0}", trainer.ErrorSum);
            Console.WriteLine("Training iterations: {0}", trainer.Iterations);
        }
    }
}
